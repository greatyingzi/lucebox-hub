// Load the DFlash draft model from a GGUF file (any quantized format).
//
// Supports F16, Q4_0, Q8_0, and any ggml quant type. The graph builder
// (qwen3_dflash_graph.cpp) uses ggml_mul_mat which handles quantized weights
// natively on CUDA, so no changes are needed there.
//
// Tensor naming follows the convention established by convert_dflash_to_gguf.py:
//   dflash.fc.weight, dflash.hidden_norm.weight, output_norm.weight,
//   blk.{i}.attn_norm.weight, blk.{i}.attn_q.weight, etc.

#include "internal.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash27b {

namespace {

// RAII mmap helper (same pattern as gguf_target_loader.cpp).
struct Mmap {
    void *  addr = nullptr;
    size_t  len  = 0;
#if defined(_WIN32)
    HANDLE  hFile = INVALID_HANDLE_VALUE;
    HANDLE  hMap  = nullptr;
#else
    int     fd   = -1;
#endif

    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            err = "CreateFileA: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) {
            err = "GetFileSizeEx: error " + std::to_string(GetLastError());
            return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            err = "CreateFileMappingA: error " + std::to_string(GetLastError());
            return false;
        }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) {
            err = "MapViewOfFile: error " + std::to_string(GetLastError());
            return false;
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + ": " + std::strerror(errno); return false; }
        struct stat st;
        if (::fstat(fd, &st) < 0) { err = "fstat: " + std::string(std::strerror(errno)); return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap: " + std::string(std::strerror(errno)); addr = nullptr; return false; }
#endif
        return true;
    }
    void release() {
        addr = nullptr; len = 0;
#if defined(_WIN32)
        hFile = INVALID_HANDLE_VALUE; hMap = nullptr;
#else
        fd = -1;
#endif
    }
    ~Mmap() {
#if defined(_WIN32)
        if (addr)                        UnmapViewOfFile(addr);
        if (hMap)                        CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

bool expect_u32(const gguf_context * g, const char * key, uint32_t expected, std::string & err) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) { err = std::string("missing gguf key: ") + key; return false; }
    uint32_t v = gguf_get_val_u32(g, id);
    if (v != expected) {
        char b[256];
        std::snprintf(b, sizeof(b), "gguf key %s=%u expected %u", key, v, expected);
        err = b;
        return false;
    }
    return true;
}

} // namespace

bool load_draft_gguf(const std::string & path,
                     ggml_backend_t       backend,
                     DraftWeights &       out) {

    // ── 1. Parse GGUF metadata + create ggml context ──────────────────
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    // ── 2. Validate arch + hyperparameters ────────────────────────────
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("missing general.architecture");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "qwen35-dflash-draft") {
            set_last_error(std::string("unexpected draft arch: ") + arch +
                           " (expected qwen35-dflash-draft)");
            gguf_free(gctx);
            return false;
        }
    }

    {
        std::string err;
        auto check = [&](const char * key, uint32_t expected) -> bool {
            return expect_u32(gctx, key, expected, err);
        };
        if (!check("qwen35-dflash-draft.block_count",              DFLASH27B_DRAFT_LAYERS) ||
            !check("qwen35-dflash-draft.embedding_length",         DFLASH27B_TARGET_HIDDEN) ||
            !check("qwen35-dflash-draft.feed_forward_length",      DFLASH27B_TARGET_INTERMEDIATE) ||
            !check("qwen35-dflash-draft.attention.head_count",     DFLASH27B_TARGET_N_HEADS) ||
            !check("qwen35-dflash-draft.attention.head_count_kv",  DFLASH27B_TARGET_N_KV_HEADS) ||
            !check("qwen35-dflash-draft.attention.key_length",     DFLASH27B_TARGET_HEAD_DIM) ||
            !check("qwen35-dflash-draft.attention.value_length",   DFLASH27B_TARGET_HEAD_DIM)) {
            set_last_error(err);
            gguf_free(gctx);
            return false;
        }
    }

    out.ctx     = meta_ctx;
    out.backend = backend;

    const int N_LAYER = DFLASH27B_DRAFT_LAYERS;
    out.layers.assign(N_LAYER, DraftLayer{});

    // ── 3. Wire tensor pointers by GGUF name ──────────────────────────
    auto g = [&](const char * name) -> ggml_tensor * {
        return ggml_get_tensor(meta_ctx, name);
    };

    // Top-level
    out.fc          = g("dflash.fc.weight");
    out.hidden_norm = g("dflash.hidden_norm.weight");
    out.out_norm    = g("output_norm.weight");

    if (!out.fc || !out.hidden_norm || !out.out_norm) {
        set_last_error("missing top-level draft tensors "
                       "(dflash.fc.weight / dflash.hidden_norm.weight / output_norm.weight)");
        gguf_free(gctx);
        return false;
    }

    // Per-layer
    for (int il = 0; il < N_LAYER; il++) {
        char name[128];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };

        DraftLayer & L = out.layers[il];
        L.attn_norm = fnd("attn_norm.weight");
        L.ffn_norm  = fnd("ffn_norm.weight");
        L.wq        = fnd("attn_q.weight");
        L.wk        = fnd("attn_k.weight");
        L.wv        = fnd("attn_v.weight");
        L.wo        = fnd("attn_output.weight");
        L.q_norm    = fnd("attn_q_norm.weight");
        L.k_norm    = fnd("attn_k_norm.weight");
        L.w_gate    = fnd("ffn_gate.weight");
        L.w_up      = fnd("ffn_up.weight");
        L.w_down    = fnd("ffn_down.weight");

        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.w_gate || !L.w_up || !L.w_down) {
            char b[128];
            std::snprintf(b, sizeof(b), "layer %d: missing tensor(s)", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }
    }

    // ── 4. Allocate backend buffer ────────────────────────────────────
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed (draft GGUF)");
        gguf_free(gctx);
        return false;
    }

    // ── 5. mmap + upload tensor bytes ─────────────────────────────────
    Mmap mm;
    {
        std::string err;
        if (!mm.open_ro(path, err)) {
            set_last_error(err);
            gguf_free(gctx);
            return false;
        }
    }

    const size_t data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    size_t total_bytes = 0;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (off + sz > mm.len) {
            set_last_error(std::string("tensor '") + tname + "' overflows file");
            gguf_free(gctx);
            return false;
        }
        ggml_backend_tensor_set(t, (const uint8_t *)mm.addr + off, 0, sz);
        total_bytes += sz;
    }

    gguf_free(gctx);

    // Print summary via set_last_error (same pattern as gguf_target_loader.cpp).
    {
        // Find the quant type of the first projection tensor for reporting.
        const char * quant_name = ggml_type_name(out.layers[0].wq->type);
        char summary[256];
        std::snprintf(summary, sizeof(summary),
            "draft loaded: %" PRId64 " tensors on GPU %.2f GiB (%s)",
            n_tensors, total_bytes / (1024.0 * 1024.0 * 1024.0),
            quant_name ? quant_name : "?");
        set_last_error(summary);
    }

    return true;
}

} // namespace dflash27b
