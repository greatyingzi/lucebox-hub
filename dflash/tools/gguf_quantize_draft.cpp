// Minimal GGUF-to-GGUF quantizer for draft models.
// Bypasses llama-quantize's architecture check by directly using ggml quantize.
//
// Usage: gguf-quantize-draft <input.gguf> <output.gguf> <Q4_0|Q8_0|F16|...>

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <cassert>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <input.gguf> <output.gguf> <Q4_0|Q8_0|F16|...>\n", argv[0]);
        return 1;
    }
    const char * in_path  = argv[1];
    const char * out_path = argv[2];
    const char * qtype_str = argv[3];

    // Parse quant type from string (case-insensitive)
    ggml_type target_type = GGML_TYPE_COUNT;
    std::string input(qtype_str);
    for (auto & c : input) c = tolower(c);
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const char * tn = ggml_type_name((ggml_type)i);
        if (tn && input == tn) { target_type = (ggml_type)i; break; }
    }
    if (target_type == GGML_TYPE_COUNT) {
        fprintf(stderr, "unknown quant type: %s\n", qtype_str);
        return 1;
    }

    // Load source GGUF with tensor metadata (ne[] dims available via ggml_context)
    ggml_context * src_meta = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &src_meta;
    gguf_context * src_gctx = gguf_init_from_file(in_path, gip);
    if (!src_gctx) {
        fprintf(stderr, "failed to open: %s\n", in_path);
        return 1;
    }

    // mmap source file for raw tensor bytes
    FILE * fp = fopen(in_path, "rb");
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> file_data(file_size);
    if (fread(file_data.data(), 1, file_size, fp) != file_size) {
        fprintf(stderr, "failed to read file\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    const size_t data_offset = gguf_get_data_offset(src_gctx);
    const int64_t n_tensors  = gguf_get_n_tensors(src_gctx);
    const int64_t n_kv       = gguf_get_n_kv(src_gctx);

    // Create output GGUF writer
    gguf_context * dst_gctx = gguf_init_empty();

    // Copy all KV metadata
    for (int64_t i = 0; i < n_kv; i++) {
        const char * key = gguf_get_key(src_gctx, i);
        enum gguf_type vtype = gguf_get_kv_type(src_gctx, i);
        switch (vtype) {
            case GGUF_TYPE_STRING:
                gguf_set_val_str(dst_gctx, key, gguf_get_val_str(src_gctx, i));
                break;
            case GGUF_TYPE_UINT32:
                gguf_set_val_u32(dst_gctx, key, gguf_get_val_u32(src_gctx, i));
                break;
            case GGUF_TYPE_INT32:
                gguf_set_val_i32(dst_gctx, key, gguf_get_val_i32(src_gctx, i));
                break;
            case GGUF_TYPE_FLOAT32:
                gguf_set_val_f32(dst_gctx, key, gguf_get_val_f32(src_gctx, i));
                break;
            case GGUF_TYPE_UINT64:
                gguf_set_val_u64(dst_gctx, key, gguf_get_val_u64(src_gctx, i));
                break;
            case GGUF_TYPE_INT64:
                gguf_set_val_i64(dst_gctx, key, gguf_get_val_i64(src_gctx, i));
                break;
            case GGUF_TYPE_FLOAT64:
                gguf_set_val_f64(dst_gctx, key, gguf_get_val_f64(src_gctx, i));
                break;
            case GGUF_TYPE_BOOL:
                gguf_set_val_bool(dst_gctx, key, gguf_get_val_bool(src_gctx, i));
                break;
            case GGUF_TYPE_ARRAY: {
                enum gguf_type arr_type = gguf_get_arr_type(src_gctx, i);
                size_t arr_n = gguf_get_arr_n(src_gctx, i);
                if (arr_type == GGUF_TYPE_STRING) {
                    std::vector<const char *> strs(arr_n);
                    for (size_t j = 0; j < arr_n; j++)
                        strs[j] = gguf_get_arr_str(src_gctx, i, j);
                    gguf_set_arr_str(dst_gctx, key, strs.data(), arr_n);
                } else if (arr_type == GGUF_TYPE_UINT32) {
                    const uint32_t * data = (const uint32_t *)gguf_get_arr_data(src_gctx, i);
                    gguf_set_arr_data(dst_gctx, key, GGUF_TYPE_UINT32, data, arr_n);
                } else if (arr_type == GGUF_TYPE_INT32) {
                    const int32_t * data = (const int32_t *)gguf_get_arr_data(src_gctx, i);
                    gguf_set_arr_data(dst_gctx, key, GGUF_TYPE_INT32, data, arr_n);
                } else if (arr_type == GGUF_TYPE_FLOAT32) {
                    const float * data = (const float *)gguf_get_arr_data(src_gctx, i);
                    gguf_set_arr_data(dst_gctx, key, GGUF_TYPE_FLOAT32, data, arr_n);
                }
                break;
            }
            default:
                fprintf(stderr, "skipping KV '%s' with unsupported type %d\n", key, (int)vtype);
                break;
        }
    }

    // Helper to dequantize F16 to F32
    auto f16_to_f32 = [](const uint16_t * f16, float * f32, size_t n) {
        for (size_t i = 0; i < n; i++) {
            uint16_t h = f16[i];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            uint32_t f;
            if (exp == 0) {
                if (mant == 0) {
                    f = sign << 31;
                } else {
                    // subnormal
                    exp = 127 - 15 + 1;
                    mant <<= 23 - 10;
                    f = (sign << 31) | ((exp - 127 + 127) << 23) | (mant & 0x7fffff);
                }
            } else if (exp == 31) {
                f = (sign << 31) | (0xff << 23) | (mant << 13);
            } else {
                f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
            }
            memcpy(&f32[i], &f, 4);
        }
    };

    // Keep quantized buffers alive until gguf_write_to_file.
    // The data pointers in the gguf tensor info point into these vectors.
    std::vector<std::vector<uint8_t>> owned_buffers;

    // Process tensors
    size_t src_total = 0, dst_total = 0;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * name = gguf_get_tensor_name(src_gctx, tid);
        size_t offset = gguf_get_tensor_offset(src_gctx, tid);
        size_t tsize  = gguf_get_tensor_size(src_gctx, tid);
        enum ggml_type stype = gguf_get_tensor_type(src_gctx, tid);

        // Get source tensor from ggml_context to read ne[] dimensions
        ggml_tensor * src_t = ggml_get_tensor(src_meta, name);
        assert(src_t);

        src_total += tsize;

        // Keep F32 tensors (norms) as-is. Quantize everything else.
        ggml_type out_type = (stype == GGML_TYPE_F32) ? GGML_TYPE_F32 : target_type;

        const uint8_t * src_bytes = file_data.data() + data_offset + offset;

        if (stype == out_type) {
            // No conversion needed — add tensor with same type and dims, set data
            ggml_tensor tmp = *src_t;  // copy ne[], nb[], name
            gguf_add_tensor(dst_gctx, &tmp);
            gguf_set_tensor_data(dst_gctx, name, src_bytes);
            dst_total += tsize;
        } else {
            // Quantize: add tensor with new type (same dims), quantize, set data
            size_t n_el = ggml_nelements(src_t);

            // Dequantize source to F32
            std::vector<float> f32_data(n_el);
            if (stype == GGML_TYPE_F16) {
                f16_to_f32((const uint16_t *)src_bytes, f32_data.data(), n_el);
            } else if (stype == GGML_TYPE_F32) {
                memcpy(f32_data.data(), src_bytes, n_el * sizeof(float));
            } else {
                fprintf(stderr, "unsupported source type %s for tensor %s\n",
                        ggml_type_name(stype), name);
                return 1;
            }

            // Quantize to target type — store buffer in owned_buffers to keep alive
            size_t out_size = ggml_row_size(out_type, n_el);
            owned_buffers.emplace_back(out_size);
            uint8_t * qbuf = owned_buffers.back().data();
            ggml_quantize_chunk(out_type, f32_data.data(), qbuf, 0, 1, n_el, nullptr);

            // Add tensor with new type but same dimensions; recalculate nb[]
            ggml_tensor tmp = *src_t;
            tmp.type = out_type;
            tmp.nb[0] = ggml_type_size(out_type);
            tmp.nb[1] = ggml_row_size(out_type, tmp.ne[0]);
            for (int d = 2; d < GGML_MAX_DIMS; d++) {
                tmp.nb[d] = tmp.nb[d - 1] * tmp.ne[d - 1];
            }
            gguf_add_tensor(dst_gctx, &tmp);
            gguf_set_tensor_data(dst_gctx, name, qbuf);
            dst_total += out_size;
        }

        fprintf(stderr, "[q] %50s  %s -> %s  (%zu -> ", name,
                ggml_type_name(stype), ggml_type_name(out_type), tsize);
        size_t actual_out = (stype != out_type)
            ? ggml_row_size(out_type, ggml_nelements(src_t))
            : tsize;
        fprintf(stderr, "%zu)\n", actual_out);
    }

    // Write output
    fprintf(stderr, "[info] total: %.2f GiB -> %.2f GiB (%s)\n",
            src_total / (1024.0 * 1024.0 * 1024.0),
            dst_total / (1024.0 * 1024.0 * 1024.0),
            ggml_type_name(target_type));

    gguf_write_to_file(dst_gctx, out_path, false);
    gguf_free(dst_gctx);
    gguf_free(src_gctx);

    fprintf(stderr, "[done] wrote %s\n", out_path);
    return 0;
}
