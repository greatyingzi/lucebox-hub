"""
OpenAI-compatible HTTP server on top of test_dflash.

    pip install fastapi uvicorn transformers
    python3 scripts/server.py                 # serves on :8000

    curl http://localhost:8000/v1/chat/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model":"luce-dflash","messages":[{"role":"user","content":"hi"}],"stream":true}'

Drop-in for Open WebUI / LM Studio / Cline by setting
  OPENAI_API_BASE=http://localhost:8000/v1  OPENAI_API_KEY=sk-any

Streams tokens as Server-Sent Events using the OpenAI delta format.
Model reloads per request (~10 s first-token latency). A daemon-mode
binary that keeps the model resident is a planned follow-up.
"""
import argparse
import asyncio
import json
import os
import signal
import struct
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, field_validator
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = ROOT / "models" / "Qwen3.5-27B-Q4_K_M.gguf"
DEFAULT_DRAFT_ROOT = Path.home() / ".cache/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots"
DEFAULT_BIN = ROOT / "build" / "test_dflash"
DEFAULT_BUDGET = 22
DEFAULT_MAX_CTX = 65536
LOCAL_TOKENIZER = ROOT / "tokenizer"
MODEL_NAME = "qwen35-27b"


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


class ChatMessage(BaseModel):
    role: str
    content: str | list

    @field_validator("content", mode="before")
    @classmethod
    def flatten_multimodal(cls, v):
        if isinstance(v, list):
            parts = [p["text"] for p in v if isinstance(p, dict) and p.get("type") == "text"]
            return "\n".join(parts)
        return v


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[Tool] | None = None

    def effective_max_tokens(self) -> int:
        return self.max_completion_tokens or self.max_tokens


def build_app(target: Path, draft: Path, bin_path: Path, budget: int,
              max_ctx: int,
              tokenizer: AutoTokenizer, stop_ids: set[int]) -> FastAPI:
    app = FastAPI(title="Luce DFlash OpenAI server")

    @app.exception_handler(RequestValidationError)
    async def debug_validation(request: Request, exc: RequestValidationError):
        print(f"[dflash] 422 validation error: {exc.body}", flush=True)
        print(f"[dflash] details: {exc.errors()}", flush=True)
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "luce"}],
        }

    def _tokenize_prompt(req: ChatRequest) -> Path:
        msgs = [m.dict() for m in req.messages]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if req.tools:
            kwargs["tools"] = [t.function.dict() for t in req.tools]
        prompt = tokenizer.apply_chat_template(msgs, **kwargs)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        tmp = Path(tempfile.mkstemp(suffix=".bin")[1])
        with open(tmp, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp

    def _spawn(prompt_bin: Path, n_gen: int):
        out_bin = Path(tempfile.mkstemp(suffix=".bin")[1])
        r, w = os.pipe()
        cmd = [str(bin_path), str(target), str(draft), str(prompt_bin),
               str(n_gen), str(out_bin),
               "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={w}"]
        proc = subprocess.Popen(cmd, pass_fds=(w,),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        os.close(w)
        return proc, r, out_bin

    async def _read_tokens(proc, r_fd, n_gen):
        """Read streamed tokens from pipe without blocking the event loop."""
        loop = asyncio.get_event_loop()
        generated = 0
        # Start stderr forwarding in a thread
        def _log_reader():
            try:
                for line in proc.stdout:
                    print(f"  [dflash] {line.decode(errors='replace').rstrip()}", flush=True)
            except Exception:
                pass
        import threading
        log_t = threading.Thread(target=_log_reader, daemon=True)
        log_t.start()

        try:
            while generated < n_gen:
                b = await loop.run_in_executor(None, os.read, r_fd, 4)
                if not b or len(b) < 4:
                    break
                tok_id = struct.unpack("<i", b)[0]
                generated += 1
                if tok_id in stop_ids:
                    break
                yield tok_id
        finally:
            # Kill subprocess if still running
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            os.close(r_fd)
            log_t.join(timeout=2)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_bin = _tokenize_prompt(req)
        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                proc, r, _ = _spawn(prompt_bin, req.effective_max_tokens())
                head = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": MODEL_NAME,
                    "choices": [{"index": 0,
                                  "delta": {"role": "assistant"},
                                  "finish_reason": None}],
                }
                yield f"data: {json.dumps(head)}\n\n"
                try:
                    async for tok_id in _read_tokens(proc, r, req.effective_max_tokens()):
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created, "model": MODEL_NAME,
                            "choices": [{"index": 0,
                                          "delta": {"content": tokenizer.decode([tok_id])},
                                          "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                finally:
                    try: prompt_bin.unlink()
                    except Exception: pass
                tail = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {},
                                  "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(tail)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        proc, r, _ = _spawn(prompt_bin, req.effective_max_tokens())
        tokens = []
        async for tok in _read_tokens(proc, r, req.effective_max_tokens()):
            tokens.append(tok)
        try: prompt_bin.unlink()
        except Exception: pass
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0,
                      "completion_tokens": len(tokens),
                      "total_tokens": len(tokens)},
        })

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    ap.add_argument("--max-ctx", type=int, default=DEFAULT_MAX_CTX)
    args = ap.parse_args()

    if not args.bin.is_file():
        raise SystemExit(f"binary not found at {args.bin}")
    if not args.target.is_file():
        raise SystemExit(f"target GGUF not found at {args.target}")
    draft = resolve_draft(args.draft) if args.draft.is_dir() else args.draft
    if not draft.is_file():
        raise SystemExit(f"draft safetensors not found at {args.draft}")

    print("[dflash] Loading tokenizer ...", flush=True)
    tok_path = str(LOCAL_TOKENIZER) if LOCAL_TOKENIZER.is_dir() else "Qwen/Qwen3.5-27B"
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path, trust_remote_code=True)
    print("[dflash] Tokenizer ready.", flush=True)
    stop_ids = set()
    for s in ("<|im_end|>", "<|tool_call|>"):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: stop_ids.add(ids[0])

    app = build_app(args.target, draft, args.bin, args.budget, args.max_ctx,
                    tokenizer, stop_ids)

    import uvicorn
    print(f"[dflash] Starting uvicorn on http://{args.host}:{args.port}", flush=True)
    print(f"  target   = {args.target}")
    print(f"  draft    = {draft}")
    print(f"  max-ctx  = {args.max_ctx}")
    print("  Note: model loads on first request (~10s).", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
