"""Install the service-side admission barrier used by the C256 benchmark."""

from __future__ import annotations

import importlib.abc
import sys


_TARGET = "vllm.entrypoints.openai.completion.serving"
_ORIGINAL = """        return await self._with_kv_transfer_rejection_cleanup(
"""
_REPLACEMENT = """        await _tpu_daily_wait_decode_barrier(raw_request)
        return await self._with_kv_transfer_rejection_cleanup(
"""
_PRELUDE = """
import asyncio as _tpu_daily_asyncio

_TPU_DAILY_DECODE_BARRIERS = {}
_TPU_DAILY_DECODE_BARRIER_LOCK = None


async def _tpu_daily_wait_decode_barrier(raw_request):
    global _TPU_DAILY_DECODE_BARRIER_LOCK
    if raw_request is None:
        return
    headers = raw_request.headers
    group = headers.get("X-AIOS-DECODE-BARRIER")
    if not group:
        return
    size = int(headers.get("X-AIOS-DECODE-BARRIER-SIZE", "0"))
    if size <= 1:
        return
    timeout_s = float(headers.get("X-AIOS-DECODE-BARRIER-TIMEOUT-S", "900"))
    if _TPU_DAILY_DECODE_BARRIER_LOCK is None:
        _TPU_DAILY_DECODE_BARRIER_LOCK = _tpu_daily_asyncio.Lock()
    async with _TPU_DAILY_DECODE_BARRIER_LOCK:
        state = _TPU_DAILY_DECODE_BARRIERS.setdefault(
            group,
            {"count": 0, "size": size, "event": _tpu_daily_asyncio.Event()},
        )
        if state["size"] != size:
            raise RuntimeError(
                f"decode barrier {group!r} changed size "
                f"from {state['size']} to {size}"
            )
        state["count"] += 1
        if state["count"] >= size:
            state["event"].set()
        event = state["event"]
        count = state["count"]
    if count == 1 or count == size:
        print(
            f"[tpu-daily-decode-barrier] group={group} count={count}/{size}",
            flush=True,
        )
    await _tpu_daily_asyncio.wait_for(event.wait(), timeout=timeout_s)

"""


class _BarrierLoader(importlib.abc.Loader):
    def __init__(self, spec, loader):
        self.spec = spec
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        source = self.loader.get_source(self.spec.name)
        if source is None or source.count(_ORIGINAL) != 1:
            raise ImportError(
                "vLLM completion serving no longer matches the decode "
                "barrier insertion point"
            )
        code = compile(
            _PRELUDE + source.replace(_ORIGINAL, _REPLACEMENT, 1),
            self.spec.origin,
            "exec",
        )
        exec(code, module.__dict__)


class _BarrierFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if (
                spec is not None
                and isinstance(spec.loader, importlib.abc.SourceLoader)
            ):
                spec.loader = _BarrierLoader(spec, spec.loader)
                return spec
        return None


sys.meta_path.insert(0, _BarrierFinder())
print("[tpu-daily-decode-barrier] import hook enabled", file=sys.stderr)
