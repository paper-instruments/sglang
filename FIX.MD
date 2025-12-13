# `pause_generation(mode="in_place_safe")`: Safe-Point Pause Without Draining

This document proposes a new pause mode, `in_place_safe`, to support **PipelineRL-style in-flight weight updates**.

## Why we need `in_place_safe` (the minimal, correct fix)

You can’t make `in_place` safe by “just taking the writer lock,” because SGLang holds a **reader lock for the entire lifetime of a streaming request**:

```python
# python/sglang/srt/managers/tokenizer_manager.py
async with self.model_update_lock.reader_lock:
    ...
    async for response in self._wait_one_response(obj, state, request):
        yield response
```

If an update path tried to acquire `writer_lock` as the barrier, it would wait until all streaming requests finish (effectively **draining**), which breaks PipelineRL.

`in_place_safe` is the smallest engine-level change that preserves PipelineRL semantics:

- **Pause without draining** (keep requests + KV alive)
- But **return only after a real safe point** (no in-flight forward kernels)

---

## Goal / contract for `mode="in_place_safe"`

When `/pause_generation` is called with `mode="in_place_safe"`:

- Stop scheduling new forwards (same as `in_place`)
- Wait until all ranks in the engine have **no in-flight forward kernels** (safe-point)
- Then return 200 OK
- Keep requests + KV alive (unlike `abort` / `retract`)

---

## Step-by-step patch (surgical) — files + exact changes

### 1) Extend the request schema: add the new mode

**File:** `python/sglang/srt/managers/io_struct.py`

Update `PauseGenerationReqInput.mode` to include `"in_place_safe"`.

Currently:

```python
class PauseGenerationReqInput(BaseReq):
    ...
    mode: Literal["abort", "retract", "in_place"] = "abort"
    def __post_init__(self):
        allowed = ["abort", "retract", "in_place"]
```

Change to:

- `Literal["abort", "retract", "in_place", "in_place_safe"]`
- `allowed = ["abort", "retract", "in_place", "in_place_safe"]`
- Add 2–3 docstring lines explaining `in_place_safe`:
  - “Like `in_place` (pause scheduling without draining), but blocks until the scheduler reaches a safe point (no in-flight forwards) before returning.”

### 2) Add an internal “pause ack” output type (minimal handshake)

**File:** `python/sglang/srt/managers/io_struct.py`

Add a new output dataclass (patterned after other `*ReqOutput` types):

- `PauseGenerationReqOutput(BaseReq)` with fields:
  - `success: bool`
  - `mode: str`
  - `message: str = ""` (optional)

This is an **internal scheduler→tokenizer message**, not a new public HTTP response schema.

### 3) Make the scheduler implement the safe-point, and emit the ack

**File:** `python/sglang/srt/managers/scheduler.py`

In `Scheduler.pause_generation(...)`:

- Treat `in_place_safe` exactly like `in_place` with respect to batch handling and pause bookkeeping.
- After pause bookkeeping, when `mode == "in_place_safe"`, add a safe-point barrier:

Safe-point barrier requirements:

- **Per-rank device sync:** `torch.get_device_module().synchronize()` (or `torch.cuda.synchronize()` on CUDA) so no kernels are in flight on that rank.
- **Cross-rank barrier:** `torch.distributed.barrier(group=self.cpu_group)` so the caller does not ACK before other ranks have synced.

Then return a `PauseGenerationReqOutput(success=True, mode=recv_req.mode, message="paused_safe")`.

Why this is minimal:

- No changes to scheduling logic beyond a sync+barrier *only* for `in_place_safe`
- Keeps existing `in_place` behavior unchanged
- Uses `cpu_group` that SGLang already configures for coordination

### 4) Make `TokenizerManager` wait for that ack (only for `in_place_safe`)

**File:** `python/sglang/srt/managers/tokenizer_manager.py`

#### 4a) Add a dispatcher handler

Add a handler mapping entry:

- `(PauseGenerationReqOutput, self._handle_pause_generation_req_output)`

#### 4b) Add tiny state for awaiting acks

In `TokenizerManager.__init__`, add fields:

- `_pause_safe_future: Optional[asyncio.Future]`
- `_pause_safe_expected: int` (set to `server_args.dp_size`)
- `_pause_safe_seen: int`
- `_pause_safe_failed: bool` (optional)

#### 4c) Update `pause_generation(...)` to block only for `in_place_safe`

Ensure `self.auto_create_handle_loop()` is called at the start so the receive loop is running.

Current behavior typically forwards non-abort to scheduler and returns immediately:

```python
if obj.mode != "abort":
    await self.send_to_scheduler.send_pyobj(obj)
```

Update flow:

- Set `self.is_pause = True` under `is_pause_cond` as today.
- If `obj.mode == "in_place_safe"`:
  - initialize `_pause_safe_future`, `_pause_safe_expected = self.server_args.dp_size`, `_pause_safe_seen = 0`
  - send the request to scheduler
  - `await asyncio.wait_for(_pause_safe_future, timeout=<small default like 5–10s>)`
  - if timeout: raise (so callers can quarantine the engine)
- Else if `obj.mode in ("in_place", "retract")`: keep current behavior (send and return)
- Else if `abort`: keep existing abort logic unchanged

#### 4d) Implement `_handle_pause_generation_req_output`

Each time a `PauseGenerationReqOutput` arrives:

- If no `_pause_safe_future`: ignore (or debug log)
- Increment `_pause_safe_seen`
- If any `success=False`: mark failed
- When `_pause_safe_seen == _pause_safe_expected`:
  - resolve the future (success if all true, else raise / set exception)

This supports `dp_size > 1`: the caller waits for all DP schedulers to ACK.

### 5) No HTTP-server changes required

`/pause_generation` already awaits `tokenizer_manager.pause_generation(obj)`, so once `TokenizerManager.pause_generation` blocks for `in_place_safe`, the HTTP endpoint becomes blocking automatically.

---

## What you gain with `in_place_safe`

- A real correctness contract for PipelineRL:
  - `pause(in_place_safe)` is a “stop-the-world-at-safe-point” equivalent, **without draining**.
- Miles/PipelineRL can remove ad-hoc safety probing and rely on a single bounded wait.

---

## Quick validation checklist (engine-side)

- Start 1–N long streaming requests.
- Call `/pause_generation` with `{ "mode": "in_place_safe" }`:
  - returns quickly (bounded by completing any in-flight forward + barrier)
  - streams remain connected but stop emitting tokens
- Apply `/update_weights_from_distributed` (or tensor update path)
- Call `/continue_generation`:
  - the **same** request IDs continue streaming tokens (KV/request state survived)

