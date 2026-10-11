"""Unit tests for Spec V2 grammar truncation and reasoning token accounting.

Grammar truncation and output stops retain a prefix of accepted tokens.
Reasoning accounting must consume only that prefix, including across batches.
"""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import (
    REQUEST_REASONING_END_TOKEN_IDS_KEY,
    SamplingParams,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeGrammar:
    """Grammar stub that terminates after `terminate_after` accepted tokens."""

    def __init__(self, terminate_after: int):
        self.accepted = []
        self.finished = False
        self._terminate_after = terminate_after

    def accept_token(self, token_id: int):
        self.accepted.append(token_id)

    def is_terminated(self) -> bool:
        return len(self.accepted) >= self._terminate_after


class _FakeSpecAlgorithm:
    def is_none(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False


class _FakeForwardMode:
    def is_decode(self) -> bool:
        return True

    def is_extend(self) -> bool:
        return False


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.has_grammar = any(req.grammar is not None for req in reqs)
        self.forward_mode = _FakeForwardMode()
        self.spec_algorithm = _FakeSpecAlgorithm()


def _make_processor() -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        beam_coordinator=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _make_req(
    terminate_after: int | None = None, *, max_new_tokens=256, **sampling_kwargs
) -> Req:
    sp = SamplingParams(max_new_tokens=max_new_tokens, temperature=0, **sampling_kwargs)
    sp.normalize(None)
    req = Req(
        rid="r0",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
    )
    if terminate_after is not None:
        req.grammar = _FakeGrammar(terminate_after=terminate_after)
    req.kv.kv_committed_len = 0
    return req


def _make_result(num_draft_tokens, accept_lens, flat_tokens):
    return GenerationBatchResult(
        next_token_ids=torch.tensor(flat_tokens, dtype=torch.long),
        accept_lens=torch.tensor(accept_lens, dtype=torch.long),
        speculative_num_draft_tokens=num_draft_tokens,
    )


def _make_reasoning_req(max_new_tokens=256, **sampling_kwargs):
    req = _make_req(max_new_tokens=max_new_tokens, **sampling_kwargs)
    req.require_reasoning = True
    req.eos_token_ids = {2}
    return req


def _make_reasoning_processor():
    return replace(
        _make_processor(),
        model_config=SimpleNamespace(think_end_ids=[7, 8]),
        token_to_kv_pool_allocator=Mock(),
        metrics_reporter=Mock(num_generated_tokens=0, forward_ct_decode=0),
        output_streamer=Mock(),
    )


def _decode_reasoning_tokens(processor, req, tokens):
    batch = _FakeBatch([req])
    batch.return_logprob = False
    batch.batch_size = lambda: 1
    result = _make_result(len(tokens), [len(tokens)], tokens)
    # Exercise real normalization, append, finish and reasoning accounting;
    # cache release/checkpointing is outside this regression.
    with patch.object(
        SchedulerBatchResultProcessor, "_handle_finish_state_updated_req"
    ):
        processor.process_batch_result_decode(batch, result)


def _commit_disagg_handoff(
    req: Req,
    processor: SchedulerBatchResultProcessor,
    token_id: int,
    *,
    replayed_boundary: bool = False,
) -> None:
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.scheduler = SimpleNamespace(
        batch_result_processor=processor, kv_checksum_computer=None
    )
    queue.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    queue.metadata_buffers = SimpleNamespace(
        get_buf=lambda _: (
            torch.tensor([token_id], dtype=torch.long),
            torch.zeros(7, dtype=torch.long),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            torch.tensor([1], dtype=torch.long),
        )
    )
    req.bootstrap_host = "127.0.0.1"
    req.bootstrap_room = 1
    if replayed_boundary:
        req.pd_rebootstrap_forced_output_id = token_id
    decode_req = DecodeRequest(
        req=req,
        kv_receiver=SimpleNamespace(clear=lambda: None),
        metadata_buffer_index=0,
        is_rebootstrap=replayed_boundary,
    )

    queue._commit_transfer_to_req(decode_req)


class TestSpecV2GrammarTruncation(CustomTestCase):
    def test_resolve_truncates_after_grammar_completion(self):
        req = _make_req(terminate_after=2)
        proc = _make_processor()
        # stride=4, accept_len=3 -> proposed [101, 102, 103]; grammar finishes at 102.
        result = _make_result(4, [3], [101, 102, 103, 0])

        predict_tokens = proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        self.assertEqual(predict_tokens, [[101, 102]])
        # No pre-claim: commit the full retained run (no -1 refund).
        self.assertEqual(req.kv.kv_committed_len, 2)

    def test_resolve_keeps_all_when_grammar_not_terminated(self):
        req = _make_req(terminate_after=99)
        proc = _make_processor()
        result = _make_result(4, [3], [201, 202, 203, 0])

        predict_tokens = proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        self.assertEqual(predict_tokens, [[201, 202, 203]])
        self.assertEqual(req.kv.kv_committed_len, 3)


class TestReasoningTokenAccounting(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(enable_metrics=False)
        override.install()
        self.addCleanup(override.restore)

    def test_speculative_reasoning_counts_only_retained_prefix(self):
        cases = [
            ("length", 24, {}, [11, 12, 13, 14], 24),
            ("eos", 256, {}, [11, 2, 13, 14], 24),
            ("stop_token", 256, {"stop_token_ids": {3}}, [11, 3, 13, 14], 24),
            ("no_cutoff", 256, {}, [11, 12, 13, 14], 26),
        ]
        for name, cap, params, tokens, retained in cases:
            with self.subTest(name=name):
                req = _make_reasoning_req(cap, **params)
                processor = _make_reasoning_processor()
                _decode_reasoning_tokens(processor, req, [10] * 22)
                _decode_reasoning_tokens(processor, req, tokens)
                self.assertEqual(len(req.output_ids_through_stop), retained)
                self.assertEqual(req.reasoning_tokens, retained)

    def test_think_end_respects_speculative_cutoff(self):
        cases = [
            # Reasoning, end marker, answer, and EOS share an accepted batch.
            ([10, 7, 8, 11, 2, 12], 256, 5, 3, True, 0),
            # End straddles the cap: only its partial match survives.
            ([11, 7, 8, 12], 2, 2, 2, False, 1),
            # Discarded tokens must not start or complete a marker.
            ([11, 2, 7, 8], 256, 2, 2, False, 0),
        ]
        for tokens, cap, retained, reasoning, over, match_len in cases:
            with self.subTest(tokens=tokens, cap=cap):
                req = _make_reasoning_req(cap)
                _decode_reasoning_tokens(_make_reasoning_processor(), req, tokens)
                self.assertEqual(len(req.output_ids_through_stop), retained)
                self.assertEqual(req.reasoning_tokens, reasoning)
                self.assertEqual(req._is_reasoning_over, over)
                if not over:
                    self.assertEqual(req._think_end_match_len, match_len)

    def test_think_end_across_batches_at_cutoff(self):
        for tokens, over in (([8, 11], True), ([7, 8], False)):
            with self.subTest(tokens=tokens):
                req = _make_reasoning_req(3)
                processor = _make_reasoning_processor()
                _decode_reasoning_tokens(processor, req, [10, 7])
                _decode_reasoning_tokens(processor, req, tokens)
                self.assertEqual(len(req.output_ids_through_stop), 3)
                self.assertEqual(req.reasoning_tokens, 3)
                self.assertEqual(req._is_reasoning_over, over)
                if not over:
                    self.assertEqual(req._think_end_match_len, 1)

    def test_prefill_reasoning_counts_only_retained_token(self):
        for cap in (0, 1):
            with self.subTest(cap=cap):
                req = _make_reasoning_req(cap)
                batch = SimpleNamespace(
                    reqs=[req],
                    return_logprob=False,
                    return_hidden_states=False,
                    return_hidden_states_mode=CaptureHiddenMode.NULL,
                    spec_info=None,
                    decoding_reqs=None,
                    forward_mode=ForwardMode.EXTEND,
                    prefill_stats=None,
                )
                result = GenerationBatchResult(next_token_ids=torch.tensor([7]))
                with (
                    patch(
                        "sglang.srt.managers.scheduler_components."
                        "batch_result_processor.release_kv_cache"
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components."
                        "batch_result_processor.checkpoint_kv_cache"
                    ),
                ):
                    _make_reasoning_processor().process_batch_result_prefill(
                        batch, result
                    )
                self.assertEqual(len(req.output_ids_through_stop), cap)
                self.assertEqual(req.reasoning_tokens, cap)
                self.assertEqual(req._think_end_match_len, cap)

    def test_multi_token_end_can_span_decode_steps(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        processor = _make_processor()
        processor.model_config.think_end_ids = [7, 8]

        processor._maybe_update_reasoning_tokens(req, [10, 7])
        processor._maybe_update_reasoning_tokens(req, [8, 11])

        self.assertEqual(req.reasoning_tokens, 3)
        self.assertTrue(req._is_reasoning_over)

    def test_request_selected_end_ignores_other_closer(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        req.sampling_params.custom_params = {
            REQUEST_REASONING_END_TOKEN_IDS_KEY: [17, 18]
        }
        processor = _make_processor()
        processor.model_config.think_end_ids = [7, 8]
        processor.model_config.request_selectable_think_end_id_sequences = [
            [7, 8],
            [17, 18],
        ]

        # The global/default closer must not end a medium request.
        processor._maybe_update_reasoning_tokens(req, [10, 7])
        processor._maybe_update_reasoning_tokens(req, [8, 11])
        self.assertFalse(req._is_reasoning_over)

        processor._maybe_update_reasoning_tokens(req, [10, 17])
        processor._maybe_update_reasoning_tokens(req, [18, 11])

        self.assertEqual(req.reasoning_tokens, 7)
        self.assertTrue(req._is_reasoning_over)

    def test_disagg_handoff_can_start_multi_token_selected_end(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        req.sampling_params.custom_params = {
            REQUEST_REASONING_END_TOKEN_IDS_KEY: [17, 18]
        }
        processor = _make_processor()
        processor.model_config.request_selectable_think_end_id_sequences = [
            [7, 8],
            [17, 18],
        ]

        _commit_disagg_handoff(req, processor, 17)
        self.assertEqual(req.reasoning_tokens, 1)
        self.assertFalse(req._is_reasoning_over)

        processor._maybe_update_reasoning_tokens(req, 18)

        self.assertEqual(req.reasoning_tokens, 2)
        self.assertTrue(req._is_reasoning_over)

    def test_disagg_rebootstrap_does_not_recount_boundary(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        req.sampling_params.custom_params = {REQUEST_REASONING_END_TOKEN_IDS_KEY: [17]}
        processor = _make_processor()
        processor.model_config.request_selectable_think_end_id_sequences = [
            [7],
            [17],
        ]

        _commit_disagg_handoff(req, processor, 17, replayed_boundary=True)

        self.assertEqual(req.reasoning_tokens, 0)
        self.assertFalse(req._is_reasoning_over)


if __name__ == "__main__":
    unittest.main()
