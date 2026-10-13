"""User stops (stop strings, stop_regex, stop_token_ids) delimit the final answer,
not the thinking. A stop the model quotes while reasoning must not end the request,
including one right before the think-end marker and one committed in the same
speculative step as the marker. Drives the real `Req.observe_answer_start` /
`Req.update_finish_state` / `Req.update_reasoning_tokens` in the scheduler's order,
the scheduler helper, and `DetokenizerManager.trim_matched_stop` with fakes; pure CPU."""

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

STOP_ID = 1
EOS_ID = 2
THINK_END = 3
USER_STOP_TOKEN = 4
ID_TO_TEXT = {
    STOP_ID: "STOP",
    EOS_ID: "",
    THINK_END: "</think>",
    USER_STOP_TOKEN: "#",
    **{i: chr(ord("a") + i % 26) for i in range(10, 40)},
}


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(ID_TO_TEXT[int(i)] for i in ids)


class _CharTokenizer:
    def encode(self, s, add_special_tokens=False):
        return list(range(len(s)))


def _make_req(*, reasoning=True, stop=None, stop_regex=None, stop_token_ids=None):
    sp = SamplingParams(
        max_new_tokens=1000, stop=stop, stop_regex=stop_regex, stop_token_ids=stop_token_ids
    )
    sp.normalize(tokenizer=_CharTokenizer())
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        eos_token_ids={EOS_ID},
        vocab_size=10_000,
        require_reasoning=reasoning,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", [])
    return req


def _commit(req, tokens):
    """One scheduler step, in the scheduler's order: append, locate the answer start,
    check finish, then count reasoning over the retained tokens only."""
    previous = len(req.output_ids)
    req.output_ids.extend(tokens)
    if req.require_reasoning:
        req.observe_answer_start(list(tokens), [THINK_END])
    req.update_finish_state(new_accepted_len=len(tokens))
    if req.require_reasoning:
        retained = list(req.output_ids[previous : req.finished_len])
        if retained:
            req.update_reasoning_tokens(retained, [THINK_END])


class TestStopAfterReasoning(unittest.TestCase):
    def test_stop_quoted_while_reasoning_does_not_finish(self):
        req = _make_req(stop=["STOP"])
        for token in (10, STOP_ID, 11):
            _commit(req, [token])
        self.assertFalse(req.finished())
        self.assertFalse(req.check_match_stop_str_prefix())  # No reasoning held back.

    def test_stop_right_before_think_end_does_not_finish(self):
        req = _make_req(stop=["STOP"])
        _commit(req, [10, STOP_ID])
        _commit(req, [THINK_END])
        self.assertFalse(req.finished())
        _commit(req, [12])
        self.assertFalse(req.finished())

    def test_think_end_and_answer_in_one_speculative_step(self):
        # Stop before the marker is reasoning; the one after it is the answer.
        req = _make_req(stop=["STOP"])
        _commit(req, [10, STOP_ID, THINK_END, 12, 13])
        self.assertFalse(req.finished())
        _commit(req, [14, STOP_ID, 15])
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, "STOP")
        self.assertEqual(req.finished_len, 7)  # Through the answer's STOP (index 6).

    def test_stop_in_answer_finishes_at_the_stop(self):
        req = _make_req(stop=["STOP"])
        _commit(req, [10, STOP_ID, 11, THINK_END])
        _commit(req, [12, 13])
        _commit(req, [STOP_ID, 14, 15])
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_len, 7)

    def test_stop_regex_is_scoped_to_the_answer(self):
        req = _make_req(stop_regex=["ST.P"])
        _commit(req, [10, STOP_ID, THINK_END, 12])
        self.assertFalse(req.finished())
        _commit(req, [STOP_ID])
        self.assertTrue(req.finished())

    def test_user_stop_token_waits_for_the_answer_but_eos_does_not(self):
        req = _make_req(stop_token_ids=[USER_STOP_TOKEN])
        _commit(req, [10, USER_STOP_TOKEN, 11])
        self.assertFalse(req.finished())
        _commit(req, [EOS_ID])
        self.assertTrue(req.finished())  # System EOS still ends a reasoning request.
        same_step = _make_req(stop_token_ids=[USER_STOP_TOKEN])
        _commit(same_step, [10, USER_STOP_TOKEN, THINK_END, 12])
        self.assertFalse(same_step.finished())  # Before the marker: still reasoning.
        answered = _make_req(stop_token_ids=[USER_STOP_TOKEN])
        _commit(answered, [10, THINK_END, 12, USER_STOP_TOKEN])
        self.assertTrue(answered.finished())
        self.assertEqual(answered.finished_reason.matched, USER_STOP_TOKEN)

    def test_requests_without_reasoning_are_unchanged(self):
        req = _make_req(reasoning=False, stop=["STOP"])
        _commit(req, [10, STOP_ID, 11])
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_len, 2)


    def test_reasoning_accounting_is_unchanged(self):
        req = _make_req(stop=["STOP"])
        _commit(req, [10, STOP_ID, THINK_END, 12, 13])
        self.assertEqual(req.reasoning_tokens, 3)  # Through the marker only.
        _commit(req, [STOP_ID, 14])
        self.assertTrue(req.finished())
        self.assertEqual(req.reasoning_tokens, 3)

    def test_without_a_think_end_marker_stops_apply_as_before(self):
        req = _make_req(stop=["STOP"])
        req.output_ids.extend([10, STOP_ID])
        req.observe_answer_start([10, STOP_ID], [])
        req.update_finish_state(new_accepted_len=2)
        self.assertTrue(req.finished())

    def test_scheduler_helper_uses_the_model_think_end_marker(self):
        stub = SimpleNamespace(
            model_config=SimpleNamespace(
                think_end_ids=[THINK_END], request_selectable_think_end_id_sequences=None
            )
        )
        req = _make_req(stop=["STOP"])
        req.output_ids.extend([10, THINK_END, 12])
        SchedulerBatchResultProcessor._observe_answer_start(stub, req, [10, THINK_END, 12])
        self.assertEqual(req._answer_start, 2)
        plain = _make_req(reasoning=False, stop=["STOP"])
        SchedulerBatchResultProcessor._observe_answer_start(stub, plain, [10])
        self.assertIsNone(plain._answer_start)


class TestTrimAtLastOccurrence(unittest.TestCase):
    def test_trim_keeps_the_answer_when_reasoning_quoted_the_stop(self):
        stub = SimpleNamespace(is_tool_call_parser_gpt_oss=False)
        output = "plan: say STOP at the end</think>The answer. STOP"
        trimmed = DetokenizerManager.trim_matched_stop(stub, output, {"matched": "STOP"}, False)
        self.assertEqual(trimmed, "plan: say STOP at the end</think>The answer. ")


if __name__ == "__main__":
    unittest.main()
