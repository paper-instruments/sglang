"""Messages tool controls must reach an enforceable native grammar or fail closed."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import xgrammar as xgr  # noqa: E402

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.function_call.function_call_parser import (  # noqa: E402
    FunctionCallParser,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAnthropicToolControls(CustomTestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {"code": {"type": "integer", "enum": [142]}},
            "required": ["code"],
            "additionalProperties": False,
        }
        self.body = {
            "model": "test",
            "max_tokens": 512,
            "messages": [
                {"role": "user", "content": "Call lookup twice with code 999"}
            ],
            "tools": [{"name": "lookup", "input_schema": self.schema}],
        }
        self.serving = AnthropicServing(
            SimpleNamespace(
                supports_inline_system=True,
                supports_native_reasoning_history=lambda: True,
                tool_call_parser="glm47",
            )
        )

    def convert(self, body):
        return self.serving._convert_to_chat_completion_request(
            AnthropicMessagesRequest.model_validate(copy.deepcopy(body))
        )

    def test_mapping_defaults_and_explicit_booleans(self):
        for strict in (False, True):
            for disable in (False, True):
                for choice in ("auto", "any", "tool", "none"):
                    with self.subTest(strict=strict, disable=disable, choice=choice):
                        body = copy.deepcopy(self.body)
                        body["tools"][0]["strict"] = strict
                        body["tool_choice"] = {
                            "type": choice,
                            "disable_parallel_tool_use": disable,
                        }
                        if choice == "tool":
                            body["tool_choice"]["name"] = "lookup"
                        converted = self.convert(body)
                        self.assertIs(converted.tools[0].function.strict, strict)
                        self.assertIs(converted.parallel_tool_calls, not disable)
        default = self.convert(self.body)
        self.assertFalse(default.tools[0].function.strict)
        self.assertTrue(default.parallel_tool_calls)

    def test_compiled_glm_grammars_enforce_schema_and_call_count(self):
        compiler = xgr.GrammarCompiler(
            xgr.TokenizerInfo(
                [bytes([i]) for i in range(256)], vocab_type=xgr.VocabType.RAW
            ),
            max_threads=1,
        )
        good = "<tool_call>lookup<arg_key>code</arg_key><arg_value>142</arg_value></tool_call>"
        bad = good.replace("142", "999")
        for strict in (False, True):
            for disable in (False, True):
                for choice in ("auto", "any", "tool"):
                    body = copy.deepcopy(self.body)
                    body["tools"][0]["strict"] = strict
                    body["tool_choice"] = {
                        "type": choice,
                        "disable_parallel_tool_use": disable,
                        **({"name": "lookup"} if choice == "tool" else {}),
                    }
                    request = self.convert(body)
                    constraint = FunctionCallParser(
                        request.tools, "glm47"
                    ).get_structure_constraint(
                        request.tool_choice,
                        parallel_tool_calls=request.parallel_tool_calls,
                    )
                    grammar = (
                        xgr.Grammar.from_ebnf(constraint[1])
                        if constraint[0] == "full_assistant_ebnf"
                        else xgr.Grammar.from_structural_tag(constraint[1])
                    )
                    compiled = compiler.compile_grammar(grammar)
                    with self.subTest(strict=strict, disable=disable, choice=choice):
                        self.assertTrue(
                            xgr.GrammarMatcher(compiled).accept_string(good)
                        )
                        if strict:
                            self.assertFalse(
                                xgr.GrammarMatcher(compiled).accept_string(bad)
                            )
                        self.assertEqual(
                            xgr.GrammarMatcher(compiled).accept_string(
                                "<tool_call>lookup</tool_call>"
                            ),
                            not strict,
                        )
                        if disable:
                            self.assertFalse(
                                xgr.GrammarMatcher(compiled).accept_string(good + good)
                            )
                        elif not (strict and choice == "tool"):
                            # XGrammar's named-tool format selects one call even
                            # when parallel calls are allowed, not required.
                            self.assertTrue(
                                xgr.GrammarMatcher(compiled).accept_string(good + good)
                            )

    def test_active_tools_and_output_format_are_rejected(self):
        for choice in (
            None,
            {"type": "auto"},
            {"type": "any"},
            {"type": "tool", "name": "lookup"},
        ):
            for strict in (False, True):
                body = copy.deepcopy(self.body)
                body["tools"][0]["strict"] = strict
                body["tool_choice"] = choice
                body["output_config"] = {
                    "format": {"type": "json_schema", "schema": self.schema}
                }
                with (
                    self.subTest(choice=choice, strict=strict),
                    self.assertRaisesRegex(
                        ValueError, "output_config.format.*active tools"
                    ),
                ):
                    self.convert(body)
                body["tool_choice"] = {"type": "none"}
                self.assertIsNotNone(self.convert(body).response_format)

    def test_missing_or_incapable_parser_is_rejected(self):
        for name in (None, "qwen25"):
            self.serving.openai_serving_chat.tool_call_parser = name
            body = {
                **self.body,
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            }
            with (
                self.subTest(parser=name),
                self.assertRaisesRegex(ValueError, "disable_parallel_tool_use"),
            ):
                self.convert(body)
        self.serving.openai_serving_chat.tool_call_parser = "glm47"
        body = copy.deepcopy(self.body)
        body["tools"][0]["strict"] = True
        with patch.object(
            FunctionCallParser, "get_structure_constraint", return_value=None
        ):
            with self.assertRaisesRegex(
                ValueError, "tools.*strict.*cannot be enforced"
            ):
                self.convert(body)

    def test_controls_require_json_booleans(self):
        for invalid in ("true", 1, None):
            body = copy.deepcopy(self.body)
            body["tools"][0]["strict"] = invalid
            with self.assertRaises(ValidationError):
                self.convert(body)
            body = {
                **self.body,
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": invalid},
            }
            with self.assertRaises(ValidationError):
                self.convert(body)


if __name__ == "__main__":
    unittest.main()
