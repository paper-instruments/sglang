"""Native Messages formats must reach the existing grammar control unchanged."""

import copy
import json
import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestAnthropicOutputFormat(CustomTestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {"marker": {"type": "string", "enum": ["SCHEMA_VALUE"]}},
            "required": ["marker"],
            "additionalProperties": False,
        }
        self.body = {
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Reply NOT_JSON"}],
        }
        self.serving = AnthropicServing(
            SimpleNamespace(
                supports_inline_system=True,
                supports_native_reasoning_history=lambda: True,
            )
        )

    def convert(self, body):
        request = AnthropicMessagesRequest.model_validate(copy.deepcopy(body))
        return self.serving._convert_to_chat_completion_request(request)

    def test_native_format_reaches_sampling_for_both_response_modes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                converted = self.convert(
                    {
                        **self.body,
                        "stream": stream,
                        "output_config": {
                            "format": {"type": "json_schema", "schema": self.schema}
                        },
                    }
                )
                self.assertEqual(converted.response_format.type, "json_schema")
                self.assertTrue(converted.response_format.json_schema.strict)
                sampling = converted.to_sampling_params(
                    stop=[], model_generation_config={}
                )
                self.assertEqual(json.loads(sampling["json_schema"]), self.schema)

    def test_missing_and_null_format_keep_existing_controls(self):
        for output in (None, {}, {"format": None}, {"effort": "high"}):
            with self.subTest(output=output):
                converted = self.convert({**self.body, "output_config": output})
                self.assertIsNone(converted.response_format)
                sampling = converted.to_sampling_params(
                    stop=[], model_generation_config={}
                )
                self.assertNotIn("json_schema", sampling)
                if output and "effort" in output:
                    self.assertEqual(converted.reasoning_effort, "high")

    def test_format_keeps_effort_and_history_projection_unchanged(self):
        body = {
            **self.body,
            "system": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
            "messages": [
                {"role": "user", "content": "Compute the adjustment"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Subtotal 137"},
                        {"type": "text", "text": "Checking"},
                        {"type": "tool_use", "id": "a", "name": "lookup", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "a",
                            "content": "Adjustment 5",
                        },
                    ],
                },
            ],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "output_config": {"effort": "high"},
        }
        before = copy.deepcopy(body)
        baseline = self.convert(body).model_dump()
        body["output_config"]["format"] = {"type": "json_schema", "schema": self.schema}
        converted = self.convert(body).model_dump()
        converted["response_format"] = None
        self.assertEqual(converted, baseline)
        body["output_config"].pop("format")
        self.assertEqual(body, before)

    def test_nested_schema_is_not_rewritten(self):
        schema = {
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"$ref": "#/$defs/item"}},
            },
            "$defs": {
                "item": {"type": "object", "properties": {"x": {"type": "integer"}}}
            },
        }
        converted = self.convert(
            {
                **self.body,
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            }
        )
        self.assertEqual(converted.response_format.json_schema.schema_, schema)

    def test_malformed_format_is_rejected_before_conversion(self):
        for invalid in (
            {"type": "invalid", "schema": {}},
            {"type": "json_schema"},
            {"type": "json_schema", "schema": []},
            "json_schema",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                self.convert({**self.body, "output_config": {"format": invalid}})


if __name__ == "__main__":
    unittest.main()
