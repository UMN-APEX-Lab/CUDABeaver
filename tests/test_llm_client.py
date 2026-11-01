"""Focused tests for LLM request parameter handling."""

from types import SimpleNamespace

from harness.llm_client import call_llm_json_action


class _FakeChatCompletions:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [
            {
                "content": '{"tool":"read_prompt","args":{}}',
                "reasoning_content": None,
                "finish_reason": "stop",
            }
        ])

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        msg = SimpleNamespace(
            content=response.get("content"),
            reasoning_content=response.get("reasoning_content"),
            reasoning=None,
        )
        choice = SimpleNamespace(message=msg, finish_reason=response.get("finish_reason", "stop"))
        return SimpleNamespace(choices=[choice])


class _FakeModels:
    def __init__(self, max_model_len):
        self.max_model_len = max_model_len
        self.calls = 0

    def list(self):
        self.calls += 1
        model = SimpleNamespace(
            id="mock-model",
            root="mock-model",
            max_model_len=self.max_model_len,
        )
        return SimpleNamespace(data=[model])


class _FakeClient:
    def __init__(self, max_model_len=512, responses=None):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(responses))
        self.models = _FakeModels(max_model_len)


def test_json_action_service_limit_uses_configured_context_window():
    client = _FakeClient()
    config = {
        "api": {
            "backend": "vllm",
            "model": "mock-model",
            "max_tokens": "service_limit",
            "max_model_len": 1000,
            "service_limit_chars_per_token": 1.0,
            "service_limit_reserve_tokens": 100,
            "service_limit_min_output_tokens": 1,
        }
    }
    messages = [{"role": "user", "content": "x" * 200}]

    resp = call_llm_json_action(client, config, messages)

    # Estimator counts len(role)+4 plus content: 4+4+200 = 208.
    assert client.chat.completions.calls[0]["max_tokens"] == 692
    assert resp["request_max_tokens"] == 692
    assert client.models.calls == 0


def test_json_action_service_limit_can_query_model_metadata():
    client = _FakeClient(max_model_len=512)
    config = {
        "api": {
            "backend": "vllm",
            "model": "mock-model",
            "max_tokens": "service_limit",
            "service_limit_chars_per_token": 1.0,
            "service_limit_reserve_tokens": 10,
            "service_limit_min_output_tokens": 1,
        }
    }
    messages = [{"role": "user", "content": "abcd"}]

    resp = call_llm_json_action(client, config, messages)

    # Estimator counts len(role)+4 plus content: 4+4+4 = 12.
    assert client.chat.completions.calls[0]["max_tokens"] == 490
    assert resp["request_max_tokens"] == 490
    assert client.models.calls == 1


def test_json_action_empty_content_uses_thinking_finalization():
    client = _FakeClient(
        responses=[
            {
                "content": None,
                "reasoning_content": "prior reasoning " * 20,
                "finish_reason": "length",
            },
            {
                "content": '{"tool":"read_code","args":{"full_file":true}}',
                "reasoning_content": "short final reasoning",
                "finish_reason": "stop",
            },
        ]
    )
    config = {
        "api": {
            "backend": "vllm",
            "model": "mock-model",
            "max_tokens": 65536,
            "temperature": 0,
            "finalization": {
                "enabled": True,
                "max_tokens": 8192,
                "reasoning_char_threshold": 120000,
                "reasoning_excerpt_chars": 80,
                "fallback_reasoning_as_content": False,
            },
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                }
            },
        }
    }
    messages = [{"role": "user", "content": "Return one JSON action."}]

    resp = call_llm_json_action(client, config, messages)

    assert len(client.chat.completions.calls) == 2
    assert client.chat.completions.calls[1]["max_tokens"] == 8192
    assert client.chat.completions.calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert client.chat.completions.calls[1]["messages"][-1]["content"].startswith(
        "FINALIZATION REQUEST FOR JSON ACTION"
    )
    assert resp["content"] == '{"tool":"read_code","args":{"full_file":true}}'
    assert resp["finalization_triggered"] is True
    assert resp["finalization_reason"] == "empty_content"
