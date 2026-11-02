from types import SimpleNamespace

from app.qa import llm_client


def test_generate_calls_configured_openai_compatible_client(monkeypatch) -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content='{"answer":"ok","citation_ids":[]}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)
    result = llm_client.generate([{"role": "user", "content": "hi"}])
    assert result == '{"answer":"ok","citation_ids":[]}'
    assert captured["model"] == llm_client.settings.groq_model
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0
