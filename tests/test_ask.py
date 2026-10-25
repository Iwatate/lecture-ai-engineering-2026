"""ask.py の会話履歴テスト。"""

from app import ask as ask_module


class FakeUsageMetadata:
    prompt_token_count = 12
    candidates_token_count = 7


class FakeResponse:
    text = "これはテスト応答です。"
    usage_metadata = FakeUsageMetadata()


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        return FakeResponse()


last_client = {}


class FakeClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.models = FakeModels()
        last_client["client"] = self


def test_ask_keeps_history_with_previous_messages(monkeypatch):
    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "dummy-key")
    ask_module.CONVERSATION_HISTORY.clear()

    previous = [
        {"role": "user", "content": "前の質問"},
        {"role": "assistant", "content": "前の答え"},
    ]

    answer = ask_module.ask("次の質問", history=previous)

    assert answer == "これはテスト応答です。"
    assert ask_module.CONVERSATION_HISTORY[-2:] == [
        {"role": "user", "content": "次の質問"},
        {"role": "assistant", "content": answer},
    ]

    call = last_client["client"].models.calls[-1]
    assert call["model"] == ask_module.MODEL
    assert call["contents"][0]["role"] == "user"
    assert call["contents"][0]["parts"][0]["text"] == "前の質問"
    assert call["contents"][1]["role"] == "assistant"
    assert call["contents"][1]["parts"][0]["text"] == "前の答え"
    assert call["contents"][2]["role"] == "user"
    assert call["contents"][2]["parts"][0]["text"] == "次の質問"
