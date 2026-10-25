"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask "1つ目の質問" "2つ目の質問"  # 会話履歴を維持する

会話履歴を持つ実装に更新し、連続した質問でも前の文脈が残るようにする。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import json
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

CONVERSATION_HISTORY: list[dict[str, str]] = []
DEFAULT_HISTORY_PATH = Path(".ask_history.json")


def _normalize_history(history: Any) -> list[dict[str, str]]:
    if history is None:
        return []
    normalized: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, str):
            normalized.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = item.get("content")
        if content is None:
            content = item.get("text")
        if content is None:
            parts = item.get("parts")
            if isinstance(parts, list):
                texts = []
                for part in parts:
                    if isinstance(part, str):
                        texts.append(part)
                    elif isinstance(part, dict):
                        texts.append(str(part.get("text", "")))
                content = "".join(texts)
        normalized.append({"role": role, "content": str(content or "")})
    return normalized


def _load_history(history_path: str | Path | None = None) -> list[dict[str, str]]:
    path = Path(history_path) if history_path else DEFAULT_HISTORY_PATH
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return _normalize_history(data)


def _save_history(history: list[dict[str, str]], history_path: str | Path | None = None) -> None:
    path = Path(history_path) if history_path else DEFAULT_HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _to_contents(history: list[dict[str, str]], question: str) -> list[dict[str, str | list[dict[str, str]]]]:
    contents: list[dict[str, str | list[dict[str, str]]]] = []
    for item in history:
        role = item.get("role", "user")
        content = item.get("content", "")
        contents.append({"role": role, "parts": [{"text": content}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    return contents


def ask(question: str, history: list[dict[str, str]] | None = None, history_path: str | Path | None = None) -> str:
    if history is None:
        history = list(CONVERSATION_HISTORY)
        if not history:
            history = _load_history(history_path)
        CONVERSATION_HISTORY[:] = history

    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=_to_contents(history, question))

    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    print(f"[input tokens] {input_tokens if input_tokens is not None else 'N/A'}")
    print(f"[output tokens] {output_tokens if output_tokens is not None else 'N/A'}")

    answer = response.text
    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    CONVERSATION_HISTORY[:] = updated_history
    if history_path is not None or history is None:
        _save_history(updated_history, history_path)

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="*", default=[DEFAULT_QUESTION])
    parser.add_argument("--reset-history", action="store_true", help="会話履歴をリセットする")
    parser.add_argument("--history-file", default=str(DEFAULT_HISTORY_PATH), help="会話履歴を保存する JSON ファイル")
    args = parser.parse_args()

    history_path = Path(args.history_file)
    if args.reset_history:
        CONVERSATION_HISTORY.clear()
        if history_path.exists():
            history_path.unlink()

    print(f"[model] {MODEL}")
    for index, question in enumerate(args.question, start=1):
        print(f"[質問 {index}/{len(args.question)}] {question}\n")
        try:
            print(ask(question, history_path=history_path))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)


if __name__ == "__main__":
    main()
