from __future__ import annotations

import base64

from basetruth.integrations import ollama


def test_parse_pan_response_content_handles_json_fence_and_aliases() -> None:
    content = """
    ```json
    {
      "name": "RAVI KUMAR",
      "fatherName": "MOHAN KUMAR",
      "pan": "abcde1234f",
      "dob": "01/02/1990"
    }
    ```
    """

    parsed = ollama.parse_pan_response_content(content)

    assert parsed == {
        "pan_number": "ABCDE1234F",
        "full_name": "RAVI KUMAR",
        "name": "RAVI KUMAR",
        "father_name": "MOHAN KUMAR",
        "date_of_birth": "01/02/1990",
    }


def test_extract_pan_details_with_ollama_uses_probe_and_normalizes_payload(monkeypatch) -> None:
    captured: dict = {}

    class _DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "message": {
                    "content": '{"full_name":"ANITA SHAH","father_name":"RAMESH SHAH","pan_number":"aaapa1234a","date_of_birth":"10/10/1991"}'
                }
            }

    def _fake_probe() -> tuple[str, list[str], list[str]]:
        return "http://localhost:11434", ["gemma4:latest"], ["http://localhost:11434"]

    def _fake_post(url: str, json: dict, timeout: tuple[int, int]):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _DummyResponse()

    monkeypatch.setattr(ollama, "probe_ollama", _fake_probe)
    monkeypatch.setattr(ollama.requests, "post", _fake_post)

    result = ollama.extract_pan_details_with_ollama(b"image-bytes")

    assert result["pan_number"] == "AAAPA1234A"
    assert result["full_name"] == "ANITA SHAH"
    assert result["father_name"] == "RAMESH SHAH"
    assert result["date_of_birth"] == "10/10/1991"
    assert result["engine"] == "gemma4_ollama"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["model"] == "gemma4:latest"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"][1]["images"] == [
        base64.b64encode(b"image-bytes").decode("ascii")
    ]