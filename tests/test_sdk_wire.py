"""Wire-level tests: the real vendor SDK against a local mock endpoint.

Every other LLM test injects a fake client object, which skips the SDK
entirely. These run the genuine ``openai`` SDK against a throwaway HTTP server
so the request we actually put on the wire is verified: the endpoint path, the
auth header, the model id and the strict-JSON schema.

Skipped when the optional SDK is not installed. No network and no API key are
required -- the mock accepts any bearer token.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src import config, llm_analyzers as la
from tests.fake_llm import verdict_payload

pytest.importorskip("openai", reason="optional LLM SDK not installed")


class _Handler(BaseHTTPRequestHandler):
    captured: dict = {}

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Handler.captured = {
            "path": self.path,
            "auth": self.headers.get("Authorization", ""),
            "body": json.loads(body),
        }
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "grok-4",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(verdict_payload(score=73))},
            }],
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def mock_endpoint(monkeypatch):
    """Serve an OpenAI-compatible endpoint on a free localhost port."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    _Handler.captured = {}
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


class TestXAIWireFormat:
    def _analyze(self, base_url, snapshot=None):
        analyzer = la.XAIAnalyzer(api_key="xai-test-key-not-real")
        analyzer.base_url = base_url
        from src.models import TokenSnapshot

        payload = la.build_analysis_payload(
            snapshot or TokenSnapshot(address="0xabc", chain="base", symbol="TEST")
        )
        return analyzer.analyze(payload), _Handler.captured

    def test_hits_the_openai_compatible_chat_endpoint(self, mock_endpoint):
        verdict, sent = self._analyze(mock_endpoint)
        assert verdict.ok is True
        assert sent["path"] == "/v1/chat/completions"

    def test_api_key_is_sent_as_a_bearer_token(self, mock_endpoint):
        _, sent = self._analyze(mock_endpoint)
        assert sent["auth"] == "Bearer xai-test-key-not-real"

    def test_requests_strict_json_schema_on_the_wire(self, mock_endpoint):
        _, sent = self._analyze(mock_endpoint)
        response_format = sent["body"]["response_format"]

        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert set(response_format["json_schema"]["schema"]["properties"]) == set(
            la.VERDICT_SCHEMA["properties"]
        )

    def test_sends_the_configured_model_and_temperature(self, mock_endpoint):
        _, sent = self._analyze(mock_endpoint)
        assert sent["body"]["model"] == config.XAI_MODEL
        assert sent["body"]["temperature"] == config.LLM_TEMPERATURE

    def test_system_and_user_messages_are_both_sent(self, mock_endpoint):
        _, sent = self._analyze(mock_endpoint)
        messages = sent["body"]["messages"]

        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == la.ANALYST_SYSTEM_PROMPT
        assert "TOKEN DATA:" in messages[1]["content"]

    def test_response_is_parsed_into_a_verdict(self, mock_endpoint):
        verdict, _ = self._analyze(mock_endpoint)

        assert verdict.overall_score == 73
        assert verdict.provider == "xai"
        assert len(verdict.dimension_scores) == 6
        assert verdict.latency_ms > 0

    def test_ensemble_runs_end_to_end_over_real_http(self, mock_endpoint, monkeypatch):
        monkeypatch.setattr(config, "XAI_API_KEY", "xai-test-key-not-real")
        monkeypatch.setattr(config, "XAI_BASE_URL", mock_endpoint)

        from src.data_fetchers import snapshot_from_pair
        from tests.fixtures import TOKEN_ADDRESS, dexscreener_pair

        pair = dexscreener_pair()
        snapshot = snapshot_from_pair(pair, [pair], address=TOKEN_ADDRESS)
        result = la.run_ensemble(snapshot, providers=["xai"])

        assert result.ok is True
        assert result.consensus.overall_score == 73
        assert result.consensus.model_count == 1
