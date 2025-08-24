import io
import os
import tempfile
from providers.kimi import KimiModelProvider


def test_kimi_upload_file_reads_and_calls_client(monkeypatch):
    prov = KimiModelProvider(api_key="test-key")

    uploaded = {}

    class DummyClient:
        class files:
            @staticmethod
            def create(file, purpose):
                # Capture file-like input
                data = file.read()
                uploaded["data"] = data
                uploaded["purpose"] = purpose
                return type("Resp", (), {"id": "file_123"})

    # Patch client with minimal files API
    monkeypatch.setattr(prov, "_client", DummyClient(), raising=False)

    # Write temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"hello")
        tmp_path = tmp.name
    try:
        file_id = prov.upload_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert file_id == "file_123"
    assert uploaded["data"] == b"hello"
    assert uploaded["purpose"] == "assistants"

