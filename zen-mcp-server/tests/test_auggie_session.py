import json
from pathlib import Path

from auggie.session import get_store, create_session, get_session, update_session, append_history, delete_session
from auggie.config import load_auggie_config


def test_file_store_roundtrip(tmp_path):
    cfg = {"auggie": {"session": {"persistence": "file", "ttl": 1}}}
    p = tmp_path / "auggie-config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    load_auggie_config(p)

    sid = "s1"
    create_session(sid, {"user": "u"})
    update_session(sid, last_model="m1")
    append_history(sid, {"tool": "aug_chat", "prompt": "hi"})

    s = get_session(sid)
    assert s.get("last_model") == "m1" and s["history"] and s["meta"]["user"] == "u"

    delete_session(sid)
    assert get_session(sid) == {}

