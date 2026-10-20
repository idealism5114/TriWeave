import gzip
import json
import requests
import pytest

from triweave.transport import observed_http, public_url


def test_public_url_removes_credentials():
    value = public_url("https://user:secret@example.org/data?token=private&symbol=RB0")
    assert "secret" not in value and "private" not in value and "symbol=RB0" in value


def test_http_archival_timeout_and_revision_detection(tmp_path, monkeypatch):
    calls = []
    def fake(session, method, url, *args, **kwargs):
        calls.append(kwargs)
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = str(len(calls)).encode()
        return response
    monkeypatch.setattr(requests.sessions.Session, "request", fake)
    with observed_http(tmp_path, "test", timeout=2, pause=0):
        requests.get("https://example.org/data", timeout=None)
        requests.get("https://example.org/data")
    events = [json.loads(line) for line in (tmp_path / "data/manifests/test/http_events.jsonl").read_text().splitlines()]
    assert all(item["timeout"] == 2 for item in calls)
    assert events[0]["body_sha256"] != events[1]["body_sha256"]
    assert gzip.decompress((tmp_path / events[0]["archive"]).read_bytes()) == b"1"
    assert requests.sessions.Session.request is fake


def test_access_denial_is_archived_and_raised(tmp_path, monkeypatch):
    def fake(session, method, url, **kwargs):
        response = requests.Response()
        response.status_code = 403
        response.url = url
        response._content = b"access denied"
        return response
    monkeypatch.setattr(requests.sessions.Session, "request", fake)
    with observed_http(tmp_path, "denied", pause=0):
        with pytest.raises(requests.HTTPError):
            requests.get("https://example.org/data")
    event = json.loads((tmp_path / "data/manifests/denied/http_events.jsonl").read_text())
    assert event["status_code"] == 403
    assert gzip.decompress((tmp_path / event["archive"]).read_bytes()) == b"access denied"


def test_monthly_zip_is_requested_once_per_snapshot_context(tmp_path, monkeypatch):
    calls=[]
    def fake(session,method,url,**kwargs):
        calls.append(url)
        response=requests.Response()
        response.status_code=200
        response.url=url
        response._content=b'ZIP fixture bytes'
        return response
    monkeypatch.setattr(requests.sessions.Session,'request',fake)
    url='https://example.org/202310.zip'
    with observed_http(tmp_path,'zip',pause=0):
        assert requests.get(url).content==requests.get(url).content
    assert len(calls)==1
    events=[json.loads(line) for line in (tmp_path/'data/manifests/zip/http_events.jsonl').read_text().splitlines()]
    assert events[1]['event']=='HTTP_CACHE_REUSE'
    assert events[0]['archive']==events[1]['archive']
    with observed_http(tmp_path,'refresh',pause=0):
        requests.get(url)
    assert len(calls)==2  # A new snapshot must check the upstream source again.
