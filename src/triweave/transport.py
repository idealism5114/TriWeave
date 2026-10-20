"""Bound and archive ordinary provider HTTP requests without changing routes or access rules."""
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import copy
from pathlib import Path
import shutil
import time
import uuid
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from triweave.collect import append_log, utc_now


def public_url(url):
    parts = urlsplit(str(url))
    private = {"token", "api_key", "apikey", "password", "authorization", "access_token"}
    query = [(key, "REDACTED" if key.lower() in private else value)
             for key, value in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit((parts.scheme, parts.netloc.split("@")[-1], parts.path, urlencode(query), ""))


@contextmanager
def observed_http(root, run_id, timeout=20, pause=1.0):
    """Sequential pipeline only. Preserve response bytes, dates, status, URL and SHA256."""
    root = Path(root)
    log = root / "data/manifests" / run_id / "http_events.jsonl"
    archive = root / "data/raw" / run_id / "_http"
    patched = []
    last_request = [0.0]
    zip_snapshots = {}

    def wrap(original):
        def request(session, method, url, *args, **kwargs):
            # AKShare's CFFEX daily parser repeatedly requests the same monthly ZIP.
            # Reuse only this invocation's frozen successful GET response, never a new run's vintage.
            cache_key = public_url(url) if str(method).upper() == 'GET' and urlsplit(str(url)).path.lower().endswith('.zip') and not kwargs.get('params') else None
            if cache_key in zip_snapshots:
                response, evidence = zip_snapshots[cache_key]
                append_log(log, {**evidence, 'event':'HTTP_CACHE_REUSE', 'reused_utc':utc_now(),
                                 'note':'Same-run immutable ZIP snapshot; no upstream request made'})
                return copy.copy(response)
            wait = pause - (time.monotonic() - last_request[0])
            if wait > 0:
                time.sleep(wait)
            last_request[0] = time.monotonic()
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = timeout
            event = {"method": str(method), "url": public_url(url), "request_url":public_url(url), "at": utc_now()}
            try:
                response = original(session, method, url, *args, **kwargs)
            except Exception as exc:
                event.update({"event": "HTTP_ERROR", "error": repr(exc)})
                append_log(log, event)
                raise
            body = response.content
            if shutil.disk_usage(root).free < len(body) + 512 * 1024 ** 2:
                raise OSError("Insufficient free disk reserve for immutable HTTP archive")
            archive.mkdir(parents=True, exist_ok=True)
            path = archive / f"{uuid.uuid4().hex}.response.gz"
            with path.open("xb") as output:
                output.write(gzip.compress(body, mtime=0))
            event.update({"event": "HTTP_RESPONSE", "url": public_url(response.url or url),
                          "status_code": response.status_code,
                          "content_type": response.headers.get("Content-Type"),
                          "body_bytes": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
                          "archive": str(path.relative_to(root)),
                          "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "etag": response.headers.get("ETag"),
                          "last_modified": response.headers.get("Last-Modified")})
            append_log(log, event)
            response.raise_for_status()
            if cache_key and response.status_code == 200:
                zip_snapshots[cache_key] = (copy.copy(response),dict(event))
            return response
        return request

    try:
        for module in ("requests.sessions", "curl_cffi.requests.session"):
            try:
                session_class = importlib.import_module(module).Session
            except ImportError:
                continue
            original = session_class.request
            session_class.request = wrap(original)
            patched.append((session_class, original))
        yield
    finally:
        for session_class, original in reversed(patched):
            session_class.request = original
