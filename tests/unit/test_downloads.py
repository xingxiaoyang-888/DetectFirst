import io
import tarfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from defectfirst.assets import download_http, extract_archive
from defectfirst.io import write_json


def test_server_ignoring_range_restarts_instead_of_appending(tmp_path):
    body = b"correct complete asset"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", "immutable")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/asset"
    destination = tmp_path / "asset.bin"
    (tmp_path / "asset.bin.part").write_bytes(b"incorrect partial")
    write_json(tmp_path / "asset.bin.part.json", {"url": url, "etag": "immutable"})
    try:
        download_http(url, destination)
        assert destination.read_bytes() == body
        destination.write_bytes(b"tampered cached asset")
        with pytest.raises(ValueError, match="Cached file"):
            download_http(url, destination)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_archive_traversal_is_rejected_before_extraction(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("../outside.txt")
        info.size = 1
        handle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "extract")
    assert not (tmp_path / "outside.txt").exists()
