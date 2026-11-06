import httpx
import pytest

from pdf_extractor.core import Candidate, discover_candidates, download_pdf


class ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks

    def close(self):
        pass


def make_client(chunks: list[bytes], headers: dict[str, str] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=headers or {"content-type": "application/pdf"},
            stream=ChunkStream(chunks),
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_discovers_pdf_links_and_embeds():
    html = '''
    <a href="/files/paper.pdf">Download</a>
    <iframe src="https://cdn.example.org/viewer?id=1"></iframe>
    <script>const documentUrl = "/assets/answer.pdf?token=abc";</script>
    '''
    candidates = discover_candidates("https://example.org/page", html)
    urls = {candidate.url for candidate in candidates}
    assert "https://example.org/files/paper.pdf" in urls
    assert "https://cdn.example.org/viewer?id=1" in urls
    assert "https://example.org/assets/answer.pdf?token=abc" in urls


def test_downloads_valid_pdf(tmp_path):
    with make_client([b"%PDF-1.7\n", b"sample content"]) as client:
        result = download_pdf(
            client, Candidate("https://example.org/paper.pdf", "test"), tmp_path
        )

    assert result is not None
    assert result.path.name == "paper.pdf"
    assert result.path.read_bytes() == b"%PDF-1.7\nsample content"
    assert result.size == len(b"%PDF-1.7\nsample content")


def test_rejects_html_response(tmp_path):
    with make_client(
        [b"<html>not a PDF</html>"], {"content-type": "text/html"}
    ) as client:
        result = download_pdf(
            client, Candidate("https://example.org/page", "test"), tmp_path
        )

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_size_limit_includes_first_chunk_and_cleans_partial_file(tmp_path):
    with (
        make_client([b"%PDF-123456", b"more data"]) as client,
        pytest.raises(ValueError, match="download limit"),
    ):
        download_pdf(
            client,
            Candidate("https://example.org/large.pdf", "test"),
            tmp_path,
            max_bytes=5,
        )

    assert list(tmp_path.iterdir()) == []


def test_existing_file_is_not_overwritten(tmp_path):
    existing = tmp_path / "paper.pdf"
    existing.write_bytes(b"keep this file")

    with make_client([b"%PDF-1.7\nnew document"]) as client:
        result = download_pdf(
            client, Candidate("https://example.org/paper.pdf", "test"), tmp_path
        )

    assert result is not None
    assert existing.read_bytes() == b"keep this file"
    assert result.path.name == "paper-2.pdf"
