from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

PDF_MAGIC = b"%PDF-"
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
PDF_URL_RE = re.compile(r'''["']([^"']+\.pdf(?:\?[^"']*)?)["']''', re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    url: str
    source: str


@dataclass(frozen=True)
class DownloadedPDF:
    path: Path
    url: str
    size: int


def _is_http_url(url: str) -> bool:
    return urlparse(url).scheme.lower() in {"http", "https"}


def _filename_from_url(url: str) -> str:
    name = unquote(Path(urlparse(url).path).name) or "document.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:180] or "document.pdf"


def _unique_destination(output_dir: Path, filename: str) -> Path:
    destination = output_dir / filename
    if not destination.exists():
        return destination

    stem, suffix = destination.stem, destination.suffix
    index = 2
    while (output_dir / f"{stem}-{index}{suffix}").exists():
        index += 1
    return output_dir / f"{stem}-{index}{suffix}"


def discover_candidates(page_url: str, html: str) -> list[Candidate]:
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, Candidate] = {}

    def add(raw_url: str | None, source: str) -> None:
        if not raw_url:
            return
        raw_url = raw_url.strip()
        if raw_url.startswith(("//", "/", "./", "../")) or not urlparse(raw_url).scheme:
            raw_url = urljoin(page_url, raw_url)
        if _is_http_url(raw_url):
            found.setdefault(raw_url, Candidate(raw_url, source))

    for tag_name, attr in (
        ("a", "href"),
        ("iframe", "src"),
        ("embed", "src"),
        ("object", "data"),
        ("source", "src"),
    ):
        for tag in soup.find_all(tag_name):
            value = tag.get(attr)
            if value and (".pdf" in value.lower() or tag_name != "a"):
                add(value, f"html:{tag_name}.{attr}")

    for meta in soup.find_all("meta"):
        key = str(meta.get("property") or meta.get("name") or "").lower()
        if any(token in key for token in ("pdf", "document", "og:url")):
            add(meta.get("content"), f"html:meta:{key}")

    for script in soup.find_all("script"):
        content = script.string or script.get_text(" ", strip=False)
        for match in PDF_URL_RE.finditer(content or ""):
            add(match.group(1), "html:inline-script")

    return list(found.values())


def _save_pdf_response(
    response: httpx.Response,
    output_dir: Path,
    *,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> DownloadedPDF | None:
    """Stream a response to disk, retaining it only if it is a valid, bounded PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    content_type = response.headers.get("content-type", "")
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = 0
        if declared_size > max_bytes:
            raise ValueError(f"PDF exceeds the {max_bytes:,}-byte download limit")

    iterator = response.iter_bytes()
    first = next(iterator, b"")
    if not first or not _looks_like_pdf(content_type, first[:1024]):
        return None

    destination = _unique_destination(output_dir, _filename_from_url(str(response.url)))
    temporary = destination.with_name(f".{destination.name}.part")
    total = 0
    try:
        with temporary.open("wb") as file:
            for chunk in chain((first,), iterator):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"PDF exceeds the {max_bytes:,}-byte download limit")
                file.write(chunk)

        with temporary.open("rb") as file:
            signature = file.read(1024)
        if not signature.lstrip().startswith(PDF_MAGIC):
            return None

        temporary.replace(destination)
        return DownloadedPDF(destination, str(response.url), total)
    finally:
        temporary.unlink(missing_ok=True)


def _looks_like_pdf(content_type: str, prefix: bytes) -> bool:
    return "application/pdf" in content_type.lower() or prefix.lstrip().startswith(PDF_MAGIC)


def download_pdf(
    client: httpx.Client,
    candidate: Candidate,
    output_dir: Path,
    *,
    timeout: float = 30,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> DownloadedPDF | None:
    with client.stream("GET", candidate.url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        return _save_pdf_response(response, output_dir, max_bytes=max_bytes)


def inspect_http(
    page_url: str,
    output_dir: Path,
    timeout: float = 30,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> list[DownloadedPDF]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PDFExtractor/0.1)"}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", page_url, timeout=timeout) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "application/pdf" in content_type.lower():
                result = _save_pdf_response(response, output_dir, max_bytes=max_bytes)
                return [result] if result else []
            html = response.read().decode(response.encoding or "utf-8", errors="replace")
            final_page_url = str(response.url)

        candidates = discover_candidates(final_page_url, html)
        results: list[DownloadedPDF] = []
        for candidate in candidates:
            try:
                result = download_pdf(
                    client, candidate, output_dir, timeout=timeout, max_bytes=max_bytes
                )
            except (httpx.HTTPError, OSError, ValueError) as exc:
                print(f"[skip] {candidate.url} ({exc})")
                continue
            if result:
                results.append(result)
                print(f"[found] {result.path} ({result.size:,} bytes; {candidate.source})")
        return results
