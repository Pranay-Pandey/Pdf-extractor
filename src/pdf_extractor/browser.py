from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from .core import MAX_DOWNLOAD_BYTES, PDF_MAGIC, DownloadedPDF, _unique_destination


def inspect_browser(
    page_url: str,
    output_dir: Path,
    *,
    timeout: float = 30,
    chrome_path: str = "/usr/bin/google-chrome",
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> list[DownloadedPDF]:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Browser fallback requires Playwright. Install with: pip install -e '.[browser]'"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[DownloadedPDF] = []
    pdf_urls: set[str] = set()

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True, executable_path=chrome_path)
        except PlaywrightError as exc:
            raise RuntimeError(f"Could not launch Chrome at {chrome_path}: {exc}") from exc

        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        def capture(response) -> None:
            url = response.url
            headers = response.headers
            content_type = headers.get("content-type", "").lower()
            if "application/pdf" not in content_type and ".pdf" not in url.lower():
                return
            if 200 <= response.status < 300:
                pdf_urls.add(url)

        page.on("response", capture)
        try:
            page.goto(page_url, wait_until="domcontentloaded", timeout=timeout * 1000)
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout * 1000, 12000))
            except PlaywrightTimeoutError:
                pass
            page.wait_for_timeout(2500)

            # Fetch each discovered PDF as a fresh full GET, not as viewer byte ranges.
            for url in sorted(pdf_urls):
                try:
                    response = context.request.get(url, timeout=timeout * 1000)
                    if response.status != 200:
                        print(f"[browser skip] Full download returned HTTP {response.status}: {url}")
                        continue

                    headers = response.headers
                    content_length = headers.get("content-length")
                    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                        print(f"[browser skip] {url} exceeds the {max_bytes:,}-byte download limit")
                        continue

                    body = response.body()
                    if len(body) > max_bytes:
                        print(f"[browser skip] {url} exceeds the {max_bytes:,}-byte download limit")
                        continue
                    if content_length and content_length.isdigit() and len(body) != int(content_length):
                        print(f"[browser skip] Incomplete response for {url}")
                        continue
                    if not body.lstrip().startswith(PDF_MAGIC):
                        print(f"[browser skip] Response is not a PDF: {url}")
                        continue

                    name = unquote(Path(urlparse(url).path).name) or f"captured-{len(results) + 1}.pdf"
                    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
                    if not name.lower().endswith(".pdf"):
                        name += ".pdf"
                    destination = _unique_destination(output_dir, name or "document.pdf")
                    destination.write_bytes(body)
                    results.append(DownloadedPDF(destination, url, len(body)))
                    print(f"[browser found] {destination} ({len(body):,} bytes)")
                except (PlaywrightError, OSError, ValueError) as exc:
                    print(f"[browser skip] Could not download {url}: {exc}")

            print(f"[browser] Inspected {page.url}; title: {page.title()}")
        except PlaywrightError as exc:
            raise RuntimeError(f"Browser inspection failed: {exc}") from exc
        finally:
            context.close()
            browser.close()

    return results
