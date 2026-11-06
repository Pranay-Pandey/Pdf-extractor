from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .browser import inspect_browser
from .core import inspect_http


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find and download accessible PDFs from a webpage"
    )
    parser.add_argument("url", help="Webpage URL or direct PDF URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Download directory (default: ./output)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Always inspect the page using installed Chrome, even if HTTP finds PDFs",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Disable automatic browser fallback when HTTP discovery finds no PDFs",
    )
    parser.add_argument("--chrome", default="/usr/bin/google-chrome", help="Chrome executable path")
    parser.add_argument("--timeout", type=float, default=30, help="Timeout in seconds (default: 30)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    parsed = urlparse(args.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        print("Error: provide a valid http(s) URL")
        return 2
    if args.no_browser and args.browser:
        print("Error: --browser and --no-browser cannot be used together")
        return 2

    try:
        results = inspect_http(args.url, args.output_dir, args.timeout)
        should_inspect_browser = args.browser or (not results and not args.no_browser)
        if should_inspect_browser:
            if not results:
                print("[browser] HTTP discovery found no PDFs; trying browser inspection")
            results.extend(
                inspect_browser(
                    args.url,
                    args.output_dir,
                    timeout=args.timeout,
                    chrome_path=args.chrome,
                )
            )
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if not results:
        print("No downloadable PDF found.")
        return 1
    print(f"\nCompleted: {len(results)} PDF file(s) saved in {args.output_dir.resolve()}")
    return 0
