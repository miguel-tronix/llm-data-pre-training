"""Tests for parallel download logic in download_utils.py.

The test suite focuses on the chunk-level functions that do the heavy lifting
(_download_chunk_attempt) and the merge logic (_merge_parts), plus end-to-end
integration against a local HTTP server that exercises the full parallel flow.
"""

import asyncio
import functools
import hashlib
import http.server
import socket
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from llm_data_pretraining.data_fetch.download_utils import (
    MB_100,
    DownloadConfig,
    HFDatasetDownloader,
    _download_chunk_attempt,
    _download_chunk_process,
)

HTTP_OK = 200
HTTP_PARTIAL = 206


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def base_config(tmp_dir: Path) -> DownloadConfig:
    return DownloadConfig(
        repo_id="test/dataset",
        raw_data_dir=tmp_dir / "rawdata",
        max_retries=2,
        timeout=10,
        chunk_size=8192,
        num_parallel_downloads=4,
    )


# ---------------------------------------------------------------------------
# Helper: a tiny HTTP server that serves a synthetic file of known content
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_content(size: int, seed: int = 42) -> bytes:
    """Deterministic pseudo-random content so we can hash-compare."""
    rng = bytearray(size)
    state = seed & 0xFFFFFFFF
    for i in range(size):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        rng[i] = (state >> 16) & 0xFF
    return bytes(rng)


class RangeHTTPHandler(http.server.BaseHTTPRequestHandler):
    """Simple handler that serves *content* and properly honours Range requests.

    Set ``server.content`` before starting.
    """

    content: bytes = b""

    def do_GET(self) -> None:
        return self._handle()

    def do_HEAD(self) -> None:
        return self._handle(send_body=False)

    def _handle(self, send_body: bool = True) -> None:
        data = self.server.content  # type: ignore[attr-defined]
        total = len(data)
        status = HTTP_OK
        resp_body = data

        range_hdr = self.headers.get("Range", "")
        if range_hdr.startswith("bytes="):
            try:
                spec = range_hdr[6:]
                start_str, _, end_str = spec.partition("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else total - 1
                start = max(0, min(start, total - 1))
                end = max(start, min(end, total - 1))
                resp_body = data[start : end + 1]
                status = HTTP_PARTIAL
            except (ValueError, IndexError):
                pass

        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(resp_body)))
        if status == HTTP_PARTIAL:
            self.send_header(
                "Content-Range",
                f"bytes {range_hdr[6:].split('-')[0]}-{len(resp_body) - 1}/{total}",
            )
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if send_body:
            self.wfile.write(resp_body)


class _ServerCtx:
    """Context manager that starts/stops a threaded HTTP server."""

    def __init__(self, content: bytes, port: int | None = None):
        self.content = content
        self.port = port or _find_free_port()
        self._thread: threading.Thread | None = None
        self._server: http.server.HTTPServer | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/data.bin"

    def __enter__(self) -> "_ServerCtx":
        handler = type("Handler", (RangeHTTPHandler,), {})
        self._server = http.server.HTTPServer(("127.0.0.1", self.port), handler)
        self._server.content = self.content  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Helper: build byte ranges for parallel download (mirrors download_utils)
# ---------------------------------------------------------------------------


def _build_ranges(file_size: int, num_parts: int) -> list[tuple[int, int]]:
    """Split a file into byte ranges for parallel download."""
    even_split = file_size // num_parts
    part_sz = even_split - (even_split % MB_100)
    cur = 0
    ranges: list[tuple[int, int]] = []
    for _ in range(num_parts - 1):
        ranges.append((cur, cur + part_sz - 1))
        cur += part_sz
    ranges.append((cur, file_size - 1))
    return ranges


async def _parallel_download_and_merge(
    downloader: HFDatasetDownloader,
    file_url: str,
    local_path: Path,
    file_size: int,
) -> bytes:
    """Run parallel download against a local server and return merged bytes."""
    num_parts = downloader.config.num_parallel_downloads
    ranges = _build_ranges(file_size, num_parts)

    loop = asyncio.get_running_loop()
    worker_func = functools.partial(
        _download_chunk_process,
        file_url,
        downloader.config.chunk_size,
        downloader.config.max_retries,
        downloader.config.timeout,
    )

    with ProcessPoolExecutor(max_workers=num_parts) as executor:
        results = await asyncio.gather(
            *[
                asyncio.wrap_future(
                    loop.run_in_executor(
                        executor,
                        functools.partial(
                            worker_func,
                            part_path=str(
                                local_path.with_name(f"{local_path.name}.part{i}")
                            ),
                            start_byte=start,
                            end_byte=end,
                        ),
                    )
                )
                for i, (start, end) in enumerate(ranges)
            ]
        )

    assert all(results), f"Not all parts succeeded: {results}"
    await downloader._merge_parts(local_path, num_parts)
    return local_path.read_bytes()


# ---------------------------------------------------------------------------
# Unit tests -- _download_chunk_attempt
# ---------------------------------------------------------------------------


class TestDownloadChunkAttempt:
    """Test the single-chunk download attempt logic directly."""

    def test_downloads_exact_range(self, tmp_dir: Path):
        """A successful range request should write exactly the requested bytes."""
        content = _make_content(500_000, seed=1)
        expected_start, expected_end = 1000, 5000
        part_path = tmp_dir / "chunk.part0"

        with _ServerCtx(content) as srv:
            result = _download_chunk_attempt(
                url=srv.url,
                chunk_size=8192,
                timeout=10,
                start_byte=expected_start,
                end_byte=expected_end,
                part_path=part_path,
                backoff=1.0,
            )

        assert result is True
        written = part_path.read_bytes()
        expected = content[expected_start : expected_end + 1]
        assert len(written) == expected_end - expected_start + 1
        assert written == expected

    def test_resume_from_partial_file(self, tmp_dir: Path):
        """If part of the chunk was already written, resume should complete it."""
        content = _make_content(500_000, seed=2)
        expected_start, expected_end = 2000, 9000
        part_path = tmp_dir / "chunk.part0"

        # Pre-write the first 3000 bytes of the range
        partial = content[expected_start : expected_start + 3000]
        part_path.write_bytes(partial)

        with _ServerCtx(content) as srv:
            result = _download_chunk_attempt(
                url=srv.url,
                chunk_size=8192,
                timeout=10,
                start_byte=expected_start,
                end_byte=expected_end,
                part_path=part_path,
                backoff=1.0,
            )

        assert result is True
        written = part_path.read_bytes()
        expected = content[expected_start : expected_end + 1]
        assert written == expected

    def test_oversized_part_file_is_discarded_and_redownloaded(self, tmp_dir: Path):
        """Oversized part file (server returned full file instead of range)
        must be discarded and redownloaded correctly."""
        content = _make_content(500_000, seed=3)
        part_path = tmp_dir / "chunk.part0"

        # Pre-write WRONG content (too large) -- simulates a server that
        # ignored the Range header and returned the entire file.
        part_path.write_bytes(b"x" * 10000)

        with _ServerCtx(content) as srv:
            result = _download_chunk_attempt(
                url=srv.url,
                chunk_size=8192,
                timeout=10,
                start_byte=1000,
                end_byte=5000,
                part_path=part_path,
                backoff=1.0,
            )

        # The fix detects the oversized part, deletes it, and downloads
        # the correct byte range.
        assert result is True
        written = part_path.read_bytes()
        expected = content[1000:5001]
        expected_len = 5001 - 1000
        assert len(written) == expected_len
        assert written == expected

    def test_already_complete_returns_true(self, tmp_dir: Path):
        """If the part file already has exactly the right bytes, return True."""
        content = _make_content(200_000, seed=4)
        expected_start, expected_end = 500, 3000
        part_path = tmp_dir / "chunk.part0"
        part_path.write_bytes(content[expected_start : expected_end + 1])

        with _ServerCtx(content) as srv:
            result = _download_chunk_attempt(
                url=srv.url,
                chunk_size=8192,
                timeout=10,
                start_byte=expected_start,
                end_byte=expected_end,
                part_path=part_path,
                backoff=1.0,
            )

        assert result is True


# ---------------------------------------------------------------------------
# Unit tests -- _download_chunk_process (the retry-loop wrapper)
# ---------------------------------------------------------------------------


class TestDownloadChunkProcess:
    """Test the multiprocessing worker wrapper."""

    def test_successful_download(self, tmp_dir: Path):
        content = _make_content(200_000, seed=5)
        part_path = tmp_dir / "chunk.part0"

        with _ServerCtx(content) as srv:
            ok = _download_chunk_process(
                url=srv.url,
                chunk_size=8192,
                max_retries=2,
                timeout=10,
                start_byte=1000,
                end_byte=9999,
                part_path=str(part_path),
            )

        assert ok is True
        written = part_path.read_bytes()
        assert written == content[1000:10000]

    def test_retries_then_fails(self, tmp_dir: Path):
        """With an invalid URL it should retry and eventually return False."""
        part_path = tmp_dir / "chunk.part0"

        ok = _download_chunk_process(
            url="http://127.0.0.1:1/nonexistent",  # nothing listening
            chunk_size=8192,
            max_retries=2,
            timeout=2,
            start_byte=0,
            end_byte=100,
            part_path=str(part_path),
        )

        assert ok is False


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------


class TestMergeParts:
    """Test the part-file merge logic."""

    @pytest.mark.asyncio
    async def test_merge_concatenates_parts_correctly(
        self, base_config: DownloadConfig
    ):
        part_count = 4
        content = _make_content(MB_100, seed=6)
        final_path = base_config.raw_data_dir / "merged.bin"
        final_path.parent.mkdir(parents=True, exist_ok=True)

        # Split content into 4 unequal parts (simulating realistic ranges)
        chunk = len(content) // part_count
        positions = [0]
        for i in range(1, part_count):
            positions.append(i * chunk)
        positions.append(len(content))

        for i in range(part_count):
            part_path = final_path.with_name(f"{final_path.name}.part{i}")
            part_path.write_bytes(content[positions[i] : positions[i + 1]])

        downloader = HFDatasetDownloader(base_config)
        await downloader._merge_parts(final_path, part_count)

        merged = final_path.read_bytes()
        assert merged == content
        # Verify parts were cleaned up
        for i in range(part_count):
            assert not final_path.with_name(f"{final_path.name}.part{i}").exists()


# ---------------------------------------------------------------------------
# Integration tests -- end-to-end parallel download against a local server
# ---------------------------------------------------------------------------


class TestParallelDownloadE2E:
    """Full integration tests using a local HTTP server."""

    @pytest.mark.asyncio
    async def test_download_file_parallel_matches_content(
        self, base_config: DownloadConfig
    ):
        """Download a file via parallel download and verify sha256."""
        # Use a small-ish file (still > MB_100 to trigger parallel path)
        content = _make_content(2 * MB_100 + 7777, seed=7)
        expected_sha = hashlib.sha256(content).hexdigest()

        with _ServerCtx(content) as srv:
            downloader = HFDatasetDownloader(base_config)
            local_path = base_config.raw_data_dir / "test.bin"
            local_path.parent.mkdir(parents=True, exist_ok=True)

            actual = await _parallel_download_and_merge(
                downloader, srv.url, local_path, len(content)
            )

            actual_sha = hashlib.sha256(actual).hexdigest()
            assert actual_sha == expected_sha, (
                f"SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
            )

    @pytest.mark.asyncio
    async def test_download_exact_mb_boundary(self, base_config: DownloadConfig):
        """File size that is an exact multiple of 100 MB (edge case)."""
        content = _make_content(MB_100 * 3, seed=8)
        expected_sha = hashlib.sha256(content).hexdigest()

        with _ServerCtx(content) as srv:
            downloader = HFDatasetDownloader(base_config)
            local_path = base_config.raw_data_dir / "exact.bin"
            local_path.parent.mkdir(parents=True, exist_ok=True)

            actual = await _parallel_download_and_merge(
                downloader, srv.url, local_path, len(content)
            )

            actual_sha = hashlib.sha256(actual).hexdigest()
            assert actual_sha == expected_sha

    @pytest.mark.asyncio
    async def test_large_file_byte_for_byte(self, base_config: DownloadConfig):
        """Byte-by-byte comparison after parallel download (500 MB+)."""
        meg = 1024 * 1024
        content = _make_content(500 * meg + 12345, seed=9)
        expected_sha = hashlib.sha256(content).hexdigest()

        with _ServerCtx(content) as srv:
            downloader = HFDatasetDownloader(base_config)
            local_path = base_config.raw_data_dir / "big.bin"
            local_path.parent.mkdir(parents=True, exist_ok=True)

            actual = await _parallel_download_and_merge(
                downloader, srv.url, local_path, len(content)
            )

            assert len(actual) == len(content), (
                f"Size mismatch: {len(actual)} vs {len(content)}"
            )
            assert hashlib.sha256(actual).hexdigest() == expected_sha

    @pytest.mark.asyncio
    async def test_server_ignoring_range_header(self, base_config: DownloadConfig):
        """If server returns 200 instead of 206, the download should fail
        (not produce corrupt output)."""

        class NoRangeHandler(RangeHTTPHandler):
            def _handle(self, send_body: bool = True) -> None:
                # Always return full file with 200, ignoring Range
                data = self.server.content  # type: ignore[attr-defined]
                self.send_response(HTTP_OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if send_body:
                    self.wfile.write(data)

        port = _find_free_port()
        content = _make_content(MB_100 + 5000, seed=10)

        server = http.server.HTTPServer(
            ("127.0.0.1", port),
            type("Handler", (NoRangeHandler,), {}),
        )
        server.content = content  # type: ignore[attr-defined]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            url = f"http://127.0.0.1:{port}/data.bin"
            part_path = base_config.raw_data_dir / "test.part0"
            part_path.parent.mkdir(parents=True, exist_ok=True)

            # Should fail because server won't return 206
            ok = _download_chunk_process(
                url=url,
                chunk_size=8192,
                max_retries=2,
                timeout=5,
                start_byte=0,
                end_byte=999,
                part_path=str(part_path),
            )

            assert ok is False, (
                "Should fail when server refuses to honour Range requests"
            )
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=2)
