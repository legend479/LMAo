"""
Compression Middleware
Response compression for improved performance
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
import gzip
import brotli
from typing import List, Optional
import io

from src.shared.logging import get_logger

logger = get_logger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware for response compression"""

    def __init__(
        self,
        app,
        minimum_size: int = 1024,  # Minimum response size to compress (bytes)
        compression_level: int = 6,  # Compression level (1-9)
        exclude_paths: Optional[List[str]] = None,
        exclude_media_types: Optional[List[str]] = None,
    ):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        self.exclude_paths = exclude_paths or []
        self.exclude_media_types = exclude_media_types or [
            "image/",
            "video/",
            "audio/",
            "application/zip",
            "application/gzip",
            "application/x-gzip",
            "application/x-bzip2",
            "application/x-compress",
            "application/x-compressed",
        ]

    async def dispatch(self, request: Request, call_next):
        """Apply compression to responses"""

        # Check if path should be excluded
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Get client's accepted encodings
        accept_encoding = request.headers.get("accept-encoding", "")

        # Determine best compression method
        compression_method = self._get_best_compression(accept_encoding)

        if not compression_method:
            return await call_next(request)

        # Process request
        response = await call_next(request)

        # Check if response should be compressed
        if not self._should_compress_response(response):
            return response

        # Compress response
        return await self._compress_response(response, compression_method)

    def _get_best_compression(self, accept_encoding: str) -> Optional[str]:
        """Determine the best compression method based on client support"""

        accept_encoding = accept_encoding.lower()

        # Check for Brotli support (best compression)
        if "br" in accept_encoding:
            return "br"

        # Check for Gzip support (widely supported)
        if "gzip" in accept_encoding:
            return "gzip"

        # Check for deflate support
        if "deflate" in accept_encoding:
            return "deflate"

        return None

    def _should_compress_response(self, response: Response) -> bool:
        """Determine if response should be compressed"""

        # Don't compress if already compressed
        if response.headers.get("content-encoding"):
            return False

        # Don't compress if content-length is too small
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.minimum_size:
            return False

        # Don't compress certain media types
        content_type = response.headers.get("content-type", "")
        if any(
            content_type.startswith(media_type)
            for media_type in self.exclude_media_types
        ):
            return False

        # Don't compress if status code indicates error that shouldn't be cached
        if response.status_code >= 400:
            return False

        return True

    async def _compress_response(self, response: Response, method: str) -> Response:
        """Compress response content"""

        try:
            # Get response content
            if isinstance(response, StreamingResponse):
                # Handle streaming responses
                return await self._compress_streaming_response(response, method)
            else:
                # Handle regular responses
                return await self._compress_regular_response(response, method)

        except Exception as e:
            logger.warning(
                "Compression failed, returning uncompressed response",
                method=method,
                error=str(e),
            )
            return response

    async def _compress_regular_response(
        self, response: Response, method: str
    ) -> Response:
        """Compress regular response"""

        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        # Check if body is large enough to compress
        if len(body) < self.minimum_size:
            return response

        # Compress body
        compressed_body = self._compress_data(body, method)

        # Calculate compression ratio
        original_size = len(body)
        compressed_size = len(compressed_body)
        compression_ratio = (
            (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        )

        logger.debug(
            "Response compressed",
            method=method,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=f"{compression_ratio:.1f}%",
        )

        # Create new response with compressed content
        compressed_response = Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        # Update headers
        compressed_response.headers["content-encoding"] = method
        compressed_response.headers["content-length"] = str(len(compressed_body))
        compressed_response.headers["vary"] = "Accept-Encoding"

        # Add compression info header for debugging
        if logger.isEnabledFor("DEBUG"):
            compressed_response.headers["x-compression-ratio"] = (
                f"{compression_ratio:.1f}%"
            )

        return compressed_response

    async def _compress_streaming_response(
        self, response: StreamingResponse, method: str
    ) -> StreamingResponse:
        """Compress streaming response"""

        async def compress_stream():
            """Generator for compressed streaming content"""

            if method == "gzip":
                compressor = gzip.GzipFile(
                    fileobj=io.BytesIO(),
                    mode="wb",
                    compresslevel=self.compression_level,
                )
            elif method == "br":
                # Brotli doesn't have a streaming interface, so we'll buffer
                buffer = io.BytesIO()
                compressor = buffer
            else:
                # Fallback to gzip for deflate
                compressor = gzip.GzipFile(
                    fileobj=io.BytesIO(),
                    mode="wb",
                    compresslevel=self.compression_level,
                )

            try:
                if method == "br":
                    # Buffer all content for Brotli
                    content = b""
                    async for chunk in response.body_iterator:
                        content += chunk

                    compressed_content = brotli.compress(
                        content, quality=self.compression_level
                    )
                    yield compressed_content
                else:
                    # Stream compression for gzip
                    async for chunk in response.body_iterator:
                        if chunk:
                            compressor.write(chunk)
                            # Get compressed data
                            if hasattr(compressor, "fileobj"):
                                compressor.fileobj.seek(0)
                                compressed_chunk = compressor.fileobj.read()
                                if compressed_chunk:
                                    yield compressed_chunk
                                compressor.fileobj.seek(0)
                                compressor.fileobj.truncate()

                    # Finalize compression
                    compressor.close()
                    if hasattr(compressor, "fileobj"):
                        compressor.fileobj.seek(0)
                        final_chunk = compressor.fileobj.read()
                        if final_chunk:
                            yield final_chunk

            except Exception as e:
                logger.error("Streaming compression error", error=str(e))
                # Fallback: yield original content
                async for chunk in response.body_iterator:
                    yield chunk

        # Create new streaming response
        compressed_response = StreamingResponse(
            compress_stream(),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        # Update headers
        compressed_response.headers["content-encoding"] = method
        compressed_response.headers["vary"] = "Accept-Encoding"

        # Remove content-length for streaming responses
        if "content-length" in compressed_response.headers:
            del compressed_response.headers["content-length"]

        return compressed_response

    def _compress_data(self, data: bytes, method: str) -> bytes:
        """Compress data using specified method"""

        if method == "gzip":
            return gzip.compress(data, compresslevel=self.compression_level)
        elif method == "br":
            return brotli.compress(data, quality=self.compression_level)
        elif method == "deflate":
            # Use gzip without header for deflate
            return gzip.compress(data, compresslevel=self.compression_level)[10:-8]
        else:
            raise ValueError(f"Unsupported compression method: {method}")
