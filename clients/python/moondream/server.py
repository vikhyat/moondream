import base64
import io
import json
import logging
import urllib.parse
from http import server
from typing import Any, Dict

from PIL import Image

from .onnx_vl import OnnxVL

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MoondreamHandler(server.BaseHTTPRequestHandler):
    model: OnnxVL = None  # Will be set when starting server

    def send_json_response(self, data: Dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_error_response(self, error: str, status: int = 400) -> None:
        self.send_json_response({"error": error}, status)

    def handle_image_request(self) -> Image.Image:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            raise ValueError("No image data received")

        image_data = self.rfile.read(content_length)
        return Image.open(io.BytesIO(image_data))

    def send_streaming_response(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def stream_tokens(self, chunk: str, completed: bool = False) -> None:
        data = {"chunk": chunk, "completed": completed}
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self) -> None:
        try:
            if self.headers.get("Content-Type") != "application/json":
                raise ValueError("Content-Type must be application/json")

            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                raise ValueError("No data received")

            payload = json.loads(self.rfile.read(content_length))

            image_url = payload.get("image_url")
            if not image_url:
                raise ValueError("image_url is required")

            parsed_path = urllib.parse.urlparse(self.path)
            endpoint = parsed_path.path

            # Convert base64 image for all endpoints
            image = self.decode_base64_image(image_url)

            if endpoint == "/caption":
                try:
                    length = payload.get("length", "normal")
                    stream = payload.get("stream", False)

                    if length not in ["normal", "short"]:
                        raise ValueError("Length parameter must be 'normal' or 'short'")

                    if stream:
                        self.send_streaming_response()
                        try:
                            for tokens in self.model.caption(
                                image, length=length, stream=True
                            )["caption"]:
                                self.stream_tokens(tokens, completed=False)
                            self.stream_tokens("", completed=True)
                        except Exception as e:
                            logger.error(
                                "Error during caption streaming", exc_info=True
                            )
                            self.stream_tokens(
                                "An error occurred during caption generation.",
                                completed=True,
                            )
                    else:
                        result = self.model.caption(image, length=length)
                        self.send_json_response(result)
                except Exception as e:
                    logger.error("Caption generation error", exc_info=True)
                    self.send_error_response("Caption generation failed.")

            elif endpoint == "/query":
                try:
                    question = payload.get("question")
                    if not question:
                        raise ValueError("question is required")

                    stream = payload.get("stream", False)
                    if stream:
                        self.send_streaming_response()
                        try:
                            for tokens in self.model.query(
                                image, question, stream=True
                            )["answer"]:
                                self.stream_tokens(tokens, completed=False)
                            self.stream_tokens("", completed=True)
                        except Exception as e:
                            logger.error("Error during query streaming", exc_info=True)
                            self.stream_tokens(
                                "An error occurred during query processing.",
                                completed=True,
                            )
                    else:
                        result = self.model.query(image, question)
                        self.send_json_response(result)
                except Exception as e:
                    logger.error("Query processing error", exc_info=True)
                    self.send_error_response("Query processing failed.")

            elif endpoint == "/detect":
                try:
                    object_name = payload.get("object")
                    if not object_name:
                        raise ValueError("object is required")
                    result = self.model.detect(image, object_name)
                    self.send_json_response(result)
                except Exception as e:
                    logger.error("Object detection error", exc_info=True)
                    self.send_error_response("Object detection failed.")

            elif endpoint == "/point":
                try:
                    object_name = payload.get("object")
                    if not object_name:
                        raise ValueError("object is required")
                    result = self.model.point(image, object_name)
                    self.send_json_response(result)
                except Exception as e:
                    logger.error("Object pointing error", exc_info=True)
                    self.send_error_response("Object pointing failed.")

        except Exception as e:
            logger.error("Unexpected error in request handling", exc_info=True)
            self.send_error_response("An unexpected error occurred.")

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Moondream Local Inference Server</title>
                <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌙</text></svg>">
                <style>
                    body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; }
                    a { color: #0066cc; }
                </style>
            </head>
            <body>
                <h1>Moondream Local Inference Server is Running!</h1>
                <p>Visit the <a href="https://docs.moondream.ai">Moondream documentation</a> to learn more.</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_error_response("Method not allowed", 405)

    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """Convert a base64 image string to a PIL Image object.

        Args:
            base64_string: Base64 encoded image string, may include data URI prefix

        Returns:
            PIL Image object

        Raises:
            ValueError: If the base64 string is invalid
        """
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        try:
            # Decode base64 string to bytes
            image_bytes = base64.b64decode(base64_string)
            # Convert bytes to PIL Image
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {str(e)}") from e
