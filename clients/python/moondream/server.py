from http import server
import json
import io
from PIL import Image
import urllib.parse
from typing import Dict, Any
import traceback

from .onnx_vl import OnnxVL


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
            parsed_path = urllib.parse.urlparse(self.path)
            endpoint = parsed_path.path
            params = urllib.parse.parse_qs(parsed_path.query)

            if endpoint == "/caption":
                image = self.handle_image_request()
                length = params.get("length", ["normal"])[0]
                stream = params.get("stream", ["false"])[0].lower() == "true"

                if length not in ["normal", "short"]:
                    raise ValueError("Length parameter must be 'normal' or 'short'")

                if stream:
                    self.send_streaming_response()
                    for tokens in self.model.caption(image, length=length, stream=True)[
                        "caption"
                    ]:
                        self.stream_tokens(tokens, completed=False)
                    self.stream_tokens("", completed=True)
                else:
                    result = self.model.caption(image, length=length)
                    self.send_json_response(result)

            elif endpoint == "/query":
                image = self.handle_image_request()
                question = params.get("question", [""])[0]
                stream = params.get("stream", ["false"])[0].lower() == "true"

                if not question:
                    raise ValueError("Question parameter is required")

                if stream:
                    self.send_streaming_response()
                    for tokens in self.model.query(image, question, stream=True)[
                        "answer"
                    ]:
                        self.stream_tokens(tokens, completed=False)
                    self.stream_tokens("", completed=True)
                else:
                    result = self.model.query(image, question)
                    self.send_json_response(result)

            elif endpoint == "/detect":
                image = self.handle_image_request()
                object_name = params.get("object", [""])[0]
                if not object_name:
                    raise ValueError("Object parameter is required")
                result = self.model.detect(image, object_name)
                self.send_json_response(result)

            elif endpoint == "/point":
                image = self.handle_image_request()
                object_name = params.get("object", [""])[0]
                if not object_name:
                    raise ValueError("Object parameter is required")
                result = self.model.point(image, object_name)
                self.send_json_response(result)

            else:
                self.send_error_response(f"Unknown endpoint: {endpoint}", 404)

        except Exception as e:
            traceback.print_exc()
            self.send_error_response(str(e))

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_error_response("Method not allowed", 405)
