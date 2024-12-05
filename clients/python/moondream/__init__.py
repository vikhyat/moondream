from typing import Optional
from .types import VLM
from .onnx_vl import OnnxVL
from .cloud_vl import CloudVL


def vl(*, model: Optional[str] = None, api_key: Optional[str] = None) -> VLM:
    if api_key:
        return CloudVL(api_key)

    if model:
        return OnnxVL.from_path(model)

    raise ValueError("Either model_path or api_key must be provided.")


def main():
    import argparse
    import sys
    from http import server
    from .server import MoondreamHandler
    from .onnx_vl import OnnxVL

    parser = argparse.ArgumentParser(description="Moondream local server")
    parser.add_argument("--model", type=str, help="Path to model file", required=True)
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=3281, help="Server port (default: 3281)"
    )

    args = parser.parse_args()

    try:
        # Initialize the model
        model = OnnxVL.from_path(args.model)
        MoondreamHandler.model = model  # Set the model for the handler

        # Create and start the server
        server_address = (args.host, args.port)
        httpd = server.HTTPServer(server_address, MoondreamHandler)

        print(f"Starting Moondream server on http://{args.host}:{args.port}")
        print("Available endpoints:")
        print("  POST /caption - Generate image caption")
        print("  POST /query?question=<question> - Answer questions about image")
        print("  POST /detect?object=<object> - Detect object in image")
        print("  POST /point?object=<object> - Point to object in image")
        print("  GET /health - Check server health")
        print("\nPress Ctrl+C to stop the server")

        httpd.serve_forever()

    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
