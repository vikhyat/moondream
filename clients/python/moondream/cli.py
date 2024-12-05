import argparse
from . import vl
from .server import MoondreamHandler
from http import server


def main():
    parser = argparse.ArgumentParser(description="Moondream CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server command
    server_parser = subparsers.add_parser("serve", help="Start the Moondream server")
    server_parser.add_argument("--model", type=str, help="Path to the model file")
    server_parser.add_argument(
        "--host", type=str, default="localhost", help="Host to bind to"
    )
    server_parser.add_argument(
        "--port", type=int, default=3281, help="Port to listen on"
    )

    args = parser.parse_args()

    if args.command == "serve":
        if args.model:
            model = vl(model=args.model)
        else:
            parser.error("Model path is required")

        MoondreamHandler.model = model
        server_address = (args.host, args.port)
        httpd = server.HTTPServer(server_address, MoondreamHandler)
        print(f"Starting Moondream server on http://{args.host}:{args.port}")
        httpd.serve_forever()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
