"""Sobe o dashboard de criativos."""
import argparse
import webbrowser

from campanhas.dashboard.app import app


def main():
    parser = argparse.ArgumentParser(description="Dashboard de criativos de Meta Ads")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--open", action="store_true", help="abre o navegador depois de subir")
    args = parser.parse_args()

    if args.open:
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
