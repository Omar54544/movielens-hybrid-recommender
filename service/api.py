import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from core import Catalog


def make_handler(cat, app_path, feedback_path):
    class Handler(BaseHTTPRequestHandler):
        def page(self):
            try:
                with open(app_path, "rb") as f:
                    raw = f.read()
            except OSError:
                return self.reply(404, {"error": "app page not found"})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def reply(self, code, body):
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.cors()
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = [p for p in url.path.split("/") if p]
            try:
                if parts in ([], ["app"]):
                    return self.page()
                if parts == ["health"]:
                    return self.reply(200, {"status": "ok", "users": len(cat.users), "movies": len(cat.titles)})
                if parts == ["users"]:
                    return self.reply(200, {"users": sorted(cat.users)})
                if len(parts) == 2 and parts[0] == "users":
                    return self.reply(200, cat.profile(int(parts[1])))
                if parts == ["recommend"]:
                    return self.reply(200, cat.recommend(int(q["user"]), float(q.get("penalty", 0)), int(q.get("n", 10))))
                if parts == ["movies", "search"]:
                    return self.reply(200, cat.search(q.get("q", ""), int(q.get("limit", 5))))
                return self.reply(404, {"error": "not found"})
            except KeyError as e:
                return self.reply(404, {"error": str(e).strip("'\"")})
            except (ValueError, TypeError):
                return self.reply(400, {"error": "invalid parameters"})

        def do_OPTIONS(self):
            self.send_response(204)
            self.cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            if urlparse(self.path).path.rstrip("/") != "/ratings":
                return self.reply(404, {"error": "not found"})
            try:
                size = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(min(size, 10_000)) or b"{}")
                result = cat.add_rating(feedback_path, int(body["user"]), int(body["movie"]), float(body["rating"]))
                print(f"rating saved: user {result['user']}, {result['title']}, {result['rating']} ({result['pending']} pending)", flush=True)
                return self.reply(201, result)
            except KeyError as e:
                return self.reply(404 if "unknown" in str(e) else 400, {"error": str(e).strip("'\"")})
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                return self.reply(400, {"error": str(e) or "invalid body"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../docs/recommendations.json")
    ap.add_argument("--app", default="../docs/telon.html")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--feedback", default="new_ratings.csv")
    args = ap.parse_args()
    cat = Catalog(args.data)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cat, args.app, args.feedback))
    print(f"API listening on http://{args.host}:{args.port} ({len(cat.users)} users). App at /", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
