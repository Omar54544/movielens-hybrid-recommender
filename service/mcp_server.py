import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API = os.environ.get("RECOMMENDER_API", "http://127.0.0.1:8000")

TOOLS = [
    {
        "name": "recommend_movies",
        "description": "Top-N movie recommendations for a MovieLens user. The popularity penalty (0 to 6) trades accuracy for variety: 0 gives the most accurate list, higher values bring in less popular titles.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "integer", "description": "MovieLens user id (1 to 610)"},
                "penalty": {"type": "number", "description": "Popularity penalty, from 0 to 6", "default": 0},
                "n": {"type": "integer", "description": "Number of movies, up to 10", "default": 10},
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "user_profile",
        "description": "Movies that a MovieLens user rated highest, and how many ratings the user has.",
        "inputSchema": {
            "type": "object",
            "properties": {"user_id": {"type": "integer", "description": "MovieLens user id (1 to 610)"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "search_movies",
        "description": "Search movies by title. Returns genres and how many ratings each movie has.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Part of the title"},
                "limit": {"type": "integer", "description": "Maximum results", "default": 5},
            },
            "required": ["query"],
        },
    },
]


def call_api(path, params=None):
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(json.load(e).get("error", f"API error {e.code}"))
    except urllib.error.URLError:
        raise RuntimeError(f"Cannot reach the recommender API at {API}")


def fmt_movie(m):
    prefix = f"{m['rank']}. " if "rank" in m else "- "
    return f"{prefix}{m['title']} ({m['genres']}), {m['ratings_count']} ratings"


def run_tool(name, a):
    if name == "recommend_movies":
        r = call_api("/recommend", {"user": a["user_id"], "penalty": a.get("penalty", 0), "n": a.get("n", 10)})
        head = f"Recommendations for user {r['user']} (popularity penalty {r['penalty']}, average {r['avg_ratings_count']} ratings per movie):"
        return head + "\n" + "\n".join(fmt_movie(m) for m in r["recommendations"])
    if name == "user_profile":
        r = call_api(f"/users/{int(a['user_id'])}")
        return f"User {r['user']} has {r['ratings']} ratings. Highest rated:\n" + "\n".join(fmt_movie(m) for m in r["top_rated"])
    if name == "search_movies":
        r = call_api("/movies/search", {"q": a["query"], "limit": a.get("limit", 5)})
        if not r["results"]:
            return f"No movies found for '{r['query']}'."
        return "\n".join(fmt_movie(m) for m in r["results"])
    raise ValueError(f"unknown tool {name}")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        version = msg.get("params", {}).get("protocolVersion", "2024-11-05")
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "movielens-recommender", "version": "1.0.0"},
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        p = msg.get("params", {})
        try:
            text, err = run_tool(p.get("name"), p.get("arguments", {})), False
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            text, err = f"Error: {e}", True
        return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}], "isError": err}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = handle(json.loads(line))
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
