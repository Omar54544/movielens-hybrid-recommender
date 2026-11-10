import json
import os
import time


class Catalog:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.lambdas = d["lambdas"]
        self.titles = d["titles"]
        self.genres = d["genres"]
        self.counts = d["counts"]
        self.movie_ids = d.get("movie_ids")
        self.avg = d.get("avg")
        self.liked_pct = d.get("liked_pct")
        self.users = {u["id"]: u for u in d["users"]}

    def movie(self, i, rank=None, user=None):
        m = {"id": i, "title": self.titles[i], "genres": self.genres[i], "ratings_count": self.counts[i]}
        if self.avg:
            m["avg_rating"] = self.avg[i]
        if self.liked_pct:
            m["liked_pct"] = self.liked_pct[i]
        if user is not None and str(i) in user.get("m", {}):
            m["match"] = user["m"][str(i)]
        if rank is not None:
            m = {"rank": rank, **m}
        return m

    def snap(self, penalty):
        return min(self.lambdas, key=lambda x: abs(x - penalty))

    def user(self, user_id):
        if user_id not in self.users:
            raise KeyError(f"unknown user {user_id}")
        return self.users[user_id]

    def recommend(self, user_id, penalty=0, n=10):
        u = self.user(user_id)
        used = self.snap(penalty)
        ids = u["rec"][self.lambdas.index(used)][: max(1, min(n, 10))]
        recs = [self.movie(i, k + 1, u) for k, i in enumerate(ids)]
        avg = sum(self.counts[i] for i in ids) / len(ids)
        return {"user": user_id, "penalty": used, "recommendations": recs, "avg_ratings_count": round(avg, 1)}

    def profile(self, user_id):
        u = self.user(user_id)
        return {"user": user_id, "ratings": u["n"], "top_rated": [self.movie(i, k + 1) for k, i in enumerate(u["liked"])]}

    def search(self, query, limit=5):
        q = query.lower().strip()
        hits = [i for i, t in enumerate(self.titles) if q and q in t.lower()]
        hits.sort(key=lambda i: -self.counts[i])
        return {"query": query, "results": [self.movie(i) for i in hits[: max(1, min(limit, 20))]]}

    def add_rating(self, path, user_id, movie, rating):
        self.user(user_id)
        if not 0 <= movie < len(self.titles):
            raise KeyError(f"unknown movie {movie}")
        if rating not in [x / 2 for x in range(1, 11)]:
            raise ValueError("rating must be between 0.5 and 5 in steps of 0.5")
        if not self.movie_ids:
            raise ValueError("this catalog has no movie ids, run export_recommendations.py again")
        movie_id = self.movie_ids[movie]
        new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write("userId,movieId,rating,timestamp\n")
            f.write(f"{user_id},{movie_id},{rating},{int(time.time())}\n")
        with open(path, encoding="utf-8") as f:
            pending = sum(1 for _ in f) - 1
        return {"saved": True, "user": user_id, "title": self.titles[movie], "rating": rating, "pending": pending}
