import json


class Catalog:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.lambdas = d["lambdas"]
        self.titles = d["titles"]
        self.genres = d["genres"]
        self.counts = d["counts"]
        self.users = {u["id"]: u for u in d["users"]}

    def movie(self, i, rank=None):
        m = {"title": self.titles[i], "genres": self.genres[i], "ratings_count": self.counts[i]}
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
        recs = [self.movie(i, k + 1) for k, i in enumerate(ids)]
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
