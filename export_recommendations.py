import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src import data, hybrid as H, metrics as M, models

LAMBDAS = [0, 1, 2, 3, 4, 5, 6]


def clip(p):
    return np.clip(p, 0.5, 5.0)


def blend_weights(path, names):
    if not os.path.exists(path):
        return {k: 1 / len(names) for k in names}
    runs = json.load(open(path))["blend_weights"]["Hybrid + implicit"]
    w = {k: float(np.mean([r[k] for r in runs])) for k in names}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def rating_weights(path, names):
    if not os.path.exists(path):
        return {k: 1 / len(names) for k in names}
    runs = json.load(open(path))["blend_weights"]["rating"]
    return {k: float(np.mean([r[k] for r in runs])) for k in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml-latest-small")
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--feedback", default="service/new_ratings.csv")
    args = ap.parse_args()

    print(f"loading {args.data}", flush=True)
    ratings, movies, tags = data.load_movielens(args.data)
    if os.path.exists(args.feedback):
        extra = pd.read_csv(args.feedback)
        users_map = dict(zip(ratings.userId, ratings.u))
        items_map = dict(zip(movies.movieId, movies.index))
        extra = extra[extra.userId.isin(users_map) & extra.movieId.isin(items_map)].copy()
        extra["u"] = extra.userId.map(users_map)
        extra["i"] = extra.movieId.map(items_map)
        ratings = pd.concat([ratings, extra[ratings.columns]], ignore_index=True).drop_duplicates(["u", "i"], keep="last")
        print(f"added {len(extra)} ratings from {args.feedback}", flush=True)
    U, I = int(ratings.u.max()) + 1, int(ratings.i.max()) + 1
    X, _ = data.build_tfidf(movies, tags)

    base = clip(models.fit_bias(ratings, U, I))
    steps = {
        "knn": lambda: clip(models.item_knn(ratings, base, U, I)),
        "als": lambda: clip(models.als_wr(ratings, base, U, I)),
        "content": lambda: clip(models.content_ridge(ratings, base, X, U, I, 3.0)),
        "ials": lambda: models.implicit_als(ratings, U, I),
        "bpr": lambda: models.bpr(ratings, U, I),
        "ease": lambda: models.ease(ratings, U, I, 1500.0),
    }
    raw = {"bias": base}
    for k, (name, fit) in enumerate(steps.items(), 1):
        t = time.time()
        print(f"[{k}/{len(steps)}] training {name}...", end=" ", flush=True)
        raw[name] = fit()
        print(f"{time.time() - t:.0f} s", flush=True)
    seen = M.dense_mask(ratings, (U, I))
    z = {k: M.zscore(v, seen) for k, v in raw.items()}
    w = blend_weights(args.results, list(raw))
    score = H.apply_ranking_blend(z, w)
    counts = np.bincount(ratings.i.values, minlength=I)
    explicit = ["bias", "knn", "als", "content"]
    pred = clip(H.apply_blend({k: raw[k] for k in explicit}, rating_weights(args.results, explicit)))
    avg = ratings.groupby("i").rating.mean().reindex(range(I)).fillna(0).round(2)
    liked_pct = (ratings.rating >= 4).groupby(ratings.i).mean().reindex(range(I)).fillna(0).mul(100).round().astype(int)
    tops = [M.top_k(H.popularity_penalty(score, counts, lam), seen, args.top) for lam in LAMBDAS]

    liked = (
        ratings.sort_values(["u", "rating", "timestamp"], ascending=[True, False, False])
        .groupby("u").head(6).groupby("u").i.apply(list)
    )
    n_rated = np.bincount(ratings.u.values, minlength=U)
    users = ratings.drop_duplicates("u").sort_values("u")
    payload = {
        "lambdas": LAMBDAS,
        "movie_ids": movies.movieId.astype(int).tolist(),
        "titles": movies.title.tolist(),
        "genres": movies.genres.str.replace("|", ", ", regex=False).tolist(),
        "counts": counts.tolist(),
        "avg": avg.tolist(),
        "liked_pct": liked_pct.tolist(),
        "users": [
            {"id": int(users.userId.iloc[k]), "n": int(n_rated[k]), "liked": liked[k],
             "rec": [t[k].tolist() for t in tops],
             "m": {str(i): int(round(pred[k, i] / 5 * 100)) for i in sorted({j for t in tops for j in t[k]})}}
            for k in range(U)
        ],
    }
    os.makedirs(args.out, exist_ok=True)
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    for template, target in [("demo_template.html", "index.html"), ("telon_template.html", "telon.html")]:
        html = open(template, encoding="utf-8").read().replace("__DATA__", blob)
        with open(f"{args.out}/{target}", "w", encoding="utf-8") as f:
            f.write(html)
    with open(f"{args.out}/recommendations.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {args.out}/index.html, telon.html and recommendations.json for {U} users, weights: " + ", ".join(f"{k}={v:.2f}" for k, v in w.items()))


if __name__ == "__main__":
    main()
