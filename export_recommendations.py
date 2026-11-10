import argparse
import json
import os

import numpy as np

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml-latest-small")
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    ratings, movies, tags = data.load_movielens(args.data)
    U, I = int(ratings.u.max()) + 1, int(ratings.i.max()) + 1
    X, _ = data.build_tfidf(movies, tags)

    base = clip(models.fit_bias(ratings, U, I))
    raw = {
        "bias": base,
        "knn": clip(models.item_knn(ratings, base, U, I)),
        "als": clip(models.als_wr(ratings, base, U, I)),
        "content": clip(models.content_ridge(ratings, base, X, U, I, 3.0)),
        "ials": models.implicit_als(ratings, U, I),
        "bpr": models.bpr(ratings, U, I),
        "ease": models.ease(ratings, U, I, 1500.0),
    }
    seen = M.dense_mask(ratings, (U, I))
    z = {k: M.zscore(v, seen) for k, v in raw.items()}
    w = blend_weights(args.results, list(raw))
    score = H.apply_ranking_blend(z, w)
    counts = np.bincount(ratings.i.values, minlength=I)
    tops = [M.top_k(H.popularity_penalty(score, counts, lam), seen, args.top) for lam in LAMBDAS]

    liked = (
        ratings.sort_values(["u", "rating", "timestamp"], ascending=[True, False, False])
        .groupby("u").head(6).groupby("u").i.apply(list)
    )
    n_rated = np.bincount(ratings.u.values, minlength=U)
    users = ratings.drop_duplicates("u").sort_values("u")
    payload = {
        "lambdas": LAMBDAS,
        "titles": movies.title.tolist(),
        "genres": movies.genres.str.replace("|", ", ", regex=False).tolist(),
        "counts": counts.tolist(),
        "users": [
            {"id": int(users.userId.iloc[k]), "n": int(n_rated[k]), "liked": liked[k],
             "rec": [t[k].tolist() for t in tops]}
            for k in range(U)
        ],
    }
    os.makedirs(args.out, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    for template, target in [("demo_template.html", "index.html"), ("telon_template.html", "telon.html")]:
        html = open(template, encoding="utf-8").read().replace("__DATA__", data)
        with open(f"{args.out}/{target}", "w", encoding="utf-8") as f:
            f.write(html)
    with open(f"{args.out}/recommendations.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {args.out}/index.html, telon.html and recommendations.json for {U} users, weights: " + ", ".join(f"{k}={v:.2f}" for k, v in w.items()))


if __name__ == "__main__":
    main()
