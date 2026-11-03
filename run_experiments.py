import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src import data, hybrid as H, metrics as M, models

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]


def clip(p):
    return np.clip(p, 0.5, 5.0)


def eval_rating(pred, df):
    y, p = df.rating.values, pred[df.u.values, df.i.values]
    return {"rmse": M.rmse(y, p), "mae": M.mae(y, p)}


def log(msg, t0):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml-latest-small")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--draws", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    ratings, movies, tags = data.load_movielens(args.data)
    U, I = int(ratings.u.max()) + 1, int(ratings.i.max()) + 1
    tr, va, te = data.temporal_split(ratings)
    X, vocab = data.build_tfidf(movies, tags)
    log(f"{len(ratings)} ratings, {U} users, {I} items, train/val/test = {len(tr)}/{len(va)}/{len(te)}, {len(vocab)} terms", t0)

    seen_tr = M.dense_mask(tr, (U, I))
    seen_trva = M.dense_mask(pd.concat([tr, va]), (U, I))
    rel_va = M.dense_mask(va, (U, I), 4.0)
    rel_te = M.dense_mask(te, (U, I), 4.0)
    counts = np.bincount(tr.i.values, minlength=I)
    cold_mask = counts[te.i.values] <= 2
    stats = {
        "ratings": int(len(ratings)), "users": U, "items": I,
        "sparsity": float(1 - len(ratings) / (U * I)),
        "train": len(tr), "val": len(va), "test": len(te),
        "tfidf_terms": int(len(vocab)),
        "cold_share_test": float(cold_mask.mean()),
        "eval_users_test": int(rel_te.any(1).sum()),
    }

    base = clip(models.fit_bias(tr, U, I))
    knn = clip(models.item_knn(tr, base, U, I))
    log("bias and kNN done", t0)

    best = (9, None, None)
    for lam in (0.3, 1.0, 3.0):
        c = clip(models.content_ridge(tr, base, X, U, I, lam))
        v = eval_rating(c, va)["rmse"]
        if v < best[0]:
            best = (v, lam, c)
    content, stats["content_lambda"] = best[2], best[1]
    log(f"content done (lambda={best[1]})", t0)

    best = (-1, None, None)
    for lam in (200.0, 500.0, 1500.0):
        e = models.ease(tr, U, I, lam)
        v = M.ranking_metrics(e, seen_tr, rel_va)["ndcg"].mean()
        if v > best[0]:
            best = (v, lam, e)
    ease, stats["ease_lambda"] = best[2], best[1]
    log(f"EASE done (lambda={best[1]})", t0)

    pop = np.tile(np.log1p(counts)[None, :], (U, 1)).astype(np.float32)
    rating_rows, rank_rows, cold_rows, rerank_rows, weights, boot = [], [], [], [], {}, {}

    for seed in range(args.seeds):
        als = clip(models.als_wr(tr, base, U, I, seed=seed))
        ials = models.implicit_als(tr, U, I, seed=seed)
        bpr = models.bpr(tr, U, I, seed=seed)
        log(f"seed {seed}: ALS, implicit ALS and BPR done", t0)

        preds = {"bias": base, "knn": knn, "als": als, "content": content}
        w_rat = H.fit_rating_blend(preds, va)
        hyb = clip(H.apply_blend(preds, w_rat))
        weights.setdefault("rating", []).append(w_rat)

        rating_rows.append({"model": "Global mean", "seed": seed, **eval_rating(np.full((U, I), tr.rating.mean()), te)})
        for name, p in [("Bias baseline", base), ("Item-kNN", knn), ("ALS-WR", als), ("Content", content), ("Hybrid", hyb)]:
            rating_rows.append({"model": name, "seed": seed, **eval_rating(p, te)})
        for name, p in [("Bias baseline", base), ("ALS-WR", als), ("Content", content), ("Hybrid", hyb)]:
            sub = te[cold_mask]
            cold_rows.append({"model": name, "seed": seed, **eval_rating(p, sub)})
            warm = te[~cold_mask]
            cold_rows.append({"model": name + " (warm)", "seed": seed, **eval_rating(p, warm)})

        raw = {"bias": base, "knn": knn, "als": als, "content": content, "ials": ials, "bpr": bpr, "ease": ease}
        zv = {k: M.zscore(v, seen_tr) for k, v in raw.items()}
        zt = {k: M.zscore(v, seen_trva) for k, v in raw.items()}
        rng_scores = np.random.default_rng(seed).random((U, I)).astype(np.float32)

        names = {"bias": "Bias baseline", "knn": "Item-kNN", "als": "ALS-WR", "content": "Content",
                 "ials": "Implicit ALS", "bpr": "BPR", "ease": "EASE"}
        results = {"Random": M.ranking_metrics(rng_scores, seen_trva, rel_te),
                   "Popularity": M.ranking_metrics(pop, seen_trva, rel_te)}
        for k in raw:
            results[names[k]] = M.ranking_metrics(raw[k], seen_trva, rel_te)

        blends = {}
        v1 = ["bias", "knn", "als", "content"]
        for label, comps in [("Hybrid", v1), ("Hybrid + implicit", list(raw))]:
            w, val_ndcg = H.fit_ranking_blend({k: zv[k] for k in comps}, seen_tr, rel_va, args.draws, seed)
            weights.setdefault(label, []).append({**w, "val_ndcg": val_ndcg})
            blends[label] = H.apply_ranking_blend({k: zt[k] for k in comps}, w)
            results[label] = M.ranking_metrics(blends[label], seen_trva, rel_te)
        for name, m in results.items():
            rank_rows.append({"model": name, "seed": seed, **M.summarize(m, I)})

        for label, s in blends.items():
            for lam in LAMBDAS:
                m = M.ranking_metrics(H.popularity_penalty(s, counts, lam), seen_trva, rel_te)
                row = M.summarize(m, I)
                row["mean_train_count"] = float(counts[m["top"]].mean())
                rerank_rows.append({"variant": label, "lambda_p": lam, "seed": seed, **row})

        if seed == 0:
            boot["hybrid_vs_popularity"] = M.paired_bootstrap(results["Hybrid"]["ndcg"], results["Popularity"]["ndcg"])
            boot["hybrid_vs_knn"] = M.paired_bootstrap(results["Hybrid"]["ndcg"], results["Item-kNN"]["ndcg"])
            boot["hybrid_implicit_vs_hybrid"] = M.paired_bootstrap(results["Hybrid + implicit"]["ndcg"], results["Hybrid"]["ndcg"])
        log(f"seed {seed}: evaluation done", t0)

    tables = {
        "rating": (pd.DataFrame(rating_rows), ["model"]),
        "ranking": (pd.DataFrame(rank_rows), ["model"]),
        "cold_start": (pd.DataFrame(cold_rows), ["model"]),
        "rerank": (pd.DataFrame(rerank_rows), ["variant", "lambda_p"]),
    }
    summary = {}
    for name, (df, keys) in tables.items():
        g = df.drop(columns="seed").groupby(keys, sort=False)
        agg = g.mean().join(g.std().add_suffix("_std")).reset_index()
        agg.to_csv(f"{args.out}/{name}.csv", index=False)
        df.to_csv(f"{args.out}/{name}_per_seed.csv", index=False)
        summary[name] = agg.to_dict(orient="records")
    summary.update({"stats": stats, "blend_weights": weights, "bootstrap_ndcg_diff": boot, "seeds": args.seeds})
    with open(f"{args.out}/results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log("results written", t0)
    print(pd.read_csv(f"{args.out}/rating.csv").round(4).to_string(index=False))
    print(pd.read_csv(f"{args.out}/ranking.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
