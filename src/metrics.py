import numpy as np


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def dense_mask(df, shape, min_rating=None):
    m = np.zeros(shape, dtype=bool)
    d = df if min_rating is None else df[df.rating >= min_rating]
    m[d.u.values, d.i.values] = True
    return m


def zscore(scores, seen):
    ok = ~seen
    n = ok.sum(1, keepdims=True)
    mean = np.where(ok, scores, 0).sum(1, keepdims=True) / n
    var = np.where(ok, (scores - mean) ** 2, 0).sum(1, keepdims=True) / n
    return ((scores - mean) / (np.sqrt(var) + 1e-9)).astype(np.float32)


def top_k(scores, seen, k=10):
    s = np.where(seen, -np.inf, scores)
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)


def ranking_metrics(scores, seen, rel, k=10):
    users = np.where(rel.any(1))[0]
    top = top_k(scores[users], seen[users], k)
    hits = np.take_along_axis(rel[users], top, 1)
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(1)
    n_rel = rel[users].sum(1)
    idcg = np.cumsum(disc)[np.minimum(n_rel, k) - 1]
    return {
        "ndcg": dcg / idcg,
        "precision": hits.sum(1) / k,
        "recall": hits.sum(1) / n_rel,
        "top": top,
        "users": users,
    }


def summarize(m, n_items):
    return {
        "ndcg": float(m["ndcg"].mean()),
        "precision": float(m["precision"].mean()),
        "recall": float(m["recall"].mean()),
        "coverage": float(len(np.unique(m["top"])) / n_items),
        "n_users": int(len(m["users"])),
    }


def paired_bootstrap(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
