import numpy as np
from scipy.optimize import nnls

from .metrics import ranking_metrics


def fit_rating_blend(preds, val):
    u, i = val.u.values, val.i.values
    A = np.column_stack([p[u, i] for p in preds.values()])
    w, _ = nnls(A, val.rating.values)
    return dict(zip(preds.keys(), w))


def apply_blend(preds, w):
    return sum(w[k] * preds[k] for k in preds)


def fit_ranking_blend(zscores, seen, rel, n_draws=300, seed=0):
    rng = np.random.default_rng(seed)
    names = list(zscores)
    cands = [np.eye(len(names))[j] for j in range(len(names))]
    cands.append(np.full(len(names), 1 / len(names)))
    cands += list(rng.dirichlet(np.ones(len(names)), n_draws))
    best, best_w = -1, None
    for w in cands:
        s = sum(wj * zscores[n] for wj, n in zip(w, names) if wj > 0)
        score = ranking_metrics(s, seen, rel)["ndcg"].mean()
        if score > best:
            best, best_w = score, w
    return dict(zip(names, best_w)), float(best)


def apply_ranking_blend(zscores, w):
    return sum(w[n] * zscores[n] for n in zscores if w[n] > 0)


def popularity_penalty(scores, train_counts, lam):
    pop = np.log1p(train_counts)
    return scores - lam * (pop / pop.max())[None, :]
