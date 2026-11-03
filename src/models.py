import numpy as np
import scipy.sparse as sp

from .data import to_csr


def fit_bias(tr, U, I, lam_i=10, lam_u=15, iters=10):
    u, i, r = tr.u.values, tr.i.values, tr.rating.values
    mu = r.mean()
    n_i = np.bincount(i, minlength=I)
    n_u = np.bincount(u, minlength=U)
    bu = np.zeros(U)
    for _ in range(iters):
        bi = np.bincount(i, r - mu - bu[u], I) / (lam_i + n_i)
        bu = np.bincount(u, r - mu - bi[i], U) / (lam_u + n_u)
    return mu + bu[:, None] + bi[None, :]


def residuals(tr, base):
    return tr.rating.values - base[tr.u.values, tr.i.values]


def item_knn(tr, base, U, I, k=40, shrink=25, reg=0.5):
    res = residuals(tr, base)
    R = to_csr(tr, (U, I), res)
    B = to_csr(tr, (U, I))
    S = (R.T @ R).toarray().astype(np.float32)
    N = (B.T @ B).toarray().astype(np.float32)
    norm = np.sqrt(np.diag(S)) + 1e-8
    S /= np.outer(norm, norm).astype(np.float32)
    S *= N / (N + shrink)
    del N
    np.fill_diagonal(S, 0)
    idx = np.argpartition(-np.abs(S), k, axis=0)[:k]
    W = np.zeros_like(S)
    np.put_along_axis(W, idx, np.take_along_axis(S, idx, 0), 0)
    del S
    num = R @ W
    den = B @ np.abs(W)
    return base + num / (den + reg)


def _solve_rows(M, F, lam, out):
    f = F.shape[1]
    eye = np.eye(f)
    for r in range(M.shape[0]):
        s, e = M.indptr[r], M.indptr[r + 1]
        if s == e:
            out[r] = 0
            continue
        Fi = F[M.indices[s:e]]
        out[r] = np.linalg.solve(Fi.T @ Fi + lam * (e - s) * eye, Fi.T @ M.data[s:e])


def als_wr(tr, base, U, I, f=30, lam=0.2, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    R = to_csr(tr, (U, I), residuals(tr, base))
    Rt = R.T.tocsr()
    P = rng.normal(0, 0.1, (U, f))
    Q = rng.normal(0, 0.1, (I, f))
    for _ in range(iters):
        _solve_rows(R, Q, lam, P)
        _solve_rows(Rt, P, lam, Q)
    return base + P @ Q.T


def content_ridge(tr, base, X, U, I, lam=1.0):
    out = base.copy()
    for u, g in tr.groupby("u"):
        idx = g.i.values
        res = g.rating.values - base[u, idx]
        Xu = X[idx]
        K = (Xu @ Xu.T).toarray() + lam * np.eye(len(idx))
        w = Xu.T @ np.linalg.solve(K, res)
        out[u] += X @ w
    return out


def implicit_als(tr, U, I, f=30, lam=10.0, alpha=20.0, iters=10, seed=0):
    rng = np.random.default_rng(seed)
    B = to_csr(tr, (U, I))
    Bt = B.T.tocsr()
    P = rng.normal(0, 0.1, (U, f))
    Q = rng.normal(0, 0.1, (I, f))
    eye = np.eye(f)

    def step(M, F, out):
        FtF = F.T @ F + lam * eye
        for r in range(M.shape[0]):
            s, e = M.indptr[r], M.indptr[r + 1]
            if s == e:
                out[r] = 0
                continue
            Fi = F[M.indices[s:e]]
            out[r] = np.linalg.solve(FtF + alpha * Fi.T @ Fi, (1 + alpha) * Fi.sum(0))

    for _ in range(iters):
        step(B, Q, P)
        step(Bt, P, Q)
    return P @ Q.T


def bpr(tr, U, I, f=30, lr=0.05, reg=0.01, epochs=60, batch=512, seed=0):
    rng = np.random.default_rng(seed)
    u_all, i_all = tr.u.values, tr.i.values
    seen = np.zeros((U, I), dtype=bool)
    seen[u_all, i_all] = True
    P = rng.normal(0, 0.1, (U, f))
    Q = rng.normal(0, 0.1, (I, f))
    bi = np.zeros(I)
    n = len(u_all)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            u, p = u_all[b], i_all[b]
            q = rng.integers(0, I, len(b))
            bad = seen[u, q]
            q[bad] = rng.integers(0, I, bad.sum())
            d = Q[p] - Q[q]
            x = bi[p] - bi[q] + (P[u] * d).sum(1)
            g = 1 / (1 + np.exp(x))
            gu = g[:, None] * d - reg * P[u]
            gp = g[:, None] * P[u] - reg * Q[p]
            gq = -g[:, None] * P[u] - reg * Q[q]
            np.add.at(P, u, lr * gu)
            np.add.at(Q, p, lr * gp)
            np.add.at(Q, q, lr * gq)
            np.add.at(bi, p, lr * (g - reg * bi[p]))
            np.add.at(bi, q, lr * (-g - reg * bi[q]))
    return P @ Q.T + bi[None, :]


def ease(tr, U, I, lam=500.0):
    X = to_csr(tr, (U, I))
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices(I)] += lam
    P = np.linalg.inv(G)
    del G
    B = -P / np.diag(P)[None, :]
    del P
    np.fill_diagonal(B, 0)
    return np.asarray(X @ B.astype(np.float32))
