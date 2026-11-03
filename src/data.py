import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def load_movielens(path):
    ratings = pd.read_csv(f"{path}/ratings.csv")
    movies = pd.read_csv(f"{path}/movies.csv")
    tags = pd.read_csv(f"{path}/tags.csv")
    users = np.sort(ratings.userId.unique())
    items = np.sort(ratings.movieId.unique())
    ratings["u"] = ratings.userId.map({u: k for k, u in enumerate(users)})
    ratings["i"] = ratings.movieId.map({m: k for k, m in enumerate(items)})
    movies = movies.set_index("movieId").loc[items].reset_index()
    return ratings, movies, tags


def temporal_split(ratings, train=0.7, val=0.1):
    r = ratings.sort_values(["u", "timestamp", "i"]).reset_index(drop=True)
    pos = r.groupby("u").cumcount().values
    n = r.groupby("u").u.transform("size").values
    cut_tr = (n * train).astype(int)
    cut_va = (n * (train + val)).astype(int)
    r["split"] = np.where(pos < cut_tr, "train", np.where(pos < cut_va, "val", "test"))
    return (r[r.split == "train"].copy(), r[r.split == "val"].copy(), r[r.split == "test"].copy())


def to_csr(df, shape, values=None):
    v = np.ones(len(df)) if values is None else values
    return sp.csr_matrix((v, (df.u.values, df.i.values)), shape=shape)


def build_tfidf(movies, tags, min_df=3):
    tag_docs = (
        tags.assign(t="tag_" + tags.tag.astype(str).str.lower().str.strip().str.replace(r"\s+", "_", regex=True))
        .groupby("movieId").t.apply(lambda s: " ".join(s))
    )
    docs = []
    for _, row in movies.iterrows():
        toks = ["g_" + g.replace(" ", "") for g in str(row.genres).split("|")]
        m = re.search(r"\((\d{4})\)\s*$", str(row.title).strip())
        if m:
            toks.append("decade_" + m.group(1)[:3])
        toks.append(tag_docs.get(row.movieId, ""))
        docs.append(" ".join(toks))
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=min_df, sublinear_tf=True)
    return vec.fit_transform(docs).tocsr(), vec.get_feature_names_out()
