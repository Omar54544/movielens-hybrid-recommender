# Hybrid Recommender for MovieLens

Hybrid movie recommender built on MovieLens `ml-latest-small` (100,836 ratings, 610 users, 9,724 movies). It combines a bias baseline, item-based kNN, ALS-WR, a TF-IDF content model and three implicit-feedback models (implicit ALS, BPR and EASE), and adds a popularity-penalty re-ranking step to increase catalog coverage.

Course project, Intelligent Agents, Universidad de Ibagué.

Authors: Angel David Arenales, Jose Omar Cabezas, Carlos Mario Montoya. Advisor: Dr. Alexandra La Cruz.

## Setup

```bash
pip install -r requirements.txt
mkdir -p data && cd data
wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip && cd ..
```

The dataset is also available in the shared Drive folder linked in the paper.

## Run

```bash
python run_experiments.py --data data/ml-latest-small --out results --seeds 5
```

On Google Colab, open `notebooks/run_colab.ipynb` and run all cells. A full run with five seeds takes roughly 15 to 25 minutes (EASE inverts a 9,724 x 9,724 matrix once per candidate lambda).

## Protocol

- Per-user temporal split: the oldest 70% of each user's ratings go to training, the next 10% to validation, the most recent 20% to test.
- Rating prediction: RMSE and MAE on the test ratings. Hybrid weights are fitted with non-negative least squares on the validation ratings.
- Ranking: NDCG@10, Precision@10, Recall@10 and catalog coverage. A test item is relevant if its rating is at least 4. Candidates are all movies the user has not rated in training or validation. Blend weights are searched on validation NDCG@10.
- Cold start: test ratings of movies with at most two training ratings.
- Diversity: re-ranking by subtracting `lambda_p * log(1 + count) / log(1 + max count)` from the blended score.
- Stochastic models (ALS, implicit ALS, BPR, blend search) are repeated over seeds; tables report mean and standard deviation. `results.json` also holds paired bootstrap confidence intervals over users.

## Layout

```
src/data.py        loading, temporal split, TF-IDF features
src/models.py      bias, item-kNN, ALS-WR, content ridge, implicit ALS, BPR, EASE
src/metrics.py     RMSE, MAE, NDCG, precision, recall, coverage, bootstrap
src/hybrid.py      NNLS rating blend, ranking blend search, popularity penalty
run_experiments.py runs everything and writes results/
notebooks/         Colab notebook
results/           CSV tables and results.json
```

## Results

Final tables are written to `results/` after a run (`rating.csv`, `ranking.csv`, `cold_start.csv`, `rerank.csv`, `results.json`).
