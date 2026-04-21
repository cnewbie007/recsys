# recsys

A hands-on implementation of recommendation system algorithms, covering retrieval and ranking stages with a path toward model serving.

## Algorithms

| Algorithm | Type | Stage |
|-----------|------|-------|
| ItemCF | Collaborative Filtering | Retrieval |
| UserCF | Collaborative Filtering | Retrieval |
| Factorization Machine | Embedding-based | Retrieval |
| Two-Tower (DSSM) | Dual Encoder | Retrieval |
| SASRec | Sequential / Transformer | Retrieval |
| DIN | Attention-based | Ranking |

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/) — 1M ratings across 6K users and 4K movies.

Download and place under `data/ml-1m/`:
```
data/
  ml-1m/
    ratings.dat
    movies.dat
    users.dat
```

## Project Structure

```
recsys/
  collaborative_filtering/   # ItemCF, UserCF
  factorization_machine/     # FM-based ID embedding
  two_tower/                 # DSSM dual encoder
  data/                      # datasets (not committed)
```

## Setup

```bash
pip install -r requirements.txt
```

## Design

Each algorithm follows a consistent interface built with deployment in mind:

- **Offline**: `train()` → `build_index()` → `save()` — fits the model and pre-computes a FAISS item index
- **Online**: `load()` → `recommend(user_id, topk)` — fast ANN lookup, no full forward pass at serve time

Evaluation uses **Recall@K**, **NDCG@K**, and **AUC** consistently across all algorithms.
