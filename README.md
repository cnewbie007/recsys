# recsys

A hands-on implementation of recommendation system algorithms, covering retrieval and ranking stages with a path toward model serving.

## Algorithms

| Algorithm | Type | Stage | Status |
|-----------|------|-------|--------|
| ItemCF | Collaborative Filtering | Retrieval | Done |
| UserCF | Collaborative Filtering | Retrieval | Done |
| Factorization Machine | Embedding-based | Retrieval | Done |
| Two-Tower (DSSM) | Dual Encoder | Retrieval | Done |
| SASRec | Sequential / Transformer | Retrieval | Next |
| DIN | Attention-based | Ranking | Planned |

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
  factorization_machine/     # FM with feature interactions
  two_tower/                 # DSSM dual encoder
  sasrec/                    # Sequential transformer (coming soon)
  data/                      # datasets (not committed)
  shared/                    # BaseRecommender interface
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Collaborative Filtering (ItemCF + UserCF comparison)
python -m collaborative_filtering.main

# Factorization Machine
python -m factorization_machine.train --emb_size 64 --epochs 50 --lr 1e-3

# Two-Tower
python -m two_tower.train --emb_size 64 --epochs 50 --lr 1e-3
```

All runs log Recall@10 to [Weights & Biases](https://wandb.ai). Set `WANDB_API_KEY` in your environment before running.

## Docker / RunPod

```bash
# Build
docker build -t recsys .

# Run a training job (override CMD per job)
docker run --gpus all -e WANDB_API_KEY=<key> recsys \
  python -m two_tower.train --emb_size 64 --epochs 50
```

## Design

Each algorithm implements a consistent interface built with deployment in mind:

- **Offline**: `train()` → `build_index()` → `save()` — fits the model and pre-computes an item embedding index
- **Online**: `load()` → `recommend(user_id, topk)` — fast ANN lookup, no full forward pass at serve time

Evaluation uses **Recall@10** consistently across all algorithms (leave-one-out split: last rated item per user is held out).
