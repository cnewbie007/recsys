# recsys

A hands-on practice repo for recommendation system algorithms, from retrieval to ranking, with an eye toward model serving.

## Dataset

**MovieLens 1M** — 1M ratings, 6K users, 4K movies.
Download from https://grouplens.org/datasets/movielens/ and place under `data/ml-1m/`.

Expected files:
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
  factorization_machine/     # FM-based ID embedding model
  two_tower/                 # DSSM-style dual encoder (retrieval)
  data/                      # datasets (not committed)
```

## Algorithm Roadmap

| Algorithm | Stage | Status |
|-----------|-------|--------|
| ItemCF / UserCF | Retrieval | Done |
| Factorization Machine | Retrieval | Done |
| Two-Tower (DSSM) | Retrieval | Done |
| SASRec | Retrieval (sequential) | Next |
| DIN | Ranking | Planned |

## Standard Algorithm Interface

Every new algorithm must implement this interface to stay consistent and deployment-ready:

```python
class BaseRecommender:
    def train(self, data): ...           # offline: fit the model
    def build_index(self): ...           # offline: precompute embeddings/ANN index
    def recommend(self, user_id, topk):  # online: return ranked item id list
    def evaluate(self, test_data): ...   # offline: return metrics dict (AUC, Recall@K, etc.)
    def save(self, path): ...            # export model + index for serving
    def load(self, path): ...            # load artifacts for serving
```

`build_index` is required for embedding-based models (Two-Tower, DSSM, SASRec): pre-compute all item embeddings into a FAISS index so online `recommend()` is just an ANN lookup, not a full forward pass.

## Deployment Direction

The goal is to eventually wrap `recommend()` behind a REST service. Design each algorithm so that:
- `train` + `build_index` + `save` is the offline pipeline
- `load` + `recommend` is the online serving path (stateless, fast)

## Evaluation Metrics

Use consistent metrics across all algorithms:
- **Recall@K** — primary retrieval metric
- **NDCG@K** — ranking quality
- **AUC** — binary interaction prediction

## Dependencies

```
pip install -r requirements.txt
```
