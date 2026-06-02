
import sys
sys.path.insert(0, "../..")
import numpy as np
import scipy.sparse as sp
import pytest

# ── fixtures ──────────────────────────────────────
@pytest.fixture
def small_matrix():
    """Tiny 4-user x 5-movie matrix for testing"""
    data = np.array([
        [5, 4, 0, 0, 1],
        [4, 5, 0, 0, 1],
        [0, 0, 4, 5, 0],
        [0, 0, 5, 4, 0],
    ], dtype=float)
    return sp.csr_matrix(data)

@pytest.fixture
def mappings():
    user2idx  = {1:0, 2:1, 3:2, 4:3}
    movie2idx = {10:0, 20:1, 30:2, 40:3, 50:4}
    idx2movie = {v:k for k,v in movie2idx.items()}
    idx2user  = {v:k for k,v in user2idx.items()}
    return user2idx, movie2idx, idx2movie, idx2user

# ── import models ─────────────────────────────────
from notebooks.week2_retrieval     .collaborative_filtering_module import (
    UserCF, ItemCF,
    cosine_similarity_sparse,
    rmse, mae, precision_at_k,
    recall_at_k, ndcg_at_k
)

# ── similarity tests ──────────────────────────────
def test_cosine_similarity_shape(small_matrix):
    sim = cosine_similarity_sparse(small_matrix)
    assert sim.shape == (4, 4)

def test_cosine_similarity_diagonal(small_matrix):
    sim = cosine_similarity_sparse(small_matrix)
    np.testing.assert_array_almost_equal(
        np.diag(sim), np.ones(4), decimal=5)

def test_cosine_similarity_range(small_matrix):
    sim = cosine_similarity_sparse(small_matrix)
    assert sim.min() >= -1.0
    assert sim.max() <= 1.0 + 1e-6

# ── metric tests ──────────────────────────────────
def test_rmse_perfect():
    preds = [(3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]
    assert rmse(preds) == 0.0

def test_mae_perfect():
    preds = [(3.0, 3.0), (4.0, 4.0)]
    assert mae(preds) == 0.0

def test_precision_at_k_perfect():
    recs     = [1, 2, 3, 4, 5]
    relevant = {1, 2, 3, 4, 5}
    assert precision_at_k(recs, relevant, 5) == 1.0

def test_precision_at_k_zero():
    recs     = [1, 2, 3]
    relevant = {4, 5, 6}
    assert precision_at_k(recs, relevant, 3) == 0.0

def test_ndcg_at_k_perfect():
    recs     = [1, 2, 3]
    relevant = {1, 2, 3}
    assert ndcg_at_k(recs, relevant, 3) == 1.0
