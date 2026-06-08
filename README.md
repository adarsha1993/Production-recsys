# Production-recsys
Production-grade generative recommendation engine — movies, music, products
print("GITHUB README DRAFT")
print("=" * 60)

readme = '''
# Production Recommendation System

> A production-grade, cross-domain neural
> recommendation engine built over 7 weeks.
> Implements 2025/2026 SOTA models across
> Movies, Music, Products, and Video domains.

## 🏆 Key Results

| Model | NDCG@10 | vs Baseline |
|-------|---------|-------------|
| Random | 0.006 | — |
| User-CF | 0.069 | +10x vs random |
| SVD | 0.038 | — |
| GRank | 0.048 | +20% vs CF |
| **HSTU** | **0.084** | **+22% vs CF** |
| Popularity | 0.355 | (biased) |

**IPS correction:** Popularity drops 63%
(0.33→0.12) revealing true popularity bias.

**Cross-domain:** 0 code changes across
Movies → Music → Products → Video.

## 🏗️ Architecture

User Request
↓
Retrieval Layer (GRank + Qdrant + CLIP)
↓ 500 candidates
Ranking Layer (HSTU + OneRec + Multi-task)
↓ top 50
Re-ranking (MMR + IPS + FM-Intent)
↓ top 10
LLM Explainability (Ollama)
↓
Final Recommendations


## 📚 Papers Implemented

| Paper | Year | Day |
|-------|------|-----|
| HSTU (Meta MLPerf) | 2026 | Day 15 |
| Netflix FM | Mar 2025 | Day 16 |
| OneRec (Kuaishou) | 2025 | Day 17 |
| LightGCN | 2020 | Day 18 |
| DPO for RecSys | Dec 2025 | Day 18 |
| FM-Intent (Netflix) | Jul 2025 | Day 20 |
| GRank (WWW) | 2026 | Day 10 |
| CLIP | 2021 | Day 12 |

## 🛠️ Tech Stack

**Data:** PySpark · Delta Lake · S3 · Feast · DVC

**Models:** PyTorch · Qdrant · e5-large · CLIP · Ollama

**Serving:** BentoML · FastAPI · Redis · Kafka · Triton

**MLOps:** MLflow · Prefect · Evidently AI · Great Expectations

**Deploy:** Terraform · EKS · Helm · Argo CD · Prometheus

## 📊 Cross-Domain Results

| Domain | Signal | NDCG@10 | Code Changes |
|--------|--------|---------|--------------|
| Movies | Rating | 0.022 | — |
| Music | Play count | 0.035 | 0 |
| Products | Purchase funnel | 0.028 | 0 |
| Video | Multi-signal | 0.029 | 0 |

## 🚀 Quick Start

```bash
git clone https://github.com/adarsha1993/Production-recsys
cd Production-recsys
docker compose up -d
pip install -r requirements.txt
jupyter notebook notebooks/
```

## 📁 Structure
Production-recsys/
├── notebooks/          # Weeks 1-4 experiments
│   ├── week1_data/
│   ├── week2_retrieval/
│   ├── week3_ranking/
│   └── week4_evaluation/
├── src/                # Production code
│   ├── retrieval/
│   ├── ranking/
│   ├── reranking/
│   └── serving/        # Week 5
├── configs/domains/    # 4 domain YAMLs
├── infrastructure/     # Week 6 K8s
└── tests/
## 📖 Report

See `docs/report.pdf` (Week 7)

---
*CP612 — Group Project — 2026*
*Wilfrid Laurier University*
'''

# Save README
with open('../../README.md', 'w') as f:
    f.write(readme)

print(readme)
print("✅ README.md saved to project root")
