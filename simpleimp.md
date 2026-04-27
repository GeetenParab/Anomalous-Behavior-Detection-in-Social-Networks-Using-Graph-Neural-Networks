# Anomalous Behavior Detection in Social Networks — Implementation Summary

---

## Core Idea

> A bot acting alone looks like a user. A coordinated botnet **looks like a structural anomaly in the graph.**

Two parallel GNN models run simultaneously — one supervised, one unsupervised — catching both individual bots and coordinated botnets.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Graph Learning | PyTorch Geometric |
| Supervised Detection | Graph Attention Network (GATConv) |
| Unsupervised Detection | Graph Autoencoder (GAE + GATConv) |
| Semantic Embeddings | RoBERTa (sentence-transformers) |
| Community Detection | Louvain Algorithm |
| AI Profiling | Google Gemini 1.5 Flash |
| Dashboard | Streamlit |
| Datasets | TwiBot-22 + Cresci-2015 |

---

## Pipeline Overview

```
Raw Dataset (TwiBot-22 / Cresci-2015)
         │
         ▼
  Phase 1: Feature Engineering
         │
   ┌─────┴─────┐
   ▼           ▼
Phase 2A      Phase 2B
BotGAT        GAE (Unsupervised)
(Supervised)       │
   │          Phase 3: Louvain + Gemini
   └─────┬────┘
         ▼
  Phase 4: Streamlit Dashboard
```

---

## Phase 1 — Feature Engineering

Each user node is represented by a **3-modal feature vector**:

| Modality | Dimension | Source |
|---|---|---|
| **Description Embedding** (RoBERTa) | 768-dim | User bio text |
| **Tweet Embedding** (RoBERTa) | 768-dim | Tweet content |
| **Numerical Properties** | 5-dim | followers, following, tweet count, listed count, account age |
| **Categorical Properties** | 3-dim | verified, default profile, default avatar (binary flags) |

All tensors are pre-computed and saved as `.pt` files. At training time they are loaded directly — no re-computation.

The graph is a **heterogeneous directed graph** (nodes = users, edges = follow/retweet/mention).

---

## Phase 2A — BotGAT (Supervised)

**Goal:** Classify each node as **Bot** or **Human**.

**Architecture:**
```
4 feature branches → Linear(→ H/4) each → Concat [N, H]
   → Dropout + Linear
   → GATConv (4-head attention) × 2 layers
   → Linear → Logits [N, 2]
```
- Hidden dim: **128**, trained with **Adam + CrossEntropyLoss**
- Mini-batch via **NeighborLoader** (4-hop sampling: `[25, 15, 10, 5]`)
- Early stopping after **50 epochs** of no validation improvement
- Output: classification per user + **t-SNE** embedding visualization

---

## Phase 2B — GAE Anomaly Detection (Unsupervised)

**Goal:** Detect spammers without labels by learning what "normal" looks like.

**Key Innovation — Human-Normality Baseline:**
The GAE is trained **only on human-to-human edges**, so it learns what authentic social patterns look like. Any account deviating from this — bots, spammers, coordinated actors — produces a **high reconstruction error**.

```
Anomaly Score[i] = MSE(adj_real[i], adj_reconstructed[i])
```

**Decoder:** Inner-product  `adj_recon = sigmoid(Z @ Z.T)` where Z is 64-dim latent embedding.

**Thresholding (two methods):**
- IQR: flag if score > `Q3 + 1.5 × IQR`
- Percentile: flag top 5% scores

Output → `spammer_accounts.csv`

---

## Phase 3 — Coordinated Attack Analysis

**Step 1 — Louvain Community Detection:**
Groups tightly-connected users by maximizing graph modularity Q. Naturally clusters coordinated botnets together.

**Step 2 — Statistical Thresholding:**
Communities flagged if `mean_anomaly_score > μ + 2σ` of all communities.
- `members > 10` → **"Coordinated Botnet"**
- `members ≤ 10` → **"Suspicious Cluster"**

**Step 3 — Gemini AI Profiling:**
Member bio texts of each flagged community are sent to **Gemini 1.5 Flash**, which returns:
```
Category Name | Behavioral analysis (3 sentences)
e.g. "Political Influence Farm | Accounts display..."
```
Output stored in `ll.csv`.

---

## Phase 4 — Streamlit Dashboard

Three analysis pillars:

| Tab | Data Source | Shows |
|---|---|---|
| **Bot Detection** | BotGAT checkpoint | Per-user bot scores, t-SNE clusters |
| **Spammer Accounts** | `spammer_accounts.csv` | Anomaly scores, error distribution histogram |
| **Community Anomalies** | `ll.csv` + live Louvain | AI-profiled community types, mean error ranking |

---

## Key Design Decisions

- **Dual-model approach** — supervised for labeled bot patterns, unsupervised for novel/unknown anomalies
- **Human-normality training** — makes the GAE robust to unseen bot types
- **GATConv (attention)** — lets each node attend to neighbors with learned importance weights
- **Louvain** — detects coordinated behavior at group scale, not just individual level
- **Gemini profiling** — converts structural anomalies into human-readable threat categories

---

*April 2026*
