# Implementation: Anomalous Behavior Detection in Social Networks Using Graph Neural Networks

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Phase 1 — Data Ingestion & Feature Engineering](#3-phase-1--data-ingestion--feature-engineering)
4. [Phase 2A — Supervised Bot Detection (BotGAT)](#4-phase-2a--supervised-bot-detection-botgat)
5. [Phase 2B — Unsupervised Anomaly Detection (GAE)](#5-phase-2b--unsupervised-anomaly-detection-gae)
6. [Phase 3 — Coordinated Attack Analysis (Louvain + Gemini)](#6-phase-3--coordinated-attack-analysis-louvain--gemini)
7. [Phase 4 — Streamlit Dashboard](#7-phase-4--streamlit-dashboard)
8. [Data Flow Summary](#8-data-flow-summary)
9. [File Reference](#9-file-reference)

---

## 1. Project Overview

This system detects malicious actors on social networks using a **multi-layered Graph Neural Network (GNN) framework**. It goes beyond single-user analysis by treating the social graph as the primary object of study — detecting both individual bots and coordinated communities through two parallel models.

**The core insight:** A bot acting alone looks like a user. A coordinated botnet **looks like a structural anomaly in the graph**. This system detects both.

### Key Technologies

| Component | Technology |
|---|---|
| Graph Learning | PyTorch Geometric (`torch_geometric`) |
| Supervised Detection | Graph Attention Network (`GATConv`) |
| Unsupervised Detection | Graph Autoencoder (`GAE` + `GATConv`) |
| Semantic Embeddings | RoBERTa (`sentence-transformers`) |
| Community Detection | Louvain Algorithm (`python-louvain`) |
| AI Profiling | Google Gemini 1.5 Flash |
| Dashboard | Streamlit |
| Dataset | TwiBot-22 + Cresci-2015 |

---

## 2. System Architecture

```
Raw TwiBot-22 / Cresci-2015 Dataset
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 1: Feature Engineering                  │
│   BotRGCN/datasets/ → preprocessing scripts (data1.py)         │
│   Output: des_tensor.pt, tweets_tensor.pt, num_properties.pt   │
│           cat_properties.pt, edge_index.pt, label.pt           │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  PHASE 2A: BotGAT   │     │   PHASE 2B: GAE Model    │
│  (Supervised)       │     │   (Unsupervised)         │
│  train.py → model   │     │   human.py → GAE trained │
│  Classifies Bot /   │     │   on human normality     │
│  Human per node     │     │   Outputs Reconstruction │
│  Checkpoint: .pt    │     │   Error per node         │
└─────────────────────┘     └────────────┬─────────────┘
          │                              │
          │                             ▼
          │               ┌──────────────────────────────┐
          │               │  PHASE 3: Louvain Clustering │
          │               │  Community Detection → flags  │
          │               │  groups by mean error > μ+2σ │
          │               │  Gemini LLM profiles them    │
          │               └────────────┬─────────────────┘
          │                            │
          └──────────────┬─────────────┘
                         ▼
          ┌─────────────────────────────┐
          │   PHASE 4: Streamlit UI     │
          │  ├── Bot Detection Tab      │
          │  ├── Spammer Accounts Tab  │
          │  └── Community Anomaly Tab │
          └─────────────────────────────┘
```

---

## 3. Phase 1 — Data Ingestion & Feature Engineering

### 3.1 Dataset Foundation

**Files:** `BotRGCN/twibot_22/`, `BotRGCN/datasets/cresci-2015/`

The system uses two datasets:
- **TwiBot-22** — Large-scale, real Twitter data with ground-truth bot labels
- **Cresci-2015** — A classic benchmark dataset, used for GAE training and inference (smaller scale, well-curated)

Both datasets provide a `node.json` (user metadata) and an edge list (follow/mention/retweet graphs).

---

### 3.2 Three-Modal User Feature Vector

For every user node, a high-dimensional feature vector is constructed by fusing **three modalities**:

#### A. Semantic Embeddings — RoBERTa (768-dim each)

Each user has two 768-dimensional vectors extracted using a pre-trained `sentence-transformers` RoBERTa model:

- **`des_tensor.pt`** — Embedding of the user's **profile description/bio**. Captures intent, tone, and topical alignment.
- **`tweets_tensor.pt`** — Aggregated embedding of the user's **tweet content**. Captures behavioral language patterns.

> These embeddings are pre-computed and saved to disk. At training time they are loaded directly from `.pt` files and never recomputed.

#### B. Numerical Properties — `num_properties_tensor.pt` (5-dim)

Normalized scalar values:
- `followers_count`
- `friends_count` (following)
- `statuses_count` (total tweets)
- `listed_count`
- `account_age` (days since creation)

Each feature is standardized using min-max or z-score normalization during preprocessing.

#### C. Categorical Properties — `cat_properties_tensor.pt` (3-dim)

Binary flags encoded as `{0, 1}`:
- `verified` — Is the account Twitter-verified?
- `default_profile` — Is a profile picture set?
- `default_profile_image` — Is the default avatar still in use?

---

### 3.3 Graph Construction

**File:** `GAT/dataset.py`

The `get_train_data(dataset_name)` function is the central data loader. It:

1. Loads all `.pt` tensors from `BotRGCN/{dataset}/processed_data/`
2. Assembles them into a PyTorch Geometric `Data` object
3. Attaches pre-computed train/val/test split indices

```python
data = Data(
    edge_index=edge_index,   # [2, num_edges] — COO-format directed edges
    edge_type=edge_type,     # [num_edges] — relation type (follow=0, retweet=1)
    y=labels,                # [num_nodes] — 0=human, 1=bot
    num_property_embedding=...,
    train_idx=..., val_idx=..., test_idx=...
)
data.des_embedding = des_tensor          # [num_nodes, 768]
data.tweet_embedding = tweets_tensor     # [num_nodes, 768]
data.cat_properties_tensor = cat_props  # [num_nodes, 3]
```

The graph is a **heterogeneous directed graph** where:
- **Nodes** = Twitter users
- **Edges** = Social interactions (follow, retweet, mention)
- **Edge Types** are encoded numerically for relation-aware message passing

---

## 4. Phase 2A — Supervised Bot Detection (BotGAT)

### 4.1 Model Architecture — `BotGAT`

**File:** `GAT/model.py`

The `BotGAT` model processes all four feature modalities and then passes them through two graph attention layers for binary classification (bot vs. human).

#### Step-by-step forward pass:

```
des_embedding [N, 768]  ──── Linear(768, H/4) + LeakyReLU ──┐
tweet_embedding [N, 768] ─── Linear(768, H/4) + LeakyReLU ──┤
num_properties [N, 5]   ──── Linear(5,   H/4) + LeakyReLU ──┼──► Concat → [N, H]
cat_properties [N, 3]   ──── Linear(3,   H/4) + LeakyReLU ──┘
                                        │
                              Dropout + Linear(H, H) + LeakyReLU
                                        │
                              GATConv(H, H/4, heads=4) → [N, H]  ← Layer 1
                                        │
                                     Dropout
                                        │
                              GATConv(H, H)              → [N, H]  ← Layer 2
                                        │
                              Linear(H, H) + LeakyReLU
                                        │
                              Linear(H, 2)               → [N, 2]  ← Logits
```

Where `H = hidden_dim = 128`.

**Key design choice:** The four feature branches project each modality into `H/4` dimensions which are then concatenated. This creates a balanced representation where no single modality dominates. The subsequent `GATConv` layers allow each node to **attend** to its neighbors' embeddings with learned importance weights.

---

### 4.2 Training Pipeline

**File:** `GAT/train.py`

#### Training Configuration

| Hyperparameter | Value |
|---|---|
| `hidden_dim` | 128 |
| `max_epoch` | 1000 |
| `batch_size` | 128 |
| `learning_rate` | 1e-4 |
| `weight_decay` | 1e-5 |
| `dropout` | 0.3 |
| `no_up` (early stop patience) | 50 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |

#### Mini-Batch Training with `NeighborLoader`

Because the graph is too large to process in full at once, **NeighborLoader** is used to sample subgraphs:

```python
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 15, 10, 5],  # 4-hop neighborhood sampling
    batch_size=128,
    input_nodes=data.train_idx,
    shuffle=True
)
```

This samples **4 hops** of neighbors for each seed node with decreasing fanout — a strategy that ensures sufficient graph context while keeping memory usage bounded.

#### Early Stopping

The best model state (by validation accuracy) is saved in memory. If no improvement is seen for `no_up=50` consecutive epochs, training halts and the best checkpoint is saved to `checkpoints/{dataset}_{accuracy}.pt`.

---

### 4.3 t-SNE Visualization of Learned Embeddings

**File:** `GAT/output.py`

After training, the penultimate layer activations (before final classification) are extracted and reduced to 2D using **t-SNE**. This produces a visual cluster separation showing how well the model separates bot and human communities in latent space. Saved as `tsne_output_visualization.png`.

---

## 5. Phase 2B — Unsupervised Anomaly Detection (GAE)

### 5.1 Core Concept: Reconstruction Error as an Anomaly Signal

**Files:** `GAT/gaemodel.py`, `GAT/gaetrain.py`, `GAT/human.py`, `GAT/humaninf.py`, `GAT/inference_anomalies.py`

The **Graph Autoencoder (GAE)** learns to reconstruct the social graph's adjacency matrix from node embeddings. When run against accounts that behave abnormally, it produces a **high reconstruction error** — because their structural pattern violates what the model learned as "normal."

The formula for a single node's anomaly score:
```
scores[i] = mean( (adj_real[i] - adj_recon[i])^2 )
```

This is the **Mean Squared Error** between the true row of the adjacency matrix and the reconstructed row, averaged over all neighbors.

---

### 5.2 The GATEncoder Architecture

**File:** `GAT/gaemodel.py`

The encoder mirrors the BotGAT feature-projection design:

```
des [N, 768]  ──── Linear(768, H/4) + LeakyReLU ──┐
tweet [N, 768] ─── Linear(768, H/4) + LeakyReLU ──┤
num_prop [N, 5] ── Linear(5,   H/4) + LeakyReLU ──┼──► Concat → [N, H]
cat_prop [N, 1] ── Linear(1,   H/4) + LeakyReLU ──┘
                              │
                   Dropout + Linear(H, H) + LeakyReLU
                              │
                   GATConv(H, H/4, heads=4) + ReLU
                              │
                   GATConv(H, out_channels=64)  → Z [N, 64]
```

The output `Z` is a 64-dimensional **latent embedding** per node. The decoder is an inner-product decoder:
```
adj_recon = sigmoid(Z @ Z.T)
```

---

### 5.3 Two-Stage Training Strategy

#### Stage A: Baseline GAE (`gaetrain.py`)

A standard GAE is trained on the full cresci-2015 dataset using edge-level reconstruction loss:

```python
transform = RandomLinkSplit(
    num_val=0.05, num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,
    split_labels=True
)
```

`RandomLinkSplit` holds out edges from the training graph and evaluates the model's ability to predict whether edges should exist. The model is trained for 200 epochs (AUC: ~0.98).

#### Stage B: Human-Normality Baseline (`human.py`) — The Key Innovation

A second, specialized GAE checkpoint is trained **exclusively on human-to-human edges**:

```python
# Only back-propagate on human↔human connections
mask = human_mask[edge_index[0]] & human_mask[edge_index[1]]
human_pos_edges = edge_index[:, mask]
loss = model.recon_loss(z, human_pos_edges.to(device))
```

**Why this matters:** By defining normality purely through authentic human social patterns, any account — bot, spammer, or coordinated actor — that deviates from these patterns will score high on reconstruction error. This makes the model robust to novel, unseen bot types.

---

### 5.4 Spammer Identification (`humaninf.py`, `inference_anomalies.py`)

After loading the human-baseline checkpoint, the GAE is run in inference mode over the full graph. Two thresholding strategies are evaluated:

**IQR Method (`inference_anomalies.py`):**
```python
Q1, Q3 = results['anomaly_score'].quantile([0.25, 0.75])
IQR = Q3 - Q1
iqr_threshold = Q3 + (1.5 * IQR)   # Standard Tukey fence
results['is_spammer'] = results['anomaly_score'] > iqr_threshold
```

**Percentile Method (`humaninf.py`):**
```python
threshold = results['anomaly_score'].quantile(0.95)
results['is_spammer'] = results['anomaly_score'] > threshold
```

Both methods output a `spammer_accounts.csv` and `spammer_dashboard_data.csv` containing flagged node IDs and their anomaly scores for dashboard integration.

---

## 6. Phase 3 — Coordinated Attack Analysis (Louvain + Gemini)

### 6.1 Step 1 — Louvain Community Detection

**File:** `GAT/louvian.py`

After computing per-node anomaly scores, the social graph is analyzed at group scale. The edge list is converted to a NetworkX graph, and **Louvain modularity optimization** is applied:

```python
import community as community_louvain
G = nx.Graph()
G.add_edges_from(list(zip(edge_index[0], edge_index[1])))
partition = community_louvain.best_partition(G, random_state=42)
```

Louvain maximizes modularity `Q` — a measure of how much denser the connections within a community are compared to a random graph. It naturally groups tightly-coordinated accounts together.

---

### 6.2 Step 2 — Statistical Thresholding

Each community is assigned an aggregate anomaly score:

```python
community_stats = results_df.groupby('community_id').agg({
    'anomaly_score': ['mean', 'std']
})

global_mean = community_stats['mean_error'].mean()
global_std = community_stats['mean_error'].std()
threshold = global_mean + (2 * global_std)   # μ + 2σ
```

Communities whose **mean reconstruction error** exceeds `μ + 2σ` are flagged as **anomalous**. A further size check distinguishes:
- `member_count > 10` → **"Coordinated Botnet"**
- `member_count ≤ 10` → **"Suspicious Cluster"**

---

### 6.3 Step 3 — Gemini AI Profiling

**File:** `GAT/pages/comm.py` (commented active section)

For each flagged community, the system aggregates the **user description text** of all member accounts and sends them to **Gemini 1.5 Flash** for natural-language behavioral profiling:

```python
prompt = f"""
Analyze this community (ID: {community_id}) with {member_count} members.
Structural Anomaly Score: {mean_error:.4f}
Descriptions: {context}   # Combined bio text of all members (max 2000 chars)

Provide exactly 2 parts separated by '|':
Category Name | Combined behavioral and semantic analysis (3 sentences).
"""
response = llm_model.generate_content(prompt)
comm_type, comm_desc = response.text.split("|")
```

The LLM returns structured output like:
- `Marketing-Centric Botnet | This community exhibits...`
- `Political Influence Farm | Accounts display...`

The parsed results are stored in `ll.csv` and rendered as a sortable dataframe in the Streamlit dashboard.

---

## 7. Phase 4 — Streamlit Dashboard

**Files:** `GAT/list.py` (or equivalent main file), `GAT/pages/comm.py`

The dashboard is organized into **three analysis pillars**:

### Pillar 1 — Bot Detection

Displays results from the supervised **BotGAT** model:
- Per-user bot probability scores
- Classification labels (Bot / Human)
- t-SNE cluster visualization (`tsne_output_visualization.png`)
- Filterable data table sorted by confidence

### Pillar 2 — Spammer Accounts

Powered by the **GAE Human-Baseline** model:
- Reads `spammer_accounts.csv`
- Displays node IDs, anomaly scores, and bot label cross-reference
- Histogram showing reconstruction error distribution with threshold boundary
- Accounts in the "High-Confidence Spammer Zone" highlighted in red

### Pillar 3 — Community Anomalies (`pages/comm.py`)

Coordinated attack dashboard:
- **Louvain partitioning** run on the live graph
- Community table filtered to `member_count > 5`, sorted by `mean_error`
- Reads `ll.csv` for AI-generated community type and description
- Visual evidence via `io.png` (behavioral radar charts)

#### Dashboard Data Flow

```
spammer_accounts.csv   ──► Spammer Table
ll.csv                 ──► Community AI Profiles
tsne_output_visualization.png ──► Bot Cluster Visual
io.png                 ──► Behavioral Evidence Chart
model checkpoints(.pt) ──► Live inference at page load
```

---

## 8. Data Flow Summary

```
TwiBot-22 / Cresci-2015 Raw JSON
         │
         ▼
  BotRGCN Preprocessing (preprocessing scripts)
  ├── RoBERTa → des_tensor.pt, tweets_tensor.pt
  ├── Min-Max Norm → num_properties_tensor.pt
  └── One-Hot → cat_properties_tensor.pt
         │
         ▼
  GAT/dataset.py: get_train_data()
  └── PyG Data object (4 features + edge_index + labels + splits)
         │
    ┌────┴────┐
    ▼         ▼
  train.py  human.py / gaetrain.py
  (BotGAT   (GAE trained on
  supervised) human normality)
    │         │
    ▼         ▼
  checkpoints/               inference_anomalies.py / humaninf.py
  {dataset}_{acc}.pt         └── spammer_accounts.csv
                                  spammer_dashboard_data.csv
                         │
                         ▼
                   louvian.py / comm.py
                   ├── Louvain Communities
                   ├── μ+2σ Threshold
                   ├── Gemini Profiling
                   └── ll.csv
                         │
                         ▼
                   Streamlit Dashboard
                   ├── Bot Detection
                   ├── Spammer Accounts
                   └── Community Anomalies
```

---

## 9. File Reference

| File | Role |
|---|---|
| `GAT/dataset.py` | Central data loader — assembles `torch_geometric.Data` from `.pt` tensors |
| `GAT/model.py` | Defines `BotGAT`, `BotGCN`, `BotRGCN` — supervised classifiers |
| `GAT/train.py` | Supervised training loop with `NeighborLoader` + early stopping |
| `GAT/gaemodel.py` | `GATEncoder` for the Graph Autoencoder |
| `GAT/gaetrain.py` | Trains GAE on full dataset for link prediction (baseline) |
| `GAT/human.py` | Trains GAE **on human-only edges** — defines "normality" |
| `GAT/humaninf.py` | Runs human-baseline GAE inference; exports `spammer_dashboard_data.csv` |
| `GAT/inference_anomalies.py` | Full inference with IQR thresholding; exports `detected_spammers.csv` |
| `GAT/louvian.py` | Louvain community detection + μ+2σ flagging script |
| `GAT/pages/comm.py` | Streamlit page: Community Anomaly Dashboard + Gemini integration |
| `GAT/output.py` | Extracts GAT embeddings and generates t-SNE visualization |
| `GAT/visualize.py` | Utility: graph visualization helpers |
| `GAT/ll.csv` | Pre-computed AI community profiles (Gemini output) |
| `GAT/spammer_accounts.csv` | Flagged spammer node IDs + anomaly scores |
| `GAT/checkpoints/` | Saved model weights (`.pt` files) |
| `BotRGCN/twibot_22/` | Processed TwiBot-22 tensors |
| `BotRGCN/datasets/cresci-2015/` | Raw Cresci-2015 JSON + processed tensors |

---

*Last updated: April 2026*
