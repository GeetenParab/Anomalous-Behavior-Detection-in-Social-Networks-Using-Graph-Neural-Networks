# Flowcharts — Anomalous Behavior Detection System

---

## Flowchart 1 — Feature Engineering & Graph Construction

```mermaid
flowchart TD
    A([Raw Dataset\nTwiBot-22 / Cresci-2015]) --> B[node.json\nUser Profiles]
    A --> C[edge.json\nSocial Interactions]

    B --> D[Profile Bio / Description Text]
    B --> E[Numerical Fields\nfollowers, friends,\nstatuses, listed, age]
    B --> F[Binary Flags\nverified, default_profile,\ndefault_profile_image]
    B --> G[Tweet Content Text]

    D --> H["RoBERTa Encoder\n(sentence-transformers)"]
    G --> H

    H --> I[des_tensor.pt\n768-dim per user]
    H --> J[tweets_tensor.pt\n768-dim per user]

    E --> K[Min-Max Normalization]
    K --> L[num_properties_tensor.pt\n5-dim per user]

    F --> M[Binary Encoding]
    M --> N[cat_properties_tensor.pt\n3-dim per user]

    C --> O[Parse Follow /\nRetweet / Mention links]
    O --> P[edge_index.pt\nCOO Format — 2 × E]
    O --> Q[edge_type.pt\nRelation Type — E]
    A --> R[label.pt\n0=Human  1=Bot]
    A --> S[train / val / test\nsplit indices .pt]

    I & J & L & N & P & Q & R & S --> T["get_train_data()\ndataset.py"]
    T --> U([PyG Data Object\nReady for GNN])

    style A fill:#1e1e2e,color:#cdd6f4,stroke:#89b4fa
    style U fill:#1e1e2e,color:#cdd6f4,stroke:#a6e3a1
    style H fill:#313244,color:#f5c2e7,stroke:#f5c2e7
    style T fill:#313244,color:#a6e3a1,stroke:#a6e3a1
```

### Explanation

The pipeline begins with two raw JSON sources: **node profiles** and **edge lists**. Each user profile is decomposed into three independent feature streams that are processed in parallel.

- **Semantic Stream** — The user's bio description and tweet content are fed separately into a pre-trained **RoBERTa** transformer. Each piece of text is converted into a 768-dimensional dense vector that encodes contextual meaning, tone, and topic. These are pre-computed once and saved as `.pt` tensors to avoid runtime overhead.

- **Numerical Stream** — Five scalar metrics (follower count, following count, total tweets, list memberships, account age in days) are extracted and normalized to a `[0, 1]` range using min-max scaling. This prevents high-magnitude features like follower counts from dominating lower-scale features.

- **Categorical Stream** — Three boolean flags (verified badge, custom profile image set, default avatar still active) are binary-encoded as `{0, 1}`. Despite being simple, these flags are strong signals because bot farms often skip profile customization.

The **edge lists** are converted into PyTorch Geometric's COO (Coordinate) sparse format — a `[2, num_edges]` tensor — representing directed social connections. A companion tensor stores the **relation type** for each edge (e.g., follow = 0, retweet = 1), preserving the semantic nature of the relationship.

All six tensors, along with ground-truth labels and data-split indices, are assembled by `get_train_data()` into a single **PyG `Data` object** that serves as the unified input to both the GAT and GAE models.

---

## Flowchart 2 — BotGAT Supervised Model Architecture & Training

```mermaid
flowchart TD
    subgraph INPUT["Input Features — per Node"]
        A1[des_embedding\n768-dim]
        A2[tweet_embedding\n768-dim]
        A3[num_properties\n5-dim]
        A4[cat_properties\n3-dim]
    end

    subgraph PROJ["Parallel Feature Projection"]
        B1["Linear 768→H/4\n+ LeakyReLU"]
        B2["Linear 768→H/4\n+ LeakyReLU"]
        B3["Linear 5→H/4\n+ LeakyReLU"]
        B4["Linear 3→H/4\n+ LeakyReLU"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B4

    B1 & B2 & B3 & B4 --> C["Concatenate\n→ H-dim vector"]

    C --> D[Dropout 0.3]
    D --> E["Linear H→H\n+ LeakyReLU"]

    E --> F["GATConv Layer 1\nH → H/4,  heads=4\nMulti-head Attention"]
    F --> G[Dropout 0.3]
    G --> H["GATConv Layer 2\nH → H"]

    H --> I["Linear H→H\n+ LeakyReLU\n← Embedding Space"]
    I --> J["Linear H→2\nRaw Logits"]

    J --> K{Training\nor Inference?}

    K -->|Training| L["CrossEntropyLoss\nvs ground truth label"]
    L --> M["Adam Optimizer\nlr=1e-4, wd=1e-5"]
    M --> N{Val Accuracy\nImproved?}
    N -->|Yes| O[Save Best State Dict\nReset patience counter]
    N -->|No| P[Increment no_up\ncounter]
    P --> Q{no_up == 50?}
    Q -->|No| R[Next Epoch]
    Q -->|Yes| S([Save Checkpoint\n.pt file])
    O --> R

    K -->|Inference| T["Softmax → P bot\nClassify node"]
    T --> U([Bot / Human Label])

    style INPUT fill:#1e1e2e,color:#cdd6f4,stroke:#89b4fa
    style PROJ fill:#181825,color:#cdd6f4,stroke:#45475a
    style F fill:#313244,color:#f5c2e7,stroke:#f5c2e7
    style H fill:#313244,color:#f5c2e7,stroke:#f5c2e7
    style I fill:#313244,color:#89dceb,stroke:#89dceb
    style S fill:#1e1e2e,color:#a6e3a1,stroke:#a6e3a1
    style U fill:#1e1e2e,color:#a6e3a1,stroke:#a6e3a1
```

### Explanation

The `BotGAT` model is a **multi-modal fusion classifier** that learns to separate bots from humans by combining content features with social graph structure.

**Feature Projection (Parallel Branches):** Each of the four feature types enters its own linear projection layer, compressing it to `H/4 = 32` dimensions. This makes every modality contribute equally to the final representation regardless of its original dimensionality (768-dim text vs. 3-dim flags). All branches use `LeakyReLU` to preserve negative-valued gradients.

**Feature Fusion:** The four projected vectors are concatenated into a single `H = 128`-dim vector per user. This fused representation is then passed through a linear input gate with dropout (0.3) and LeakyReLU, acting as a mixing layer before the graph convolutions.

**Graph Attention Layers:** Two `GATConv` layers perform **message passing** over the social graph. The key mechanism is the **attention coefficient** — each node computes a learned importance weight for each of its neighbors before aggregating their features. Layer 1 uses 4 attention heads (`heads=4`) which independently learn different relational patterns and concatenate results; Layer 2 collapses them back to `H` dims. This allows the model to naturally prioritize influential connections (e.g., an account heavily retweeted by known bots carries more signal than a random follower).

**Training Loop:** Uses `NeighborLoader` for scalable mini-batch training with 4-hop neighborhood sampling (`[25, 15, 10, 5]`). An **early stopping** mechanism monitors validation accuracy and saves the best checkpoint in memory — training halts after 50 epochs with no improvement.

**The Embedding Space:** The penultimate layer output (before the final `Linear H→2`) captures the learned graph representation and can be visualized using **t-SNE** to confirm that bot and human clusters are separable in latent space.

---

## Flowchart 3 — GAE Spammer Detection & Community Anomaly Analysis

```mermaid
flowchart TD
    subgraph TRAIN["GAE Training — Two-Stage Strategy"]
        direction TB
        T1([Full Graph\nCresci-2015]) --> T2[RandomLinkSplit\ntrain / val / test edges]
        T2 --> T3["Stage A — Standard GAE\nTrain on all edges\n200 epochs, LR=1e-3"]
        T3 --> T4[Checkpoint A\ngae_cresci15_auc_0.98.pt]

        T1 --> T5[Filter: human_mask\nhuman indices only]
        T5 --> T6["Stage B — Human Baseline GAE\nLoss only on human↔human edges\n200 epochs"]
        T6 --> T7[Checkpoint B\ngae_human_baseline.pt]
    end

    subgraph INFER["Spammer Inference"]
        direction TB
        T7 --> I1[Load Human-Baseline Model]
        I1 --> I2["GATEncoder Forward Pass\nAll nodes → Z  N×64"]
        I2 --> I3["Decode: adj_recon\n= sigmoid Z @ Z.T"]
        I3 --> I4["Compute MSE per node\nerror = mean pow adj_real − adj_recon  2"]
        I4 --> I5{"Threshold\nΔ = Q3 + 1.5×IQR\nor 95th Percentile"}
        I5 -->|score > Δ| I6([Flagged as SPAMMER\nspammer_accounts.csv])
        I5 -->|score ≤ Δ| I7([Normal / Authentic User])
    end

    subgraph COMM["Community Anomaly Detection"]
        direction TB
        I4 --> C1[Build NetworkX Graph\nfrom edge_index]
        C1 --> C2["Louvain Algorithm\nbest_partition G\nMaximize Modularity Q"]
        C2 --> C3[Assign each node\nto Community ID]
        C3 --> C4["Aggregate per community\nmean anomaly score\nmember count"]
        C4 --> C5{"Flag Anomalous\nμ + 2σ Threshold"}
        C5 -->|mean_error > threshold\nsize > 10| C6[Coordinated Botnet]
        C5 -->|mean_error > threshold\nsize ≤ 10| C7[Suspicious Cluster]
        C5 -->|mean_error ≤ threshold| C8[Authentic Community]
        C6 & C7 --> C9["Gemini 1.5 Flash\nFeed member descriptions\nGet Type + Behavioral Summary"]
        C9 --> C10([ll.csv\nAI Community Profiles])
    end

    subgraph DASH["Streamlit Dashboard"]
        I6 --> D1[Spammer Table\nAnomaly Scores + Bot Cross-ref]
        I7 --> D2[Normal Users Feed]
        C10 --> D3[Community Anomaly Page\ncomm.py]
        C6 & C7 --> D3
    end

    style TRAIN fill:#1e1e2e,color:#cdd6f4,stroke:#89b4fa
    style INFER fill:#181825,color:#cdd6f4,stroke:#f38ba8
    style COMM fill:#181825,color:#cdd6f4,stroke:#fab387
    style DASH fill:#1e1e2e,color:#cdd6f4,stroke:#a6e3a1
    style T4 fill:#313244,color:#89b4fa,stroke:#89b4fa
    style T7 fill:#313244,color:#f5c2e7,stroke:#f5c2e7
    style I6 fill:#313244,color:#f38ba8,stroke:#f38ba8
    style C6 fill:#313244,color:#f38ba8,stroke:#f38ba8
    style C7 fill:#313244,color:#fab387,stroke:#fab387
    style C8 fill:#313244,color:#a6e3a1,stroke:#a6e3a1
    style C10 fill:#313244,color:#a6e3a1,stroke:#a6e3a1
```

### Explanation

This component operates entirely **without labels** — it detects anomalies by learning what "normal" looks like and flagging everything that deviates.

#### Two-Stage GAE Training

**Stage A (Standard Baseline):** A Graph Autoencoder is first trained using `RandomLinkSplit` — edges are hidden from the model during training, and the model earns a score based on how well it can reconstruct them (AUC ≈ 0.98). This produces a general-purpose graph encoder.

**Stage B (Human-Normality Baseline) — The critical innovation:** A second model is trained using **only human-to-human edges** in the loss function. Even though all nodes participate in the forward pass, the gradient signal comes exclusively from authentic social connections. After training, this model embodies a precise structural definition of *what a real person's social neighborhood looks like*. Any account — bot, spammer, or coordinated actor — whose neighborhood structure cannot be reconstructed accurately by this human-normality model will produce a **high reconstruction error**.

#### Spammer Detection via Reconstruction Error

At inference time, the trained encoder maps every user to a 64-dim latent vector `Z`. The decoder reconstructs the full adjacency matrix via `sigmoid(Z @ Z.T)`. The **per-node anomaly score** is the Mean Squared Error between the true adjacency row and the reconstructed row. Accounts are flagged as spammers if their score exceeds a statistical fence:
- **IQR fence:** `Q3 + 1.5 × IQR` (Tukey outlier criterion)
- **Percentile fence:** 95th percentile (stricter, used for dashboard)

Flagged accounts are saved to `spammer_accounts.csv`.

#### Community Anomaly Detection via Louvain

The same per-node anomaly scores are used at a **group level**. The social graph is converted to a NetworkX graph and the **Louvain algorithm** partitions it into communities by maximizing modularity `Q` — a measure that rewards dense intra-community connections and sparse inter-community connections. This naturally surfaces groups of accounts that interact with each other far more than with the rest of the network (a hallmark of coordinated bot behavior).

Each community is then scored by **averaging the anomaly scores of its members**. Communities whose mean score exceeds the global `μ + 2σ` threshold are considered structurally anomalous. These flagged communities — along with the combined bio text of all their members — are fed to **Gemini 1.5 Flash**, which produces a natural-language category label (e.g., *"Marketing-Centric Botnet"*) and a 3-sentence behavioral summary. Results are stored in `ll.csv` and rendered in the Streamlit community dashboard page.
