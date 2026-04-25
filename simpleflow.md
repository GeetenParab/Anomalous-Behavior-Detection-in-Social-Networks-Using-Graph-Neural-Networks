# Simple Flowcharts — Anomalous Behavior Detection System

---

## 1. Feature Engineering & Graph Construction

```mermaid
flowchart TD
    A([Raw Dataset\nTwiBot-22 / Cresci-2015])
    A --> B[User Profiles\nnode.json]
    A --> C[Social Edges\nfollow / retweet]

    B --> D["RoBERTa Transformer\nBio + Tweet Text → 768-dim vectors"]
    B --> E[Numerical Features\nfollowers, account age → normalized]
    B --> F[Categorical Features\nverified, profile image → 0 or 1]

    C --> G[Edge Index\nCOO format tensor]

    D & E & F & G --> H([PyG Graph Data Object\nReady for GNN Training])
```

**Each user node = 4 feature types fused together. Edges represent real social links. The final PyG Data object is the input to both detection models.**

---

## 2. BotGAT — Supervised Bot Detection

```mermaid
flowchart TD
    A([User Node Features\n4 modalities]) --> B[Linear Projection\nEach feature → H/4 dims]
    B --> C[Concatenate → H-dim vector]
    C --> D["GATConv Layer 1\nAttend to neighbors, 4 heads"]
    D --> E["GATConv Layer 2\nAggregate graph context"]
    E --> F[Fully Connected\nH → 2 logits]
    F --> G{Predicted Class}
    G -->|Class 1| H([BOT 🤖])
    G -->|Class 0| I([HUMAN ✅])

    F --> J[CrossEntropyLoss]
    J --> K[Adam Optimizer\nEarly Stopping at 50 epochs]
    K --> L([Best Checkpoint Saved])
```

**The attention mechanism lets each node weigh how important each neighbor is before aggregating, making the model effective at spotting coordinated clusters.**

---

## 3. GAE + Louvain — Spammer & Community Detection

```mermaid
flowchart TD
    A([Train GAE on\nHuman-Only Edges]) --> B[Encoder learns\nnormal social structure]
    B --> C[Run on ALL users\nEncode → Reconstruct Graph]
    C --> D[Compute Reconstruction\nError per node]

    D --> E{Error > Threshold?}
    E -->|Yes| F([Flagged as SPAMMER 🚨])
    E -->|No| G([Normal User ✅])

    D --> H[Build NetworkX Graph]
    H --> I[Louvain Algorithm\nGroup into Communities]
    I --> J{Community Mean\nError > μ + 2σ?}
    J -->|Yes| K[Anomalous Community]
    J -->|No| L[Authentic Community]

    K --> M["LLM \nProfile the group behavior"]
    M --> N([Community Type +\nBehavioral Summary])
```

**High reconstruction error = the model couldn't rebuild this user's social pattern from its learned idea of "normal". At community level, Louvain finds tightly-coordinated groups, and Gemini labels them in plain language.**

