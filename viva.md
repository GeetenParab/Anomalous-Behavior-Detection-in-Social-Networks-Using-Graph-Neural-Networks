# Viva Q&A — Anomalous Behavior Detection Using GNNs

---

## Q1. Where did you find the dataset?

**TwiBot-22:**
Found on the official website **[twibot22.github.io](https://twibot22.github.io/)** and the corresponding GitHub repo at [github.com/LuoUndergradXJTU/TwiBot-22](https://github.com/LuoUndergradXJTU/TwiBot-22).
It was introduced in the paper **"TwiBot-22: Towards Graph-Based Twitter Bot Detection"** (Feng et al., NeurIPS 2022, arXiv: 2206.04564).
Access is **restricted** — you must email `shangbin@cs.washington.edu` with your institution, advisor, and use case to get the download link.

**Cresci-2015:**
Published by Cresci et al. in the paper *"Fame for sale: efficient detection of fake Twitter followers"* (2015). Widely available through academic repositories and the TwiBot-22 benchmark suite.

---

## Q2. Size of the Dataset?

| Dataset | Users | Edges | Notes |
|---|---|---|---|
| **TwiBot-22** | ~1,000,000 users | ~170 million edges | Large-scale, real Twitter data with ground-truth labels |
| **Cresci-2015** | ~7,251 labeled users | Follow relationships | Smaller, classic benchmark, well-curated |

TwiBot-22 is approximately **5× larger** than its predecessor TwiBot-20.
Cresci-2015 is used in this project for GAE training due to its manageable size and clean graph structure.

---

## Q3. Format of the Dataset?

Both datasets follow the same file structure:

| File | Content |
|---|---|
| `node.json` | User metadata (profile, bio, tweet text, account stats) |
| `edge.csv` | Source → Target pairs with relation type (follow, retweet, mention) |
| `label.csv` | 0 = human, 1 = bot for each user node |
| `split.csv` | Pre-defined train / val / test splits |

TwiBot-22's `node.json` is further split into `user.json`, `tweet.json`, `list.json`, `hashtag.json`.

---

## Q4. How Can We Create a Dataset Using API, and What Are the Privacy Issues?

### Creating a Dataset via Twitter API:

You would use the **Twitter API v2** (now X API):
1. Apply for **Academic Research access** (required for large-scale collection)
2. Use endpoints like:
   - `/2/users/:id` — fetch user profiles
   - `/2/users/:id/tweets` — fetch tweets
   - `/2/users/:id/followers` and `/following` — build the follow graph
3. Store user IDs, metadata, tweet text, and edges in JSON/CSV

**Rate Limits are a major challenge:**
- The API has strict rate limits (e.g., 500,000 tweets/month on basic tier)
- TwiBot-22 was collected over a long period using multiple API keys and distributed crawling to bypass limits
- This is confirmed in the GitHub issue #22 of the TwiBot-22 repo, where the author also acknowledged this challenge

### Privacy Issues (from GitHub Issue #22):
- Twitter's **Terms of Service** allow sharing only **User IDs**, not raw tweet content, in public datasets
- TwiBot-22 complies by distributing only IDs; users must "hydrate" (re-fetch) content via API themselves
- **GDPR / data privacy**: User bios and tweets are public but aggregating them at scale raises ethical concerns
- Deleted or suspended accounts cannot be re-fetched — this causes **data staleness**
- The edge `followed` relation in `edge.csv` had source/target reversed compared to the paper — a known issue noted in Issue #22

---

## Q5. How Is Data Preprocessed?

Raw JSON → Processed `.pt` tensor files via preprocessing scripts in `BotRGCN/`:

1. **Text Preprocessing:**
   - User bio and tweet text are cleaned (remove URLs, special characters)
   - Passed through **RoBERTa** (sentence-transformers) to generate 768-dim embeddings
   - Saved as `des_tensor.pt` and `tweets_tensor.pt`

2. **Numerical Features:**
   - `followers_count`, `friends_count`, `statuses_count`, `listed_count`, `account_age`
   - Normalized using **min-max or z-score normalization**
   - Saved as `num_properties_tensor.pt`

3. **Categorical Features:**
   - `verified`, `default_profile`, `default_profile_image` → binary {0, 1}
   - Saved as `cat_properties_tensor.pt`

4. **Graph:**
   - Edge list parsed from `edge.csv` → COO format `edge_index` tensor
   - Relation types encoded numerically

5. **Labels & Splits:**
   - `label.pt` from `label.csv`
   - `train_idx`, `val_idx`, `test_idx` from `split.csv`

---

## Q6. How Were These Features Selected?

Features were selected based on **prior research in bot detection** and **domain knowledge** about how bots behave differently from humans:

| Feature | Reason for Selection |
|---|---|
| Bio description | Bots often have generic/copied or empty bios |
| Tweet content | Bots post repetitive, spammy, or politically driven content |
| Followers/Following | Bots often have unnatural ratios (many following, few followers) |
| Statuses count | Bots post at abnormal volumes |
| Account age | Fresh accounts are suspicious |
| Verified | Real humans are more likely to be verified |
| Default profile/avatar | Bots skip profile customization |

The graph structure (edges) was selected because **coordinated botnets are invisible from single-user features alone** — only the graph reveals their coordinated behavior.

---

## Q7. How Is Data Converted Into Vectors?

### Tool Used: RoBERTa via `sentence-transformers`

```
User Bio Text / Tweet Text
        ↓
  Tokenization (BPE, vocab size 50K)
        ↓
  RoBERTa Transformer (12 layers, 768 hidden)
        ↓
  [CLS] token embedding → 768-dim vector
        ↓
  Saved as .pt tensor file
```

### Why RoBERTa and Not BERT?

| Property | BERT | RoBERTa |
|---|---|---|
| Masking strategy | Static (fixed masks) | Dynamic (masks change each epoch) |
| NSP task | Yes (sometimes harmful) | Removed |
| Training data | Wikipedia + BookCorpus | Larger + more diverse internet text |
| Vocabulary | 30K WordPiece tokens | 50K BPE tokens |
| Social media fit | Moderate | Better (handles slang, hashtags, rare words) |

**Output dimension: 768 per user** (for both bio and tweets separately)
- RoBERTa's dynamic masking forces it to learn more robust, context-aware representations
- Its larger vocabulary handles Twitter-specific tokens like hashtags, mentions, slang better than BERT

### Other Features → Vectors:
- Numerical: raw floats → normalized scalar vector (5-dim)
- Categorical: binary flags → {0,1} vector (3-dim)
- All four vectors concatenated per user before input to GNN

---

## Q8. What Is a Neural Network and How Does GNN Work?

### Neural Network (in short):
A neural network is a system of layers of mathematical functions (neurons) that learn patterns from data. Each layer transforms the input using learned weights.
```
Input → [Layer 1] → [Layer 2] → ... → Output
```
Training: adjust weights to minimize error using backpropagation + gradient descent.

### How GNN Works:
A **Graph Neural Network** extends this to graph-structured data. Instead of processing nodes independently, each node **aggregates information from its neighbors**.

```
Step 1: Each node starts with its own feature vector
Step 2: Message Passing — each node collects feature vectors from neighbors
Step 3: Aggregation — sum/mean/attention over neighbor messages
Step 4: Update — node updates its own embedding
Step 5: Repeat for k layers (k-hop neighborhood)
```

After `k` layers, each node's embedding captures information about its `k`-hop neighborhood.

**Why GNNs for bot detection?**
Because bots are not isolated — they exist in a social graph. Their connections reveal patterns (e.g., all following the same account, retweeting each other) that flat feature tables miss.

---

## Q9. How Does the GAT (BotGAT) Model Work?

**GAT = Graph Attention Network**

The key idea: instead of treating all neighbors equally, **learn attention weights** so important neighbors contribute more.

### Architecture Summary:

```
4 Feature Modalities
  ↓ Each: Linear(input_dim → H/4) + LeakyReLU
  ↓ Concatenate → [N, H]   (H = 128)
  ↓ Dropout + Linear(H, H) + LeakyReLU
  ↓ GATConv(H → H, 4 attention heads)   ← Layer 1
  ↓ Dropout
  ↓ GATConv(H → H)                       ← Layer 2
  ↓ Linear(H, 2) → Logits
  ↓ CrossEntropyLoss → Bot / Human
```

### GATConv (Attention Mechanism):
For node `i` attending to neighbor `j`:
```
attention(i,j) = softmax( LeakyReLU( a · [Wh_i || Wh_j] ) )
new_h_i = sum over j of ( attention(i,j) × Wh_j )
```
- Uses **4 attention heads** (multi-head) in Layer 1 → richer representation
- This allows the model to focus on the most relevant neighbors, e.g., ignoring a bot's legitimate followers

### Training Parameters:

| Parameter | Value | Meaning |
|---|---|---|
| `hidden_dim` | 128 | Size of embedding at each layer |
| `learning_rate` | 1e-4 | Step size for gradient descent |
| `weight_decay` | 1e-5 | L2 regularization to prevent overfitting |
| `dropout` | 0.3 | 30% neurons randomly dropped during training |
| `max_epoch` | 1000 | Maximum training iterations |
| `early_stop patience` | 50 | Stop if no val improvement for 50 epochs |
| `batch_size` | 128 | Nodes per mini-batch |
| `num_neighbors` | [25,15,10,5] | 4-hop neighbor sampling fanout |
| Optimizer | Adam | Adaptive learning rate optimizer |
| Loss | CrossEntropyLoss | Standard classification loss |

**NeighborLoader** is used for mini-batch training — it samples subgraphs so the full 1M-node graph doesn't need to fit in memory.

---

## Q10. How Does the GAE Model Work?

**GAE = Graph Autoencoder** — an unsupervised model.

### Core Idea:
Train the model to **reconstruct the social graph** from node embeddings. A node that doesn't fit the learned "normal" pattern will have **high reconstruction error**.

### Architecture:

```
Encoder (GATEncoder):
  Same 4-modality projection as BotGAT
  → GATConv → GATConv → Z [N, 64]   (64-dim latent embedding)

Decoder (Inner Product):
  adj_reconstructed = sigmoid(Z @ Z.T)
```

### Anomaly Score:
```
score[i] = MSE(adj_real[i], adj_reconstructed[i])
         = mean of (true edge - predicted edge)² for node i
```

### Key Innovation — Human-Normality Baseline:
- Instead of training on the full graph, trained **only on human-to-human edges**
- The model learns what authentic social behavior looks like
- Any bot, spammer, or coordinated actor **deviates from this pattern → high score**
- This makes the model detect **novel/unseen bots** without needing labels

### Training:
- Dataset: Cresci-2015 (smaller, cleaner for GAE)
- `RandomLinkSplit`: holds out 5% val edges + 10% test edges, trains on rest
- Trained for ~200 epochs, achieves AUC ~0.98

### Thresholding (flagging spammers):
- **IQR method:** flag if score > Q3 + 1.5 × IQR (standard Tukey fence)
- **Percentile method:** flag top 5% by score
- Output → `spammer_accounts.csv`

---

## Q11. How Does Louvain Community Detection Work?

### What Is Community Detection?
Grouping nodes in a graph such that connections within a group are **denser** than between groups.

### Louvain Algorithm (in short):

1. **Phase 1 — Local Optimization:**
   Each node tries moving to a neighbor's community. Keep the move if it **increases modularity Q**.
   ```
   Q = (edges within communities) / (total edges)
     - (expected edges within communities in random graph)
   ```

2. **Phase 2 — Aggregation:**
   Each community is collapsed into a single "super-node". Repeat Phase 1 on the reduced graph.

3. Repeat until modularity no longer improves.

### Application in Our Project:

```python
partition = community_louvain.best_partition(G, random_state=42)
```

- Naturally groups **coordinated botnets** together (they follow/retweet each other heavily)
- Then compute `mean anomaly score` per community
- Flag communities where `mean_score > μ + 2σ` of all communities
- `members > 10` → **"Coordinated Botnet"**
- `members ≤ 10` → **"Suspicious Cluster"**

---

## Q12. Word Cloud and Community Profiling

### Word Cloud:
A visual representation where **word size = word frequency**. In our project:
- Member bios/descriptions of each flagged community are aggregated
- Common words (stop words removed) are visualized as a word cloud
- Reveals the **thematic focus** of a botnet (e.g., "crypto", "trading", "political")

### Community Profiling via Gemini AI:

For each anomalous community, the system sends aggregated member bio text to **Gemini 1.5 Flash**:

```
Prompt:
  Community ID, member count, anomaly score, combined bio text (max 2000 chars)
  → Return: "Category Name | 3-sentence behavioral analysis"

Example Output:
  "Political Influence Farm | Accounts display synchronized posting behavior..."
```

Results stored in `ll.csv` and shown in the Streamlit dashboard. This turns a raw anomaly score into a **human-readable threat category** that a security analyst can act on.

---

## Q13. Future Use of This Project

1. **Real-time Detection** — Deploy as a streaming pipeline (Twitter Filtered Stream API → live graph → inference)
2. **Cross-platform Extension** — Apply to Instagram, Facebook, Reddit graphs with similar structure
3. **Disinformation Campaign Detection** — Track coordinated political influence operations
4. **Financial Fraud Detection** — Adapt graph structure for transaction networks (banks, crypto)
5. **Content Moderation Support** — Provide bot probability scores to platform trust & safety teams
6. **Federated Learning** — Train across platforms without sharing raw user data (privacy-preserving)
7. **Temporal GNNs** — Use evolving graph models to detect bots that change behavior over time

---

## Q14. Traditional ML vs GNN Models

| Aspect | Traditional ML (SVM, Random Forest, etc.) | GNN (GAT, GAE) |
|---|---|---|
| Input | Per-user feature vector only | Feature vector + graph structure |
| Coordination Detection | ❌ Cannot detect coordinated botnets | ✅ Detects via graph structure |
| Context | Each user analyzed in isolation | Each user informed by neighbors |
| New Bot Types | Needs re-training with new labeled data | GAE detects without labels |
| Interpretability | High (feature importance, rules) | Lower (attention weights partially help) |
| Scalability | Fast on large datasets | Requires graph sampling (NeighborLoader) |
| Performance (benchmark) | ~85–88% accuracy | ~90–95% accuracy on TwiBot-22 |

**Key point:** Traditional ML fails on **novel bots** that mimic human behavior at the individual level but reveal themselves through **collective graph patterns**.



---

## Additional Important Questions

**Q: What is t-SNE and why is it used?**
t-SNE (t-distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that maps high-dimensional embeddings (128-dim) to 2D for visualization. Used to visually confirm that BotGAT has learned separable bot vs. human clusters in latent space.

**Q: What is the edge_index and why COO format?**
`edge_index` is a `[2, num_edges]` tensor where row 0 = source nodes and row 1 = destination nodes. COO (Coordinate format) is the standard sparse graph format in PyTorch Geometric — memory-efficient for sparse graphs where most pairs of nodes have no edge.

**Q: What is modularity Q in Louvain?**
Q measures how much denser within-community connections are compared to a random graph with the same degree sequence. Ranges from -1 to 1; higher Q = better community structure.

**Q: Why use two separate models (BotGAT + GAE) instead of one?**
BotGAT needs labeled data and catches known bot types. GAE needs no labels and catches novel, unseen anomalies. Together they cover both the known and unknown threat space — neither model alone is sufficient.

**Q: What is NeighborLoader?**
A mini-batch sampling strategy from PyTorch Geometric. For each seed node, it samples a fixed number of neighbors at each hop. Allows training on graphs too large to fit in GPU memory by processing subgraphs.

**Q: Why use Adam optimizer?**
Adam (Adaptive Moment Estimation) adapts the learning rate per parameter using first and second moment estimates of gradients. More stable and faster than SGD for sparse, high-dimensional graph data.


