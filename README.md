🚀 Anomalous Behavior Detection in Social Networks Using Graph Neural Networks

A scalable, multi-layered security framework that leverages Graph Neural Networks (GNNs) to detect malicious and coordinated behavior in large-scale social networks.

📌 Overview

This project addresses the limitations of traditional security systems in modern social platforms by shifting from individual user analysis to graph-based relational modeling. It detects both known threats (bots) and unknown anomalies (coordinated attacks) using a hybrid deep learning approach.

🎯 Key Features

🔗 Graph-Based Modeling
Represents users as nodes and interactions (follows, mentions, retweets) as edges to capture network dynamics.

🧠 Multi-Modal Feature Engineering
Combines:

Numerical features (followers, activity rate)

Categorical metadata (verification status, profile info)

Semantic embeddings (via RoBERTa)

🤖 Dual Detection Engine

Graph Attention Network (GAT) → Supervised bot detection

Graph Autoencoder (GAE) → Unsupervised anomaly detection

🌐 Community-Level Threat Detection

Louvain clustering for detecting dense user communities

Statistical anomaly scoring using reconstruction error

🧾 AI-Powered Profiling

Integrates Gemini AI for generating human-readable summaries of suspicious communities



Community Anomalies

(Upcoming) Compromised Accounts
