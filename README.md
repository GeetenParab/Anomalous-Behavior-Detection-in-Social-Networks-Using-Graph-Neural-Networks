# Anomalous Behavior Detection in Social Networks using Graph Neural Networks

A scalable AI system designed to detect bots, spammers, and coordinated malicious behavior in social networks using Graph Neural Networks (GNNs).

## Overview
This project focuses on identifying anomalous and coordinated behavior in large-scale social networks. Instead of analyzing users individually, it models the entire platform as a graph to capture relationships and interaction patterns.

## Tech Stack

Core:
- Python
- PyTorch
- PyTorch Geometric

NLP:
- RoBERTa (for semantic embeddings)

Graph & Algorithms:
- Graph Attention Network (GAT)
- Graph Autoencoder (GAE)
- Louvain Community Detection

Frontend:
- Streamlit

Dataset:
- TwiBot-22, cresc-15

## Features

- Graph-based modeling of social networks (nodes = users, edges = interactions)
- Multi-modal feature engineering (numerical, categorical, semantic)
- Supervised bot detection using GAT
- Unsupervised anomaly detection using GAE
- Detection of coordinated communities using clustering
- Statistical anomaly detection using reconstruction error
- Interactive dashboard for visualization

## System Pipeline

- Data preprocessing and feature extraction
- Graph construction from user interactions
- Train GAT model for bot classification
- Train GAE model for anomaly detection
- Community detection using Louvain algorithm
- Identify anomalous groups using statistical thresholds
- Visualize results using Streamlit dashboard

## Results

- Accurate detection of bots and fake accounts
- Identification of unknown anomalies and coordinated attacks
- Robust performance on evolving social network behavior

## Purpose

This project was built to:
- Explore Graph Neural Networks in real-world scenarios
- Detect complex and coordinated malicious behavior
- Combine supervised and unsupervised learning approaches
- Build an end-to-end AI system

## What I Learned

- Graph-based machine learning using PyTorch Geometric
- Designing hybrid AI systems (classification + anomaly detection)
- Working with large-scale social network data
- Integrating NLP with graph models



## Repository
https://github.com/your-username/gnn-anomaly-detection
