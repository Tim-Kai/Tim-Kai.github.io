---
title: "Financial Fraud Detection on DGraphFin"
excerpt: "Short description of portfolio item number 1<br/><img src='/images/500x300.png'>"
collection: portfolio
---

# Introduction and Motivation

Graphs present a natural way of modelling a large variety of phenomenon. These include social networks, financial networks, e-commerce activities and so on. GNNs can seamlessly model interactions between various users, and their activities like transactions, reviews, and posts.

In most networks, malicious users pose a great threat to the stability and experience of other normal users. Malicious users can enter a network, produce fraudulent information, participate in fraudulent transactions and facilitate other harmful activities.  Our aim is to discover these malignant users in the graph through their node features and local graph structure. We reduce the problem of Fraudulent Activity Recognition into a node-level classification task. 

Financial fraud rarely happens in isolation. Fraudsters operate in rings, move money through intermediary accounts, and leave traces not in any single row, but in the *structure* of transactions. This is exactly the regime where graph modeling wins: a suspect's risk depends on the risk of the accounts it transacts with.

This project walks through a fraud-detection system I built on the DGraphFin dataset released by Xinye and hosted on OpenI. The task is node-level fraud classification on a large, heterogeneous financial transaction graph. The core idea is simple: a graph captures who transacts with whom, which a pure tabular model never sees — but a well-engineered tabular model still carries information the graph can under-use. So I built two branches, a time-aware GraphSAGE GNN and an XGBoost model over hand-crafted features and blended their probabilities.

# Datasets and Problem Definition

DGraph-Fin is a directed, unweighted dynamic graph that represents a social network among users of Finvolution Group. In this graph, a node represents a Finvolution user, and an edge from one user to another means that the user regards the other user as the emergency contact person. An illustrative overview of dataset is provided below.

Each node in the graph is a user of FinVolution, and edges in the graph between two users (u and v) represent user u designating user v as their emergency contact. This is a directional graph between the users with 3,700,550 nodes and 4,300,999 edges. Nodes in the graph are classified as foreground nodes and background nodes. Foreground nodes are the ones that labeled as normal (Class 0) and fraud (Class 1), which are also the nodes of our prediction task. Background nodes, on the other hand, are irrelevant to the task but play an important role in maintaining the connectivity of the graph. There are 1,210,092 users labeled as Normal and 15509 users labeled as Fraud. The dataset also provides 17 anonymous features based on user demographics.

I frame the problem as graph node binary classification:

- **Nodes** = accounts (borrowers / transaction entities).
- **Edges** = interactions, each carrying an edge type and a timestamp.
- **Label** = fraud (1) or normal (0), per node.
- **Metric** = ROC-AUC, evaluated on a held-out test set of nodes.

The engineering goal is to produce a probability for every node, then threshold or rank. Fraud is heavily imbalanced, so AUC — which is rank-based — is the natural metric.


DGraphFin is large and messy in exactly the ways that make production anti-fraud hard:

| Property           | Value                              |
| ------------------ | ---------------------------------- |
| Nodes              | ~3.7M                              |
| Edge types         | 11                                 |
| Edge features      | type + timestamp                   |
| Node features      | 17 raw + engineered                |
| Label distribution | heavily imbalanced (fraud is rare) |

The main challenges I had to solve:

1. **Scale.** ~3.7M nodes mean full-batch GNN training does not fit in GPU memory. Scaling GNN to massive dataset is an indispensable factor to be taken into account.
2. **Class imbalance.** Fraud is rare; naive training degrades badly. I used focused negative sampling.
3. **Split semantics.** DGraphFin ships `train/valid/test_mask` as **index arrays**, not boolean masks over all nodes — a subtle difference that bites you the moment you index predictions. (More on this in the lessons section.)
4. **Label leakage.** Neighbor-label statistics (e.g. "what fraction of my neighbors are fraud") are extremely predictive but leak information for transductive training. I kept them out of the tabular feature branch.
5. **Timestamps.** Edges have time, and when someone transacts (sudden bursts, first/last activity) carries signal beyond raw edge counts. How to fully utilize this information and make the model incorporate this feature is tricky.

