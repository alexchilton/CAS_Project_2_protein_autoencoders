# Protein Graph VAE — CAS AML Module 3 Project
**University of Bern — CAS in Advanced Machine Learning**

An unsupervised deep learning project applying **Graph Variational Autoencoders** to protein structure data. Proteins are modelled as molecular graphs and a VAE is trained to learn compact latent representations of their 3D structure — enabling latent space analysis, clustering, and generation.

---

## Overview

Rather than treating proteins as sequences, this project represents them as **graphs** where:
- **Nodes** = C-alpha atoms, with 28-dimensional feature vectors (coordinates, amino acid type, physicochemical properties)
- **Edges** = spatial proximity (K-nearest neighbours or distance threshold ≤ 5 Å)

A Graph VAE then learns to encode these variable-sized protein graphs into a fixed-dimensional latent space and reconstruct node features from it.

---

## Architecture

```
Protein PDB → ProteinAnalyzer → GraphCreator → Graph VAE → Latent Space
                                                    ↓
                                          Reconstruction / Generation
```

### Encoder
- Two-layer **GCNConv** (Graph Convolutional Network)
- Global mean pooling → fixed-dimension latent vector
- Outputs mean (μ) and log-variance (σ²) for reparameterisation

### Decoder
- MLP that reconstructs node features from sampled latent vector
- Trained with reconstruction loss + KL-divergence

---

## Feature Engineering

Each node (C-alpha atom) has a **28-dimensional feature vector**:

| Feature Group | Description |
|--------------|-------------|
| 3D coordinates | X, Y, Z position |
| Amino acid type | One-hot encoding (20 standard amino acids) |
| Physical properties | Average mass, neighbourhood distance metrics, neighbour counts |

---

## Latent Space Analysis

A key goal is validating that the learned latent space captures meaningful protein structure:
- **t-SNE / UMAP** visualisation of embeddings coloured by protein family
- Tracking latent space evolution across training epochs
- Density analysis of learned representations
- 3D interactive protein visualisation (Plotly) paired with embeddings

If the VAE is learning effectively, proteins from the same structural family should cluster together in the latent space.

---

## Repository Structure

```
├── Graph_VAE_protein.ipynb      # Main training notebook
├── protein_notebook2.ipynb      # 3D Plotly visualisation experiments
├── protein_analyzer.py          # PDB parsing, C-alpha extraction, feature computation
├── graph_creator_onehotencoder.py  # Graph construction, edge definition, padding
├── utils.py                     # Plotting and helper functions
├── prep.py                      # Dataset preparation (copy/filter PDB files)
└── requirements.txt             # Dependencies
```

---

## Getting Started

```bash
pip install -r requirements.txt

# Prepare dataset — copy smallest N PDB files for training
python prep.py  # edit N inside to control dataset size

# Run main training notebook
jupyter notebook Graph_VAE_protein.ipynb
```

---

## Course Context

**Programme:** CAS in Advanced Machine Learning, University of Bern — Module 3  
**Focus:** Generative models, graph neural networks, unsupervised learning  
**Key Libraries:** PyTorch, PyTorch Geometric, Biopython, scikit-learn, Plotly, UMAP

This project established the Graph VAE baseline extended in the [CAS AML Final Project](https://github.com/alexchilton/CAS_AML_Final_Project), which adds conditional generation, pH-optimised protein design, and GRAN-based dual generative modelling.
