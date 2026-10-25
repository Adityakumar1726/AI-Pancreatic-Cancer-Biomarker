# AI-Based Pancreatic Cancer Biomarker Identification

## Project Overview

This project aims to identify and prioritise potential pancreatic cancer biomarkers using gene-expression data, biological interaction networks, and artificial intelligence.

The project uses Relational Graph Convolutional Networks (RGCN) to learn graph-based representations of RNA molecules and a Deep Q-Network (DQN)-based approach to prioritise candidate nodes.

## Objectives

- Analyse pancreatic cancer RNA-expression data.
- Identify differentially expressed mRNA, lncRNA, and miRNA.
- Construct a multi-relational biological interaction graph.
- Learn node embeddings using RGCN.
- Prioritise candidate nodes using a DQN-based approach.
- Integrate multiple evidence sources to rank potential biomarkers.
- Perform independent validation using a GEO dataset.

## Dataset

The project uses TCGA-PAAD (The Cancer Genome Atlas — Pancreatic Adenocarcinoma).

The current pipeline includes mRNA, lncRNA, and miRNA expression data.

Raw datasets are not included in this repository. Refer to the original data providers and their data-use terms to obtain the data.

## Methodology

1. Data collection and preprocessing
2. Differential expression analysis
3. Biological graph construction
4. RGCN-based node representation learning
5. DQN-based candidate prioritisation
6. Evidence score integration
7. Candidate biomarker ranking
8. Independent validation

## Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- pandas
- NumPy
- SciPy
- scikit-learn
- NetworkX
- Matplotlib
- Seaborn

## Current Phase

Phase 1 focuses on gene-based biomarker prioritisation using RGCN and DQN.

## Future Work

The proposed Phase 2 extension includes:

- Clinical data analysis
- Medical image analysis using computer vision
- Investigation of multimodal integration, subject to suitable data availability and patient-level matching

## Disclaimer

This project is for academic and research purposes only. The computationally prioritised candidates are not clinically validated diagnostic biomarkers.

## Authors

Aditya Kumar  
Prasanth M.

Department of Computer Science and Engineering — AI & Data Science  
VISTAS, Chennai
