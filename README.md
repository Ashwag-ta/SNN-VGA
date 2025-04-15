# SNN-VGA
Repository for ICCABS 2025 paper : Link Prediction in Disease-Disease Interactions Network Using a Hybrid Deep Learning Model

Authors: Ashwag Altayyar and Li Liao

# Overview
We introduce SNN-VGA, a novel hybrid deep learning framework that integrates Subgraph Neural Networks (SUBGNN) and Variational Graph Auto-Encoders (VGAE) for comorbidity prediction in the disease-disease interactions (DDIs) network. In this framework, disease modules are modeled as subgraphs within the Protein-Protein Interactions (PPIs) network. Subsequently, SUBGNN is adopted to generate embeddings for these subgraphs by capturing graph properties and considering multiple disconnected components. These embeddings are then passed to VGAE, which predicts links (i.e., comorbidities) between disease nodes based on their subgraph-level representations.

# How to Use SNN_VGA
## Prerequisites
Conda must be installed on your system.
## Setup the Environment
Install the environment using the provided .yml file:
<pre lang="bash"> ```bash conda env create --file SNNVGA_environment.yml ``` </pre>


