# CS150 Project Plan
# Project Title: Graph Inpainting with Diffusion Models: A Comparative Study of Conditional GDSS and GGDiff-Style Optimal Control
## Team Members
- Haoming Su
- HyunS Lee
- Zejia You

## Motivation
Many real-world graphs suffer from missing corrupted or coarsed-grained structures, which can significantly impact the performance of graph-based machine learning models. Graph inpainting aims to recover these missing parts, enhancing the quality and utility of graph data. Recent advancements in diffusion models, particularly Conditional GDSS and GGDiff-Style Optimal Control, have shown promise in various generative tasks. This project seeks to explore and compare these two approaches for graph inpainting, providing insights into their effectiveness and potential applications.

## Problem Definition
Given a partial or corrupted graph $G' = (V', E')$, where $V'$ and $E'$ represent the observed nodes and edges respectively, the goal is to reconstruct the original graph $G_{\text{full}} \sim p(G\mid G')$ using diffusion-based generative models. The project will focus on evaluating the performance of Conditional GDSS and GGDiff-Style Optimal Control in terms of reconstruction accuracy, computational efficiency, and scalability.



## Methodology
1. Baseline: Unconditional GDSS
   - Implement the unconditional GDSS model as a baseline for comparison.
   - Train the model on the selected datasets
2. Conditional GDSS
    - Implement the Conditional GDSS model for graph inpainting, inject conditional information from the partial graph $G'$ during the diffusion process, the model learns $s_X(G_t, G') \approx \nabla_{X_t} \log p(G \mid G'), \quad s_A(G_t, G') \approx \nabla_{A_t} \log p(G \mid G')$
3. Zero-order GGDiff Guidance
   - Use unconditional GDSS as the prior, add reward function to guide the reverse SDE towards desired graph structures $dG_t = dG_t^{GDSS} + g(t)\, U(G_t, t)\, dt$
  
  
## Experiment Setup
### Datasets
We will use the subset of the following datasets for our experiments:
- QM9
- ENZYMES

### Corruption Process
To simulate missing or corrupted graph structures, we will apply the following corruption processes:
- Random Node Masking: Randomly mask out a percentage of nodes 
- Edge Removal: Randomly remove edges from the graph
- Subgraph Blanking: Remove entire subgraphs 

### Evaluation Metrics
- Node attribute reconstruction accuracy (e.g., Mean Squared Error)
- Edge reconstruction accuracy (e.g., Precision, Recall, F1-score)
- Structural similarity metrics (e.g., Graph Edit Distance, Graph Kernel-based metrics)
- Diversity of generated graphs
- Computational efficiency (e.g., training and inference time)
  
## Timeline
| Week | Task                                                                                                    |
| ---- | ------------------------------------------------------------------------------------------------------- |
| 1    | Literature review on graph inpainting and diffusion models, implement baseline unconditional GDSS model |
| 2    | Implement Conditional GDSS for graph inpainting and guidance mechanism, finalize the project            |

## References
- [GDSS Paper](https://arxiv.org/pdf/2202.02514)

- [GGDiff Paper](https://arxiv.org/pdf/2505.19685)
