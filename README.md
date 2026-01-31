# AI for Protein: Course Materials

**Course:** Special Topics in Smart Convergence: Protein & AI
**Institution:** KAIST
**Semester:** Spring 2026
**Instructor:** Prof. Sungsoo Ahn

## Overview

This repository contains lecture materials for the AI portion of the course, covering modern deep learning approaches for protein science. The course progresses from foundational concepts to state-of-the-art methods including AlphaFold, RFDiffusion, and ProteinMPNN.

All lecture notes are written in a **textbook narrative style** with rich explanations, biological motivations, and accessible language for students new to machine learning.

## Prerequisites

- Python programming experience
- Linear algebra and calculus fundamentals
- Basic probability and statistics
- No prior deep learning experience required

## Setup

### Option 1: Local Installation (Recommended for development)

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e .

# Using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Option 2: Google Colab

All notebooks are designed to run on Google Colab with T4 GPU. Each notebook includes a setup cell that installs required dependencies.

## Course Structure

### Foundational Lectures (Lectures 1-4)

| Lecture | Topic | Description |
|---------|-------|-------------|
| 1 | [Python & Data Basics](lectures/lec01_python_data_basics/) | NumPy, Pandas, protein file formats (FASTA, PDB) |
| 2 | [Protein Representations](lectures/lec02_protein_representations/) | Sequence encodings, structure representations, graphs |
| 3 | [AI Fundamentals](lectures/lec03_ai_fundamentals/) | PyTorch, neural networks, training loops |
| 4 | [Training & Optimization](lectures/lec04_training_optimization/) | Regularization, learning rates, debugging, protein-specific challenges |

### Architecture Lectures (Lectures 5-7)

| Lecture | Topic | Description |
|---------|-------|-------------|
| 5 | [Transformers & GNNs](lectures/lec05_neural_architectures_transformer_gnn/) | Attention mechanisms, GCN, GAT, MPNN |
| 6 | [Generative Models](lectures/lec06_generative_models/) | VAEs, diffusion models for proteins |
| 7 | [Protein Language Models](lectures/lec07_protein_language_models/) | ESM-2, embeddings, fine-tuning with LoRA |

### Case Study Lectures (Lectures 8-10)

| Lecture | Topic | Description |
|---------|-------|-------------|
| 8 | [AlphaFold](lectures/lec08_alphafold_implementation/) | Evoformer, IPA, structure prediction |
| 9 | [RFDiffusion](lectures/lec09_rfdiffusion_implementation/) | SE(3) diffusion, protein backbone generation |
| 10 | [ProteinMPNN](lectures/lec10_proteinmpnn_implementation/) | Inverse folding, sequence design |

## Directory Structure

```
.
├── lectures/                # Lecture materials
│   ├── lec01_python_data_basics/
│   │   ├── lec01_notes.md
│   │   └── notebooks/
│   ├── lec02_protein_representations/
│   ├── lec03_ai_fundamentals/
│   ├── lec04_training_optimization/
│   ├── lec05_neural_architectures_transformer_gnn/
│   ├── lec06_generative_models/
│   ├── lec07_protein_language_models/
│   ├── lec08_alphafold_implementation/
│   ├── lec09_rfdiffusion_implementation/
│   └── lec10_proteinmpnn_implementation/
├── src/ai4protein/          # Reusable Python utilities
│   ├── data/                # PDB loading, sequence utils
│   ├── models/              # Mini model implementations
│   └── utils/               # Visualization, metrics
└── data/
    ├── sample/              # Small datasets for notebooks
    └── processed/           # Processed training data
```

## Sample Proteins

The following proteins are used throughout the course:
- **1UBQ** - Ubiquitin (76 residues, well-studied model protein)
- **1CRN** - Crambin (46 residues, very small, good for quick tests)
- **2GB1** - GB1 domain (56 residues, common benchmark)

## Mini Model Implementations

This course includes educational "mini" implementations of major protein AI models:

1. **Mini AlphaFold** (~500 lines)
   - Single sequence input (no MSA)
   - 4-layer simplified Evoformer
   - Basic IPA structure module

2. **Mini RFDiffusion** (~400 lines)
   - CA-only frame diffusion
   - Simplified SE(3) transformer
   - Conditional scaffold generation

3. **Mini ProteinMPNN** (~300 lines)
   - 3-layer GNN encoder
   - Autoregressive sequence decoder

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- ~12GB GPU memory (T4 compatible)

See `pyproject.toml` for full dependency list.

## Key Resources

### Models & Tools
- [ESM-2](https://github.com/facebookresearch/esm) - Protein language models
- [AlphaFold](https://github.com/google-deepmind/alphafold) - Structure prediction
- [RFDiffusion](https://github.com/RosettaCommons/RFdiffusion) - Structure generation
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - Sequence design

### Databases
- [UniProt](https://www.uniprot.org/) - Protein sequences and annotations
- [PDB](https://www.rcsb.org/) - Protein structures
- [AlphaFold DB](https://alphafold.ebi.ac.uk/) - Predicted structures

## References

1. Jumper et al. (2021). "Highly accurate protein structure prediction with AlphaFold." *Nature*.
2. Watson et al. (2023). "De novo design of protein structure and function with RFdiffusion." *Nature*.
3. Dauparas et al. (2022). "Robust deep learning-based protein sequence design using ProteinMPNN." *Science*.
4. Lin et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [AlphaFold](https://github.com/deepmind/alphafold) - DeepMind
- [ESM](https://github.com/facebookresearch/esm) - Meta AI
- [RFDiffusion](https://github.com/RosettaCommons/RFdiffusion) - Baker Lab
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - Baker Lab
