# Datasets

This document describes the datasets used throughout the course, including download instructions and preprocessing steps.

---

## Sample Proteins

The following proteins are included in `data/sample/` for quick testing:

| PDB ID | Protein | Length | Description |
|--------|---------|--------|-------------|
| 1UBQ | Ubiquitin | 76 | Universal protein modifier, well-studied |
| 1CRN | Crambin | 46 | Very small plant protein, good for quick tests |
| 2GB1 | GB1 domain | 56 | IgG-binding domain, common benchmark |

### Download Script

```python
from biotite.database import rcsb
import biotite.structure.io.pdb as pdb

pdb_ids = ['1UBQ', '1CRN', '2GB1']
for pdb_id in pdb_ids:
    file_path = rcsb.fetch(pdb_id, 'pdb', target_path='data/sample/')
    print(f"Downloaded {pdb_id} to {file_path}")
```

---

## Sequence Datasets

### UniProt Sample (Lecture 1-3)
- **Size:** ~1,000 sequences
- **Use:** Sequence analysis, property prediction
- **Source:** UniProt SwissProt (reviewed)

```python
# Download sample sequences
from Bio import SeqIO
import urllib.request

url = "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=reviewed:true&size=1000"
urllib.request.urlretrieve(url, "data/sample/uniprot_sample.fasta")
```

### DeepSol Dataset (Lecture 3, 7)
- **Size:** ~62,000 sequences
- **Task:** Solubility prediction (binary classification)
- **Source:** [DeepSol Paper](https://academic.oup.com/bioinformatics/article/34/15/2605/4938490)
- **Download:** Available upon request from authors

**Preprocessing:**
```python
# Load DeepSol data
import pandas as pd

df = pd.read_csv('data/deepsol/deepsol_train.csv')
# Columns: sequence, label (0=insoluble, 1=soluble)
```

---

## Structure Datasets

### CATH S40 (Lectures 4-10)
- **Size:** ~10,000 structures
- **Description:** Non-redundant set at 40% sequence identity
- **Use:** Contact prediction, structure-based tasks

**Download:**
```bash
# Download CATH domain list
wget http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt

# Download structures via script (see src/ai4protein/data/download_cath.py)
```

### CB513 (Lecture 4)
- **Size:** 513 proteins
- **Task:** Secondary structure prediction (8-class)
- **Source:** [CB513 Paper](https://academic.oup.com/bioinformatics/article/15/11/937/258881)

**Classes:**
- H: Alpha helix
- B: Beta bridge
- E: Extended strand
- G: 3-10 helix
- I: Pi helix
- T: Turn
- S: Bend
- C: Coil

### CASP14/15 Targets (Lecture 8)
- **Use:** Structure prediction evaluation
- **Source:** [CASP Website](https://predictioncenter.org/)
- **Note:** Use for testing only, not training

---

## Preprocessed Features

### MSA Features (for AlphaFold)
Due to computational requirements, we provide pre-computed MSA features for sample proteins:

```
data/processed/
├── 1UBQ_msa.a3m
├── 1CRN_msa.a3m
└── 2GB1_msa.a3m
```

### ESM Embeddings
Pre-computed ESM-2 embeddings for rapid prototyping:

```python
import torch

# Load pre-computed embeddings
embeddings = torch.load('data/processed/esm_embeddings.pt')
# Shape: (n_proteins, max_length, 1280)
```

---

## Data Splits

For reproducibility, we use consistent train/val/test splits:

| Dataset | Train | Validation | Test |
|---------|-------|------------|------|
| DeepSol | 80% | 10% | 10% |
| CATH S40 | 80% | 10% | 10% |
| CB513 | N/A | N/A | 100% (test only) |

**Split by Structure:**
To avoid data leakage, we split by CATH superfamily to ensure train/test proteins are structurally dissimilar.

---

## Memory Requirements

| Dataset | Disk Size | Memory (loaded) |
|---------|-----------|-----------------|
| Sample proteins | ~1 MB | ~10 MB |
| UniProt sample | ~5 MB | ~50 MB |
| DeepSol | ~20 MB | ~200 MB |
| CATH S40 (structures) | ~500 MB | ~2 GB |
| ESM embeddings (sample) | ~100 MB | ~500 MB |

All datasets fit within Google Colab's 12GB RAM limit when loaded individually.

---

## Data Formats

### FASTA Format
```
>sp|P00001|PROTEIN_NAME
MKWVTFISLLFLFSSAYSRGVFRR...
```

### PDB Format
Standard Protein Data Bank format with ATOM records.

### A3M Format (MSA)
```
>query
MKWVTFISLLFLFSSAYSRGVFRR
>hit1
MK-VTFISLLFLFSSAYSRGVFRR
>hit2
MKWVTFIS--FLFSSAYSRGVFRR
```

### PyTorch Format
```python
# Graph data (for GNNs)
from torch_geometric.data import Data

data = Data(
    x=node_features,      # [N, F] node features
    edge_index=edges,     # [2, E] edge connectivity
    edge_attr=edge_feat,  # [E, D] edge features
    y=label               # target label
)
```

---

## Ethical Considerations

- All datasets are from public sources
- No patient or personal data included
- Protein structures are from published research
- Dataset licenses permit academic use
