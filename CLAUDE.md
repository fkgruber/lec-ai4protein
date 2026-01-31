# AI4Protein Course Materials - Development Guidelines

## Project Overview
Lecture materials for "Special Topics in Smart Convergence: Protein & AI" (KAIST Spring 2026)
Instructor: Prof. Ahn

## Language & Environment
- Primary language: Python 3.10+
- Package manager: `uv` (preferred) or pip
- Deep learning framework: PyTorch 2.0+
- All notebooks must run on Google Colab T4 GPU (<12GB memory)

## Coding Style
- Keep code educational and well-documented
- Prioritize readability over optimization
- Use type hints for function signatures
- Include docstrings for public functions
- Follow PEP 8 conventions

## Notebook Guidelines
- Each notebook should be self-contained (can run independently)
- Start with necessary imports and setup cell
- Include clear markdown explanations between code cells
- Add "Expected output" comments for verification
- Target runtime: <30 minutes per notebook on Colab

## Directory Structure
```
lectures/lecXX_topic/
├── lecXX_notes.md     # Detailed lecture notes in Markdown
└── notebooks/         # Jupyter notebooks with exercises
    ├── 01_topic.ipynb
    ├── 02_topic.ipynb
    └── ...

src/ai4protein/
├── data/              # Data loading utilities (PDB, FASTA, graphs)
├── models/            # Mini implementations (AlphaFold, RFDiffusion, etc.)
└── utils/             # Visualization, metrics, geometry helpers
```

## Lecture Content Guidelines
- Each lecture note should include:
  - Learning objectives
  - Prerequisites
  - Main content with equations (LaTeX)
  - Code examples
  - References and further reading
- Use consistent notation across all lectures
- Include "Key Takeaways" summary at the end

## Lecture Writing Style (Textbook Format)

Lectures follow a **textbook/blog narrative style** (like alchemybio.substack.com), NOT slide-deck bullet points.

### Style Principles
- **~60% explanation, ~40% technical content** (code/math)
- Rich explanatory paragraphs that build understanding
- Accessible to undergraduates with no prior ML knowledge
- Define jargon when first introduced

### Structure for Each Section
1. **Motivating question or hook** - Start with "why does this matter?"
2. **Conceptual explanation** - Intuition and analogies BEFORE formalism
3. **Technical content** - Math/code with context
4. **Key insight summary** - Crystallize the main takeaway

### Before Code Blocks
Always explain:
- What problem are we solving?
- Why this approach?
- What should the reader expect?

### Biological Motivations
Connect every concept to proteins:
- "Distance matrices matter because proteins are defined by 3D structure..."
- "Attention helps because co-evolution tells us which positions interact..."
- "SE(3) equivariance matters because rotating a protein doesn't change its function..."

### Example Transformation
**BAD (slide-deck style):**
```markdown
## 1.2 Broadcasting
Broadcasting allows operations between arrays of different shapes:
```python
coords = np.random.randn(100, 3)
centroid = coords.mean(axis=0)
centered = coords - centroid
```
```

**GOOD (textbook style):**
```markdown
## 1.2 Broadcasting: Operating on Mismatched Shapes

One of NumPy's most powerful features is broadcasting—the ability to perform operations between arrays of different shapes. This might sound abstract, but it solves a problem you'll encounter constantly in protein work.

Consider centering a protein structure. You have 100 atoms, each with x, y, z coordinates, and you want to move the protein so its center of mass is at the origin. Mathematically, this means subtracting the mean position from every atom.

The naive approach would loop through each atom—but that's slow and un-Pythonic. Broadcasting lets us do it elegantly:

```python
coords = np.random.randn(100, 3)    # 100 atoms, xyz coordinates
centroid = coords.mean(axis=0)       # Shape: (3,) - one mean per dimension
centered = coords - centroid          # Broadcasting: (100, 3) - (3,) → (100, 3)
```

**Key insight:** NumPy automatically "stretches" the smaller array to match the larger one. The centroid (shape `(3,)`) is broadcast across all 100 atoms, subtracting the same x-mean from every x-coordinate.
```

### Lecture Directory Structure
```
lectures/
├── lec01_python_data_basics/
├── lec02_protein_representations/
├── lec03_ai_fundamentals/
├── lec04_training_optimization/      # Training, regularization, debugging
├── lec05_neural_architectures_transformer_gnn/
├── lec06_generative_models/
├── lec07_protein_language_models/
├── lec08_alphafold_implementation/
├── lec09_rfdiffusion_implementation/
└── lec10_proteinmpnn_implementation/
```

### When Updating Lectures
1. Read the existing content first to understand structure
2. Preserve all code examples (they are tested)
3. Maintain the narrative flow—don't regress to bullet points
4. Keep biological motivations and analogies
5. Target word counts: Lec1-3 (~3-5K), Lec4-7 (~5-6K), Lec8-10 (~5-7K)

## Git Workflow
- Use conventional commit format:
  - `feat:` new lecture content or notebooks
  - `fix:` corrections to existing materials
  - `docs:` documentation updates
  - `refactor:` code restructuring
  - `test:` adding tests

## Sample Data Guidelines
- Include small protein examples (1UBQ, 1CRN, 2GB1)
- Keep sample datasets under 50MB total
- Document data sources and preprocessing steps

## Testing
- Verify all notebooks execute end-to-end
- Run on fresh Colab runtime before release
- Test toy model training converges on sample data
