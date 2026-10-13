# vGEP

vGEP (virtual Gene Expression Patterns) reconstructs the spatial transcriptome of a tissue from single-cell RNA-seq data and a reference atlas of gene expression patterns. It was developed for Leroy, van Leen et al., *Morphodynamic domains enable integration of live morphometrics and spatial transcriptomics*, Cell Reports (2026). Code and data: [10.5281/zenodo.19633285](https://doi.org/10.5281/zenodo.19633285).

Each cell is mapped to the tissue positions whose atlas expression it correlates with, and the full transcriptome is then projected onto the tissue. In the paper, 4,312 gene expression patterns of the *Drosophila* pupal notum were reconstructed from 24 atlas genes.

The package also tunes vGEP and benchmarks it against novoSpaRc, PERLER, Tangram and CytoSPACE, infers gene regulatory networks for regions of the tissue, and plots the reconstructed patterns.

## Getting started

Install the package:

```bash
pip install git+https://github.com/BellaicheLab/vgep.git
```

PERLER and Tangram, used only in the benchmark, need extra steps described in [Installation](installation.md).

The [walkthrough notebook](https://morphodomains.readthedocs.io/en/latest/notebooks/02_spatial_transcriptome_reconstruction.html) in the morphodomains documentation runs a full reconstruction on the notum data. All functions are documented in the [API reference](api/index.md).

## Citing

If you use vGEP, please cite:

- Leroy A., van Leen E. et al. (2026). Morphodynamic domains enable integration of live morphometrics and spatial transcriptomics. *Cell Reports*. Code and data: [10.5281/zenodo.19633285](https://doi.org/10.5281/zenodo.19633285)

## Related repositories

[morphodomains](https://github.com/BellaicheLab/morphodomains) reproduces the figures of the paper. [AnimalProcessing](https://github.com/BellaicheLab/AnimalProcessing) quantifies tissue mechanics from movies, and [vgep_gui](https://github.com/BellaicheLab/vgep_gui) is the web app to browse the reconstructed transcriptome ([live app](https://ybellaichelab.shinyapps.io/NotumvGEP/)).

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

installation
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

api/index
```
