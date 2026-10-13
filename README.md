# vGEP

Python package to reconstruct spatial gene expression patterns (vGEPs) from single-cell RNA-seq data and a reference atlas, used in Leroy, van Leen et al., Cell Reports (2026).

Adrien Leroy, Lorette Noiret. Bellaiche Lab, Institut Curie (CNRS UMR3215, Inserm U934).

Documentation: [https://vgep.readthedocs.io](https://vgep.readthedocs.io)

The package reconstructs vGEPs, benchmarks vGEP against novoSpaRc, PERLER, Tangram and CytoSPACE, infers gene regulatory networks from transcription factor specificity, and plots the results on the tissue.

## Installation

```bash
pip install git+https://github.com/BellaicheLab/vgep.git
```

PERLER, used in the benchmark, pins `scipy==1.6.0`, which fails to build on Python 3.10 or later. Install it with `pip install "git+https://github.com/yasokochi/Perler.git" --no-deps`, then change line 348 of `perler/perler_class.py` to `DM = distance.cdist(self.__mu, self.__r, metric="mahalanobis", VI=VI)`. PERLER passes this matrix as `V=`, which SciPy 1.7 and later reject.

## Package structure

```
src/vgep/
├── spatial_transcriptome.py     # Core spatial transcriptome generation
├── processing.py                # Normalization and scaling
├── optimization/                # Reconstruction methods (vGEP, novoSpaRc, PERLER, Tangram, CytoSPACE)
├── evaluation/                  # CDF and survival curves of reconstruction accuracy
├── grn/                         # Gene regulatory network analysis
│   ├── specificity.py           # Tissue-specific TF scoring (tau, tspex)
│   └── network.py               # GRN construction, Leiden clustering, Cytoscape export
├── plotter/                     # Geometric plotting, palettes, network visualization
└── utils/                       # Data handling, MuData index utilities
```

## Citation

Leroy, van Leen *et al.*, *Morphodynamic domains enable integration of live morphometrics and spatial transcriptomics*, Cell Reports (2026). Code and data: [10.5281/zenodo.19633285](https://doi.org/10.5281/zenodo.19633285)

## License

MIT
