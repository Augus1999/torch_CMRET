# Comprehensive Molecular Representation from Equivariant Transformer
CMRET provides spin (S) and/or molecular charge (Q) awared machine learning force fields. Note this is still an experimental project.

![OS](https://img.shields.io/badge/OS-Windows%20|%20Linux%20|%20macOS-blue?color=00b166)
![python](https://img.shields.io/badge/Python-3.9%20|%203.10-blue.svg?color=dd9b65)
![torch](https://img.shields.io/badge/torch-2.0-blue?color=708ddd)
![black](https://img.shields.io/badge/code%20style-black-black)
[![arxiv](https://img.shields.io/badge/arXiv-2308.10752-red)](https://arxiv.org/abs/2308.10752)

## Requirements
```txt
torch>=2.0.0
ase>=3.22.0
lightning>=2.1.2
```

## Usage
See examples:

[train and test on singlet/triplet CH<sub>2</sub> dataset](script/run_ch2.py)

[train and test on ISO17 dataset](script/run_iso17.py)

[running molecular dynamic simulation](script/molecular_dynamics.py)

## Known issue
* Using a compiled TorchScript Module for MD simulation (ASE-based) will lead to _RuntimeError_.

## Cite
```bibtex
@misc{2023cmret,
      title={Comprehensive Molecular Representation from Equivariant Transformer}, 
      author={Nianze Tao and Hiromi Morimoto and Stefano Leoni},
      year={2023},
      eprint={2308.10752},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph}
}
```
