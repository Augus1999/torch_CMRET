# Comprehensive Molecular Representation from Equivariant Transformer
CMRET is an equivarient garph neural network that maps molecular geometry to its energy and atomic forces. Note this is still an experimental project.

![OS](https://img.shields.io/badge/OS-Windows%20|%20Linux%20|%20macOS-blue?color=00b166)
![python](https://img.shields.io/badge/Python-3.9%20|%203.10-blue.svg?color=dd9b65)
![torch](https://img.shields.io/badge/torch-1.13-blue?color=708ddd)
![black](https://img.shields.io/badge/code%20style-black-black)

## API
### import model
```python
from cmret.representation import CMRETModel

model = CMRETModel()
```

### train the model
```python
from torch.utils.data import DataLoader
from cmret.utils import train, DataSet

workdir = "carbene"
dataset = DataSet("QM.CH2", "dataset", mode="train", limit=None)
trainset = DataLoader(dataset.data, batch_size=10, shuffle=True)
train(model=model, train_set=trainset, unit=dataset.unit, work_dir=workdir)
```

### test the model
```python
from cmret.utils import test, DataSet

dataset = DataSet("QM.CH2", "dataset", mode="test", limit=None).data
print(test(model=model, dataset=dataset, load=f"{workdir}/trained.pt"))
```

### running Molecular Dynamics
```python
import torch
from cmret.utils import Molecule

carbene = Molecule()
carbene.from_file("carbene.xyz")
carbene.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
carbene.calculator = model
carbene.run(temperature=298, delta_t=1e-18, step=100000)
molecules = carbene.molecule
```

## Cite
```bibtex
@article{tao2022cmret,
	title={Comprehensive Molecular Representation from Equivariant Transformer},
	author={Tao, Nianze and Morimoto, Hiromi and Leoni, Stefano},
	journal={arXiv preprint arXiv:},
	year={2022}
}
```