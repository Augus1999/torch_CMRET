## usage
```python
from ase.io import read
from cmret.representation import CMRETModel

model = CMRETModel.from_checkpoint("isotropic_polarizability.pt")
atoms = read("your_mol.sdf")
z = torch.tensor(atoms.numbers, dtype=torch.long)[None, :]
r = torch.tensor(atoms.positions, dtype=torch.float32)[None, :, :]
mol = {"Z": z, "R": r, "batch": torch.ones_like(z)[:, :, None]}
out = model(mol)
print("predicted:", out["scalar"].item(), model.unit)
```