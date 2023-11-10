## Pre-trained models

* [iso17.pt](iso17.pt) — model trained on ISO17 dataset
* [des370k.pt](des370k.pt) — model trained on DES370K dataset
* [ani1x.pt](ani1x.pt) — model trained on ANI-1x dataset
* [coll.pt](coll.pt) — toy model pre-trained on COLL dataset
* [md17.pt](md17.pt) — toy model pre-trained on combined MD17 dataset
* [ani1ccx.pt](ani1ccx.pt) — toy model pre-trained on ANI-1ccx dataset

### usage
```python
from cmret import trained_model

model = trained_model("iso17")  # ISO17 model
model = trained_model("dimer")  # DES370K model
model = trained_model("coll")   # COLL model
model = trained_model("md17")   # MD17 model
model = trained_model("ccsd")   # ANI-1ccx model
model = trained_model("ani1x")  # ANI-1x model
```