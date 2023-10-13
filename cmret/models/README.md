## Pre-trained models

* [iso17.h5](iso17.h5) — model trained on ISO17 dataset
* [des370k.h5](des370k.h5) — model trained on DES370K dataset
* [ani1x.h5](ani1x.h5) — model trained on ANI-1x dataset
* [coll.h5](coll.h5) — toy model pre-trained on COLL dataset
* [md17.h5](md17.h5) — toy model pre-trained on combined MD17 dataset
* [ani1ccx.h5](ani1ccx.h5) — toy model pre-trained on ANI-1ccx dataset

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