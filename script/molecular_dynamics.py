# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
MD test
"""
import torch
from cmret.utils import Molecule
from cmret import trained_model

carbene = Molecule()
carbene.from_file("carbene.xyz")
carbene.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
carbene.calculator = trained_model(name="coll")
carbene.run(temperature=298, delta_t=1e-18, step=100000)
molecules = carbene.molecule
