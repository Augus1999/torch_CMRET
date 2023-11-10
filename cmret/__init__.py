# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
CMRET package.
We have added supports for TorchScript!
"""
from . import utils
from .models import trained_model, CMRETModel

__all__ = ["CMRETModel", "utils", "trained_model"]
__version__ = "1.0.0"
__author__ = "Nianze A. Tao (Omozawa Sueno)"
