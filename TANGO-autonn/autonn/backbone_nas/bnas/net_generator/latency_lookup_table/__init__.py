'''
Once for All: Train One Network and Specialize it for Efficient Deployment
Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
International Conference on Learning Representations (ICLR), 2020.
'''

import copy
import os

from .lut import LatencyEstimator
from .lut import LatencyTable