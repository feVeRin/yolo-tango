import torch
import torch.nn as nn
from yolov7.models.common import Conv, Concat

class ELAN_block(nn.Module):
    def __init__(self, chan, ker, depth):
        super(ELAN_block, self).__init__()
        self.chan = chan
        self.chan2 = chan//2

        self.ker = ker
        self.depth = depth

        # Declaration might be useless?
        # Conv() : ch_in, ch_out, kernel, stride=1, padding=none, groups=1
        #self.conv1 = Conv(self.chan, self.chan2, ker)
        #self.conv2 = Conv(self.chan2, self.chan2, ker)

        elans = {}
        for d in range(self.depth):
            elans['elan_block_{0}'.format(d)] = self.construct_elan(d)

        self.elan_list = nn.ModuleDict(elans)
        self.cat = Concat(dimension=1)

    def construct_elan(self, depth):
        elan = nn.Sequential()

        for i in range(depth):
            if i == 0:
                elan.add_module('conv_{0}'.format(i), Conv(self.chan, self.chan2, self.ker))
            else:
                elan.add_module('conv_{0}'.format(i), Conv(self.chan2, self.chan2, self.ker))

        return elan
    
    def forward(self, x):
        tmp_out = []

        for _, elan in self.elan_list.items():
          tmp_out.append(elan(x))

        out = self.cat([tmp_out[d] for d in range(self.depth)])

        return out