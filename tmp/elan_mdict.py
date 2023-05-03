import torch
import torch.nn as nn
from yolov7.models.common import Conv, Concat

class ELAN_block(nn.Module):
    def __init__(self, chan1, chan2, ker, depth):
        super(ELAN_block, self).__init__()
        self.chan1 = chan1
        #self.c_ = chan1 // 2
        self.chan2 = chan2

        self.ker = ker
        self.depth = depth

        self.act_idx = [i for i in range(self.depth * 2) if (i % 2 == 1 or i == 0)]

        # Declaration might be useless?
        # Conv() : ch_in, ch_out, kernel, stride=1, padding=none, groups=1
        #self.conv1 = Conv(self.chan, self.chan2, ker)
        #self.conv2 = Conv(self.chan2, self.chan2, ker)

        elans = {}
        for d in range(self.depth*2):
            elans['elan_block_{0}'.format(d)] = self.construct_elan(d)

        self.elan_list = nn.ModuleDict(elans)
        self.cat = Concat(dimension=1)

    def construct_elan(self, depth):
        elan = nn.Sequential()
        #print('module',depth)
        if depth == 0:
            elan.add_module('conv_0', Conv(self.chan1, self.chan2, 1, 1))
        else:
          if depth % 2 == 0:
            for i in range(depth):
            #print('depth',depth, '/', i)
              if i == 0:
                elan.add_module('conv_{0}'.format(i), Conv(self.chan1, self.chan2, 1, 1))
              else:
                elan.add_module('conv_{0}'.format(i), Conv(self.chan2, self.chan2, self.ker, 1))
          else:
            for i in range(depth):
            #print('depth',depth, '/', i)
              if i == 0:
                elan.add_module('conv_{0}'.format(i), Conv(self.chan1, self.chan2, 1, 1))
              else:
                elan.add_module('conv_{0}'.format(i), Conv(self.chan2, self.chan2, self.ker, 1))
                elan.add_module('conv_{0}'.format(i), Conv(self.chan2, self.chan2, self.ker, 1))

        return elan
    
    def forward(self, x):
        tmp_out = []

        for _, elan in self.elan_list.items():
          #print('-----------')
          tmp_out.append(elan(x))
        
        #print(len(tmp_out))
        #out = self.cat([tmp_out[d] for d in range(self.depth*2)])
        out = self.cat([tmp_out[d] for d in self.act_idx])

        return out