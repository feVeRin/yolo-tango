import torch
import torch.nn as nn
from models.common import Conv, Concat

class BBoneELAN(nn.Module):
    def __init__(self, chan1, chan2, ker, depth):
        super(BBoneELAN, self).__init__()

        self.chan1 = chan1 # input channel size
        self.chan2 = chan2 # output channel size (= input_channel // 2)
        self.ker = ker # kernel size
        self.depth = depth # depth (could be variable)

        self.idx = [idx for idx in range(self.depth * 2) if (idx % 2 == 1 or idx == 0)] #include idx 0 always

        self.ELAN = self.construct_elan(self.depth*2)
        self.cat = Concat(dimension=1) # from yolov7 modules

    def construct_elan(self, depth):
        '''
        construct ELAN modules
        '''
        full_elan = nn.Sequential()
        elan_span = nn.Sequential()
        
        full_elan.add_module('Conv_1', Conv(self.chan1, self.chan2, 1, 1)) # 왼쪽

        for d in range(depth-1): # 오른쪽 elan span
          if d == 0:
            elan_span.add_module('Conv_{0}'.format(d+2), Conv(self.chan1, self.chan2, 1, 1))
                
          else:
            elan_span.add_module('Conv_{0}'.format(d+2), Conv(self.chan2, self.chan2, self.ker, 1))

        full_elan.add_module('Elan_span', elan_span)

        return full_elan
    
    def forward(self, x):
        tmp_out = []

        for d in range(self.depth*2):
          if d == 0:
            tmp_out.append(self.ELAN[0](x)) # 왼쪽
          else:
            tmp_out.append(self.ELAN[1][:d](x)) # elan span에서 해당 index까지 convolution된 output
  
        out = self.cat([tmp_out[d] for d in self.idx]) # concat

        return out

#============================================================

class HeadELAN(nn.Module):
    '''
    Head ELAN
    '''
    def __init__(self, chan1, chan2, ker, depth):
        super(HeadELAN, self).__init__()

        self.chan1 = chan1
        self.chan2 = chan2
        self._chan = chan2 // 2 # hidden channel (channel2 // 2)
        self.ker = ker 
        self.depth = depth 

        self.idx = [idx for idx in range(self.depth+1)] # use all idx

        self.ELAN = self.construct_elan(self.depth*2)
        self.cat = Concat(dimension=1)

    def construct_elan(self, depth):
        full_elan = nn.Sequential()
        elan_span = nn.Sequential()

        full_elan.add_module('Conv_1', Conv(self.chan1, self.chan2, 1, 1)) # 왼쪽

        for d in range(depth-1): # 오른쪽 elan span
          if d == 0:
            elan_span.add_module('Conv_{0}'.format(d+2), Conv(self.chan1, self.chan2, 1, 1))
          elif d == 1:
            elan_span.add_module('Conv_{0}'.format(d+2), Conv(self.chan2, self._chan, self.ker, 1))
          else:
            elan_span.add_module('Conv_{0}'.format(d+2), Conv(self._chan, self._chan, self.ker, 1))

        full_elan.add_module('Elan_span', elan_span)

        return full_elan
    
    def forward(self, x):
        tmp_out = []

        for d in range(self.depth*2):
          if d == 0:
            tmp_out.append(self.ELAN[0](x))
          else:
            tmp_out.append(self.ELAN[1][:d](x)) # elan span에서 해당 index까지 convolution된 output
  
        out = self.cat([tmp_out[d] for d in self.idx]) # concat

        return out
