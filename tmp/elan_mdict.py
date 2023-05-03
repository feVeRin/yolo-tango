import torch
import torch.nn as nn
from yolov7.models.common import Conv, Concat

class BBone_ELAN(nn.Module):
    '''
    Back Bone ELAN
    '''
    def __init__(self, chan1, chan2, ker, depth):
        super(BBone_ELAN, self).__init__()

        self.chan1 = chan1 # input channel size
        self.chan2 = chan2 # output channel size (= input_channel // 2)
        self.ker = ker # kernel size
        self.depth = depth # depth (could be variable)
        
        # index를 ELAN 정의할 때 parameter로 받으면? (헌성이형 아이디어와 비슷해보임)
        self.idx = [idx for idx in range(self.depth * 2) if (idx % 2 == 1 or idx == 0)] #include idx 0 always

        elans = {} # elan module dictionary
        for d in range(self.depth*2):
            elans['elan_block_{0}'.format(d)] = self.construct_elan(d) # ELAN block consturct

        self.elan_dict = nn.ModuleDict(elans) # constructed ELAN dictionary to ModuleDict
        self.cat = Concat(dimension=1) # from yolov7 basic modules

    def construct_elan(self, depth):
        '''
        construct ELAN modules
        '''
        elan = nn.Sequential()

        if depth == 0:
            elan.add_module('conv_0', Conv(self.chan1, self.chan2, 1, 1))

        else:
          if depth % 2 == 0:
            for i in range(depth):
              if i == 0:
                elan.add_module('conv_{0}'.format(i), Conv(self.chan1, self.chan2, 1, 1))
              else:
                elan.add_module('conv_{0}'.format(i), Conv(self.chan2, self.chan2, self.ker, 1))
          else:
            for i in range(depth):
              if i == 0:
                elan.add_module('conv_{0}'.format(i), Conv(self.chan1, self.chan2, 1, 1))
              else:
                elan.add_module('conv_{0}'.format(i), Conv(self.chan2, self.chan2, self.ker, 1))
                elan.add_module('conv_{0}'.format(i), Conv(self.chan2, self.chan2, self.ker, 1))

        return elan
    
    def forward(self, x):
        tmp_out = []
        
        # (idea?) idx에 해당하는것 먼저 select하고 for loop 돌리는게 나을지도?

        for _, elan in self.elan_dict.items(): # get each elan module output
          tmp_out.append(elan(x))
        
        out = self.cat([tmp_out[d] for d in self.idx]) # concat

        return out

#=============================================================================================

class Head_ELAN(nn.Module):
    '''
    Head ELAN
    '''
    def __init__(self, chan1, chan2, ker, depth):
        super(Head_ELAN, self).__init__()

        self.chan1 = chan1
        self.chan2 = chan2
        self._chan = chan2 // 2 # hidden channel (channel2 // 2)
        self.ker = ker 
        self.depth = depth 

        self.idx = [idx for idx in range(self.depth+1)] # use all idx

        elans = {}
        for d in range(self.depth+1):
            elans['elan_block_{0}'.format(d)] = self.construct_elan(d)

        self.elan_dict = nn.ModuleDict(elans)
        self.cat = Concat(dimension=1)

    def construct_elan(self, depth):
        elan = nn.Sequential()

        if depth == 0:
            elan.add_module('conv_0', Conv(self.chan1, self.chan2, 1, 1))

        else:
          for i in range(depth):
            if i == 0:
                elan.add_module('elan_conv_{0}'.format(i), Conv(self.chan1, self.chan2, 1, 1))
            elif i == 1:
                elan.add_module('elan_conv_{0}'.format(i), Conv(self.chan2, self._chan, self.ker, 1))
            else:
                elan.add_module('elan_conv_{0}'.format(i), Conv(self._chan, self._chan, self.ker, 1))

        return elan
    
    def forward(self, x):
        tmp_out = []

        for _, elan in self.elan_dict.items():
          tmp_out.append(elan(x))
        
        out = self.cat([tmp_out[d] for d in self.idx])

        return out