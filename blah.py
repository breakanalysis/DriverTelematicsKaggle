# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
class SumAccumulator:
    def __init__(self):
        self.values = [0]
        self.count = 0

    def add( self, val ):
        self.values.append( val )
        self.count = self.count + 1
        i = self.count
        while i & 0x01:
            i = i >> 1
            v0 = self.values.pop()
            v1 = self.values.pop()
            self.values.append( v0 + v1 )

    def get_total(self):
        return sum( reversed(self.values) )

    def get_size( self ):
        return self.count
    
x = SumAccumulator()
for i in range(100):
    x.add(np.array([1,1]))
    #print(x.values)

x.get_total()

# <codecell>


