# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:24:50 2019

@author: nidha
"""

import pandas as pd

dic = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
          {'a': 100, 'b': 200, 'c': 300, 'd': 400},
          {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]

data = pd.DataFrame(dic)

print(data)
print("="*50)

print(data.iloc[0:2])
