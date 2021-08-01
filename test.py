import torch
import numpy as np


c=[]
for i in range(4):
    c.append(i)

list1= []
list1.append(c)
list1[-1]+=c
print(list1)