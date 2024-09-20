# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:24:30 2024

@author: lunel
"""

x = 5
if x > 10:
    print('positive')
else:
    print('negative') 
    
y = 1
for y in range(20):
    y += 1
if y > 10:
    print('positive')
else:
    print('negative') 



def cal(x):
        if x > 10:
            print('positive ' + str(x)) 
        else:
            print('negative ' + str(x))
cal(5)

y = 1
for y in range(20):
    y += 1
cal(y)

