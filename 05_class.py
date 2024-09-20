# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:42:32 2024

@author: lunel
"""

class YourClass: 
    def __init__(self):
        self.x = 5
        self.y = 1

    def your_function(self, x):
        if x > 10:
            print('positive')
        else:
            print('negative') 

    def next_func(self, y): 
        for y in range(20):
            y += 1
        return y

if __name__ == '__main__':
    your_instance = YourClass()
    print(your_instance.x)
    new_value = your_instance.next_func(your_instance.y)
    print(new_value)
    your_instance.your_function(new_value)


