# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:38:37 2024

@author: lunel
"""


class HistAndCyto():
    def __init__(self):
        self.student = {
            'Kadono':5,
            'Yang':4,
            'Li':3,
            'Wang':2,
            'Lu':1
        }
        
    def degree(self, person):
        if self.student[person] >= 4:
            print('Doctor')
        elif self.student[person] < 4 and self.student[person] >= 1:
            print('Master')
        else:
            print('Bachelor')


if __name__ == '__main__':
    our_lab = HistAndCyto()
    our_lab.degree('Kadono')
    our_lab.degree('Yang')
    our_lab.degree('Li')
    our_lab.degree('Wang')
    our_lab.degree('Lu')   