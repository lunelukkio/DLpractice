# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:38:37 2024

@author: lunel
"""


class Lab:
    def __init__(self, lab_member):
        self.student = lab_member
        
    def degree(self, person):
        if self.student[person] >= 4:
            print('Doctor')
        elif self.student[person] < 4 and self.student[person] >= 1:
            print('Master')
        else:
            print('Bachelor')


if __name__ == '__main__':
    lab_member = {'Sato':6, 'Suzuki':5, 'Tanaka':3}
    his_and_cyto = Lab(lab_member)

    his_and_cyto.degree('Sato')
    his_and_cyto.degree('Suzuki')
    his_and_cyto.degree('Tanaka')
    
    physi_lab_member = {'Ono':0}
    physiology = Lab(physi_lab_member)
    physiology.degree('Ono')
