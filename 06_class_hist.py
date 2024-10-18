# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:38:37 2024

@author: lunel
"""

# 設計図
class Lab:
    def __init__(self, lab_member_dict):
        self.student = lab_member_dict
        
    def degree(self, person):
        if self.student[person] >= 4:
            print('Doctor')
        elif self.student[person] < 4 and self.student[person] >= 1:
            print('Master')
        else:
            print('Bachelor')
            
    def get_all_person(self):
        return self.student


if __name__ == '__main__':
    lab_member = {'Kadono':6, 'Yang':5, 'Li':3, 'Wang':2, 'Lu':1}
    # 実例１
    his_and_cyto = Lab(lab_member)
    his_and_cyto.degree('Li')

