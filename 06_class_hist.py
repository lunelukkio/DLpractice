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
    lab_member = {'Sato':6, 'Suzuki':5, 'Tanaka':3}
    # 実例１
    his_and_cyto = Lab(lab_member)
    his_and_cyto.degree('Sato')
    his_and_cyto.degree('Suzuki')
    his_and_cyto.degree('Tanaka')
    
    # 実例２
    physi_lab_member = {'Ono':0, 'Saito':3}
    physiology = Lab(physi_lab_member)
    physiology.degree('Ono')
    
    # python recognaize a valiable as function
    print(his_and_cyto.get_all_person()) 
    # python recognaize a valiable as object
    print(physiology.get_all_person)  
    
    # 実例３
    test = Lab(physiology.get_all_person)
    print(test.student())