# -*- coding: utf-8 -*-
# creator : sun yongzhuang 20170919

import lasio
import numpy as np
from syn import *

test=lasio.read('test.las')
lith=list(test['LITH'])#1:coal 2:shale 3:tight sand 4:pore_sand
lith_thick={1:[],2:[],3:[],4:[]}
lith_count={1:[],2:[],3:[],4:[]}
lith_thick[1]=thick_cal(lith,1)
lith_thick[2]=thick_cal(lith,2)
lith_thick[3]=thick_cal(lith,3)
lith_thick[4]=thick_cal(lith,4)
print(max(lith_thick[4]))
lith_count[1]=count_cal(lith_thick[1],[0.5,1,1.5,2])
print('lith1',lith_count[1])
lith_count[2]=count_cal(lith_thick[2],[1,5,10,15,20,25,30,35])
print('lith2',lith_count[2])
lith_count[3]=count_cal(lith_thick[3],[1,5,10,15,20,25,30,35])
print('lith3',lith_count[3])
lith_count[4]=count_cal(lith_thick[4],[1,5,10,15,20,25,30,35])
print('lith4',lith_count[4])