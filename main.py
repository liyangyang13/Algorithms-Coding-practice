import matplotlib.pyplot as plt
import math
import copy
import sklearn.datasets as datasets
import pandas as pd
from pandas.core.frame import DataFrame 
import numpy as np

import functions as fc

#请使用csv文件进行输入，第一行请保持0，1，以帮助后续画图
csv_file = pd.read_csv('C:\\Users\\Administrator\\Downloads\\liyangyang1.csv')
#重复点会干扰凸包算法
csv_file = csv_file.drop_duplicates()
csv_file.plot.scatter(x=0,y=1)
#判断是否需要用到凸包算法
if len(csv_file) < 3:
    print("There exists {0} triangles. Need more points".format(0))
    
elif len(csv_file) ==3:
    print("There exists {0} triangles".format(1))
else:
        #使用凸包算法构建三角形
    outer_convex = None
    sum_of_tri = 0
    while len(csv_file) >=2:
        #凸包
        print(len(csv_file))
        result, index_del = fc.graham_scan(csv_file)
        #去除已被做凸包边缘的点
        csv_file = csv_file.drop(index_del)
        
        length = len(result)
        for i in range(0, length-1):
            plt.plot([result[i][0], result[i+1][0]], [result[i][1], result[i+1][1]], c='r')
        plt.plot([result[0][0], result[length-1][0]], [result[0][1], result[length-1][1]], c='r')

        #new_list衡量外凸包边缘点被链接次数=构成三角形个数
        new_list=fc. Tri(result,outer_convex)
        sum_of_tri = sum_of_tri + len(new_list)
           
        outer_convex = result
        #核实内部是否有剩余的点
    if len(csv_file)==1:
        sum_of_tri = len(outer_convex) + sum_of_tri
            
        for i in range(0, len(outer_convex)):
            plt.plot([outer_convex[i][0], csv_file.iloc[0][0]], [outer_convex[i][1], csv_file.iloc[0][1]], c='r')
        
    print("There exists {0} triangles ".format(sum_of_tri))    
           
plt.show()


