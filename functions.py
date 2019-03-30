import matplotlib.pyplot as plt
import math
import copy
import sklearn.datasets as datasets
import pandas as pd
from pandas.core.frame import DataFrame 
import numpy as np


def calAngle(p1,p2):
    """
    返回给定点与x轴正方向形成的夹角角度以及符号
    param p1,p2: 指定点坐标
    
    """
    dy = p1[1]-p2[1]
    dx = p1[0]-p2[0]
    angle = round(math.atan2(dy,dx),3)
    sign = angle
    if angle == 0 :
        angle= 0

    else:
        if angle < 0:
            angle = 360-(abs(angle)*180/math.pi)
        else:
            angle = angle * 180 / math.pi
    
    return angle,sign


def get_bottom_point(points):
    """
    返回points中纵坐标最小的点的索引，如果有多个纵坐标最小的点则返回其中横坐标最小的那个
    param points: 所有给出坐标点的集合    
    """
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points.iat[i,1] < points.iat[min_index,1] or (points.iat[i,1] == points.iat[min_index,1] and points.iat[i,0] < points.iat[min_index,0]):            #寻找最左边的y值最小的点作为凸包的起始点
            min_index = i
    return min_index
 
 
def sort_polar_angle_cos(points, center_point):
    """
    按照与中心点的极角进行排序，使用的是余弦的方法
    param points: 需要排序的点
    param center_point: 中心点
    """
    n = len(points)
    points_c = copy.deepcopy(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points_c.iloc[[i]].to_numpy()        
        point = [point_[0][0]-center_point[0], point_[0][1]-center_point[1]] 
        rank.append(i)     
        norm_value = (math.sqrt(point[0]*point[0] + point[1]*point[1]))   
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)
 
    for i in range(0, n-1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index-1] or (cos_value[index] == cos_value[index-1] and norm_list[index] > norm_list[index-1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index-1]
                rank[index] = rank[index-1]
                norm_list[index] = norm_list[index-1]
                cos_value[index-1] = temp
                rank[index-1] = temp_rank
                norm_list[index-1] = temp_norm
                index = index-1
            else:
                break
    sorted_points = []
    sorted_index = []
    for i in rank:
        sorted_points.append(points_c.iloc[i].values)
        sorted_index.append(points_c.iloc[[i]].index.values)
    return sorted_points,sorted_index
 
 
def vector_angle(vector):
    """
    返回一个向量与向量 [1, 0]之间的夹角， 这个夹角是指从[1, 0]沿逆时针方向旋转多少度能到达这个向量
    param vector:
    """
    norm_ = math.sqrt(vector[0]*vector[0] + vector[1]*vector[1])
    if norm_ == 0:
        return 0
 
    angle = math.acos(vector[0]/norm_)
    if vector[1] >= 0:
        return angle
    else:
        return 2*math.pi - angle
 
 
def coss_multi(v1, v2):
    """
    计算两个向量的叉乘
    param v1:
    param v2:
    """
    return v1[0]*v2[1] - v1[1]*v2[0]
 
 
def graham_scan(points):
    """
    返回能构成最小凸包(包括线段)的点的集合
    param points
    """
    next_points = copy.deepcopy(points)
    bottom_index = get_bottom_point(next_points)
    bottom_point = next_points.iloc[bottom_index].values
    bottom_label = next_points.iloc[[bottom_index]].index.values
    next_points=next_points.drop(bottom_label,axis=0) 
    sorted_points, sorted_index = sort_polar_angle_cos(next_points, bottom_point)
    
    stack = []
    stack_index = []

    m = len(sorted_points)
    
    if m<2:
        stack.append(bottom_point)
        stack.append(sorted_points[0])
        stack_index.append(bottom_label.tolist()[0])
        stack_index.append(sorted_index[0].tolist()[0])
        return stack,stack_index
 


    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])

    stack_index.append(bottom_label.tolist()[0])
    stack_index.append(sorted_index[0].tolist()[0])
    stack_index.append(sorted_index[1].tolist()[0])
    

    for i in range(2, m):
        length = len(stack)
        top = stack[length-1]#点P1
        next_top = stack[length-2]#点P2
        v1 = [sorted_points[i][0]-next_top[0], sorted_points[i][1]-next_top[1]]
        v2 = [top[0]-next_top[0], top[1]-next_top[1]]
        
        if (sorted_points[i][0]==top[0]).all() and (coss_multi(v1,v2)<0).all():
            stack.append(sorted_points[i])
            stack_index.append(sorted_index[i].tolist()[0])
        else:
            while (round(coss_multi(v1,v2),3)>0).all():            
                stack.pop()
                stack_index.pop()
                length = len(stack)
                top = stack[length-1]
                next_top = stack[length-2]
                v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
                v2 = [top[0] - next_top[0], top[1] - next_top[1]]
           
            stack.append(sorted_points[i])
            stack_index.append(sorted_index[i].tolist()[0])
            

    return stack, stack_index 
 

def Tri(result, outer_convex):
    """
    返回能被连接的外凸包的点的集合
    param result: 构成内凸包的点集合
    param outer_convex: 构成外凸包的点集合
    """
    del_list =[]
    orginal_convex = copy.deepcopy(outer_convex)
    outer_convex_remian=[]
    marked_list=[]

    #保留原始起始点数据
    if outer_convex:
        base_y = outer_convex[0][1]

    if outer_convex is not None:        
        length=len(result)
        #对每一个内维点构建三角形
        for i in range(0,length): 
            compare_list=[] 
            del_list = []              
            if i == 0:
                angle_left,sign_left = calAngle(result[length-1],result[0])
                angle_right,sign_right = calAngle(result[1],result[0])
                compare_list.append(angle_left)
                compare_list.append(angle_right)
                compare_list.sort()   
                    
            else:
                if i != length-1:
                    angle_left,sign_left=calAngle(result[i-1],result[i])
                    angle_right,sign_right=calAngle(result[i+1],result[i])
                    compare_list.append(angle_left)
                    compare_list.append(angle_right)
                    compare_list.sort()
                   
                else:
                    angle_left,sign_left=calAngle(result[i-1],result[i])
                    angle_right,sign_right=calAngle(result[0],result[i])
                    compare_list.append(angle_left)
                    compare_list.append(angle_right)
                    compare_list.sort()
                        
            len_outer = len(outer_convex)
            #对该点求其对外凸包点是否可连                  
            for a in range(0,len_outer):                
                compare_angle, sign = calAngle(outer_convex[a],result[i])               
                #排除离得太近的点                
                if (sign_left<0 and sign_right >0) or (sign_left<0 and sign_right <0) or (sign_left>0 and sign_right>=0) or (sign_left==0 and sign_right>0):
                #在夹角范围内不可连,保留该外围点                        
                    if compare_list[0]<=compare_angle<=compare_list[1]:
                        outer_convex_remian.append(outer_convex[a])
                        
                    #夹角范围外外围点可连，并从外围点中去除
                    else:                    
                        marked_list.append(outer_convex[a])
                        del_list.append(outer_convex[a])                
                    
                elif((sign_left<0 and sign_right==0) or (sign_left>0 and sign_right<0) or (sign_left==0 and sign_right<0)):                       
                    if compare_list[0]<compare_angle<compare_list[1]:
                        marked_list.append(outer_convex[a])
                        del_list.append(outer_convex[a])                            
                        
                    #夹角范围外外围点可连，并从外围点中去除
                    else:                                             
                        outer_convex_remian.append(outer_convex[a])                                   

                else:
                    if result[i][0] > result[0][1]:
                        if 0<compare_angle<math.pi:
                            marked_list.append(outer_convex[a])
                            del_list.append(outer_convex[a])
                 #夹角范围外外围点可连，并从外围点中去除
                        else:                                            
                            outer_convex_remian.append(outer_convex[a])                                    
                    else:
                        if math.pi<compare_angle<2*math.pi:
                            marked_list.append(outer_convex[a])
                            del_list.append(outer_convex[a])
                                                      
                    #夹角范围外外围点可连，并从外围点中去除
                        else:                                                
                            outer_convex_remian.append(outer_convex[a])                                      


            outer_convex = outer_convex_remian
            if del_list==[]:
                pass
            else:
                c=len(del_list)                
                if i!=0:                      
                    outer_convex.insert(0,del_list[c-1])                 
                else:
                    if c>2:
                        max_index = 0
                        for z in range(0,c):                        
                            if del_list[z][0] >=del_list[max_index][0] :                              
                                max_index=z                  
                        outer_convex.insert(0,del_list[max_index])
                        max_index_y=0
                        for p in range(0,c):                                                  
                            if del_list[p][1] >=del_list[max_index_y][1] :                              
                                max_index_y=p                               
                        outer_convex.append(del_list[max_index_y])
                    else:
                        outer_convex = orginal_convex
            outer_convex_remian=[]
            
                # 把连线画出来
            for b in range(0, len(del_list)):    
            
                plt.plot([result[i][0], del_list[b][0]], [result[i][1], del_list[b][1]], c='y')

    return marked_list


if __name__ == "__main__":

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
            
            result, index_del = graham_scan(csv_file)
            #去除已被做凸包边缘的点
            csv_file = csv_file.drop(index_del)
        
            length = len(result)
            for i in range(0, length-1):
                plt.plot([result[i][0], result[i+1][0]], [result[i][1], result[i+1][1]], c='r')
            plt.plot([result[0][0], result[length-1][0]], [result[0][1], result[length-1][1]], c='r')

            #new_list衡量外凸包边缘点被链接次数=构成三角形个数
            new_list=Tri(result,outer_convex)
            sum_of_tri = sum_of_tri + len(new_list)
           
            outer_convex = result
        #核实内部是否有剩余的点
        if len(csv_file)==1:
            sum_of_tri = len(outer_convex) + sum_of_tri
            
            for i in range(0, len(outer_convex)):
                plt.plot([outer_convex[i][0], csv_file.iloc[0][0]], [outer_convex[i][1], csv_file.iloc[0][1]], c='r')
        
        print("There exists {0} triangles ".format(sum_of_tri))    
           
    plt.show()
    