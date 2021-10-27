# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:30:45 2021

@author: marcs
"""
import numpy as np
import math
import matplotlib.pyplot as plt


class sudoku:
    def __init__(self, setup):
        self.board=np.array(setup,dtype=np.uint8)
        self.shape=setup.shape
        self.size=np.uint8(np.sqrt(setup.shape[0]))
        self.clusters=[self.board[     self.size*np.uint8(n/self.size)  :  self.size*(1+np.uint8(n/self.size)) ,  self.size*(n%self.size)  :  self.size*(n%self.size+1)] for n in range((self.size)**2)]
        self.columns=[self.board[:,n] for n in range(self.size**2)]   
        self.rows=[self.board[n,:] for n in range(self.size**2)]
        self.total=2**(setup.shape[0])-1
        
       
        
    def CheckElement(self,element):
        var=binarize(element)
        return var==self.total
    
    def CheckRow(self, n):
        return self.CheckElement(self.rows[n])
    
    def CheckColumn(self, n):
        return self.CheckElement(self.columns[n])
    
    def CheckCluster(self, n):
        return self.CheckElement(self.clusters[n])
    
    def CheckSudoku(self):
        B=True
        for n in range(self.size**2):
            B*=self.CheckRow(n)
            B*=self.CheckColumn(n)
            B*=self.CheckCluster(n)
        return B
    
    
        
    
    def coordinates(self,row,col):#returns the number of (row, column, cluster)
        clust=math.floor(col/self.size) + math.floor(row/self.size)*self.size
        return (row,col,clust)
    
    
    def constraints(self, row, col):
        coord=self.coordinates(row,col)
        return [
                binarize(self.rows[coord[0]]),
                binarize(self.columns[coord[1]]),
                binarize(self.clusters[coord[2]])
                ]
    def constraintsB(self, row, col):
        coord=self.coordinates(row,col)
        return [
                self.rows[coord[0]],
                self.columns[coord[1]],
                self.clusters[coord[2]]
                ]
        
    def possibilities(self,row,col):
        if self.board[row,col] != 0:
            return 0
        poss=np.uint32(0)
        for element in self.constraints(row,col):
            poss |= element
        return self.total-poss
    
    def boardPos(self):
        tabla=np.zeros(self.shape, dtype='uint32')
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                tabla[x,y]=self.possibilities(x,y)            #OJO AQUI QUE HI POT HAVER UN CAMBI DE COORDENADES EASY
        return tabla
    
    
    def boardPosVisualize(self):
        tabla=np.zeros(self.shape,dtype='uint64')
        binario=self.boardPos()
        
        
        for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    numbs='0'
                    for i in range(0,self.shape[0]):
                            if (2**i & binario[x,y]) != 0:
                                numbs+=str(i+1)
                    tabla[x,y]=numbs           
        return tabla
    
    def Options(self):
        tabla=np.zeros(self.shape,dtype='uint8')
        binario=self.boardPos()
        
        for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    for i in range(0,self.shape[0]):
                            if (2**i & binario[x,y]) != 0:
                                tabla[x,y]+=1        
        return tabla
    
    
    def Solve(self):
        for n in range (100):
            self.solveStep()
            if(self.CheckSudoku()):
                return n
        return
    
    def solveStep(self):
        tabla=self.solveexclusion()
        for n in range(self.shape[0]):
            tabla[n,:] += self.solverow(n)
            tabla[:,n] += self.solvecol(n)
        
        return tabla
    
    def solveexclusion(self):
        A=np.where(self.Options()==1)
        tabla=np.zeros(self.shape, dtype='uint8')
        for x,y in zip(A[0],A[1]):
            tabla[x,y]= np.log2(self.boardPos()[x,y]) +1#ojo els index 
        self.board += tabla
        return tabla
    

    
    
    def solverow(self,n):
        row=np.zeros(self.shape[0],dtype='uint8')
        for element in self.onlyoptionrow(n):
            for i in range(self.shape[0]):
                if ( (self.possibilities(n,i) & 2**(element-1)) != 0):
                    row[i]=element
        self.rows[n]+=row
        return row
        
    def onlyoptioncol(self, n):
        lista=[]
        for i in range(self.shape[0]):
            counter=0
            for j in range(self.shape[0]): 
                counter += ( (2**i & self.possibilities(j,n) ) != 0)#OJO els index
            if(counter==1):
                lista +=[i+1]
        return lista

    def solvecol(self,n):
        col=np.zeros(self.shape[0],dtype='uint8')
        for element in self.onlyoptioncol(n):
            for i in range(self.shape[0]):
                if ( (self.possibilities(i,n) & 2**(element-1)) != 0):
                    col[i]=element
        self.columns[n]+=col
        return col
        
    def onlyoptionrow(self, n):
        lista=[]
        for i in range(self.shape[0]):
            counter=0
            for j in range(self.shape[0]): 
                counter += ( (2**i & self.possibilities(n,j) ) != 0)#OJO els index
            if(counter==1):
                lista +=[i+1]
        return lista


    
    
    def combinations(self):
        combs=np.uint64(1)
        for row in self.Options():
            for element in row:
                if element!=0:
                    combs*=element
        return combs    

    
def binarize(lista):
    var=np.uint32(0)
    for i in lista.flatten():
        var += (2**(i)-(i==0))
    return np.uint32(var/2)
#%% MAIN
        
easy=[
      #[1,2,3 ,4,0,6 ,7,8,9],
      [5,3,0 ,0,7,0 ,0,0,0],
      [6,0,0 ,1,9,5 ,0,0,0],
      [0,9,8 ,0,0,0 ,0,6,0],
      
      [8,0,0 ,0,6,0 ,0,0,3],
      [4,0,0 ,8,0,3 ,0,0,1],
      [7,0,0 ,0,2,0 ,0,0,6],
      
      [0,6,0 ,0,0,0 ,2,8,0],
      [0,0,0 ,4,1,9 ,0,0,5],
      [0,0,0 ,0,8,0 ,0,7,9]
     ]

hard=[
      [0,0,2 ,0,0,1 ,0,4,9],
      [0,8,0 ,0,0,0 ,5,0,1],
      [3,0,0 ,0,0,9 ,0,0,0],
      
      [0,0,0 ,0,5,0 ,0,0,0],
      [0,0,0 ,3,9,6 ,2,0,0],
      [6,0,1 ,0,0,4 ,0,0,3],
      
      [0,0,3 ,0,0,0 ,0,0,2],
      [9,0,4 ,0,6,0 ,0,0,0],
      [0,6,0 ,0,2,0 ,0,0,0]      
      ]

clust=[
      [1,1,1 ,2,2,2 ,3,3,3],
      [1,1,1 ,2,2,2 ,3,3,3],
      [1,1,1 ,2,2,2 ,3,3,3],
      
      [4,4,4 ,5,5,5 ,6,6,6],
      [4,4,4 ,5,5,5 ,6,6,6],
      [4,4,4 ,5,5,5 ,6,6,6],
      
      [7,7,7 ,8,8,8 ,9,9,9],
      [7,7,7 ,8,8,8 ,9,9,9],
      [7,7,7 ,8,8,8 ,9,9,9]
     ]

big36=[
       [20,13,0,34,7,15,  16,1,0,19,0,24,  4,12,0,5,0,0,  0,28,35,0,14,30,  21,10,8,22,0,0,  32,29,11,9,0,26],
       []       
       
       ]

StartBoard=np.array(hard,dtype=np.uint8)
Sud=sudoku(StartBoard)

#%%
import pandas as pd


def translate(code,board):
    output=board
    output=np.where(output=='x',0,output)
    for i,element in enumerate(code,start=1):
        output=np.where(output==element,i,output)
    return output




df=pd.read_csv('16x16.csv', sep=',',header=None)
code=df.values[0,:]
board=df.values[1:,:]
board=translate(code,board)
Sud2=sudoku(board)
















































