#%% 

from gurobipy import *
from gurobipy import quicksum
from gurobipy import GRB
import pandas as pd
import numpy as np
import datetime
import random
import plotly
import plotly.figure_factory
import matplotlib.pyplot as plt
import ast

model = Model("Final_Project")
file = 'Job_Machine_Information_Big.xlsx'

#%% Parameters (Coefficients)

# Processing Time (job i on machine j)
t = pd.read_excel( file, sheet_name ='Processing time' , index_col = [0])
job= list(t.columns.values)#工件名稱 
machine = list(t.index) #機台名稱
t = t.T.to_numpy()

# Electricity Comsumption Rate (job i on machine j)
p  = pd.read_excel(file , sheet_name ='Comsumption rate' , index_col = [0])
p = p.T.to_numpy()

# Dynamic Electricity Price
c_interval = pd.read_excel(file, sheet_name ='Electricity price' , index_col = [0])
Interval= list(c_interval.columns.values) #區間名稱
c_interval = c_interval.T.to_numpy() 

# 工件可跨越的最大機台數目（隨機產生）
Pai = pd.read_excel(file, sheet_name ='Pai' , index_col = [0])
Pai = Pai.T.to_numpy()

# 每個durations對應的價格
c = pd.read_excel(file, sheet_name = 'Period price' , index_col = [0])
c = c.T.to_numpy()

# Duration of period k
T = pd.read_excel(file, sheet_name ='Durations time' , index_col = [0]) 
DurationsName= list(T.columns.values) #區間名稱
T = T.T.to_numpy()

makesplan = T.sum()

M= len(machine)
N= len(job)
K= len(DurationsName)
Num_Interval= len(Interval) 

#%% Decision Variables

X={}
for i in range(N) :
    for j in range(M) :
        for k in range(K):
            X[i,j,k] = model.addVar(vtype= GRB.CONTINUOUS, name= "X_%s_%s_%s"%(i,j,k))
           
Y={}
for i in range(N) :
    for j in range(M) :
        for k in range(K):
            Y[i,j,k] = model.addVar(vtype= GRB.BINARY, name= "Y_%s_%s_%s"%(i,j,k))
             
U={}
for i in range(N) :
    for j in range(M) :
        for k in range(K):
            U[i,j,k] = model.addVar(vtype= GRB.BINARY, name= "U_%s_%s_%s"%(i,j,k))
            
V={}
for i in range(N) :
    for j in range(M) :
        V[i,j] = model.addVar(vtype= GRB.BINARY, name= "V_%s_%s"%(i,j))
            
model.update()


#%% Constraints

#Constraint(1)
for i in range(N) :
    model.addConstr(quicksum((X[i,j,k]/t[i,j]) for k in range(K) for j in range(M)) == 1 , name= "Con0")
#Constraint(2)
for i in range(N) :
    for j in range(M) :
        for k in range(K):
            model.addConstr( X[i,j,k] <= t[i,j]*Y[i,j,k] , name= "Con1")         
#Constraint(3) 
for j in range(M) :
    for k in range(K):
          model.addConstr(quicksum(X[i,j,k] for i in range(N)) <= T[k] , name= "Con2")

#Constraint(4)
for i in range(N) :
    for j in range(M) :
        for k in range(K-1):
            model.addConstr( U[i,j,k] >= Y[i,j,k] + Y[i,j,k+1] -1 , name= "Con3")

#Constraint(5)
for j in range(M) :
    for k in range(K):
          model.addConstr(quicksum(U[i,j,k] for i in range(N)) <= 1 , name= "Con4")

#Constraint(6)
for i in range(N) :
    for j in range(M) :
        for k in range(K-2):
            model.addConstr(quicksum((Y[i,j,l] for l in range(k+1,K))) <= K * (1-Y[i,j,k]+Y[i,j,k+1]) , name= "Con5")
         
#Constraint(7) 
for i in range(N) :
    for j in range(M) :
        for k in range(1,K-1):
            model.addConstr(X[i,j,k] >= (Y[i,j,k-1]+Y[i,j,k+1]-1)*T[k] , name= "Con6")
         
#Constraint(8)
for i in range(N) :
    for j in range(M) :
        model.addConstr(V[i,j] >=(quicksum((Y[i,j,k] for k in range(K)))/Pai[i,j]), name= "Con7")  
        
#Constraint(9)
for i in range(N):
    model.addConstr(quicksum(V[i, j] for j in range(M)) == 1 , name= "Con8")
    
#Constraint(13)
for i in range(N) :
    for j in range(M) :
        for k in range(K):
            model.addConstr( X[i,j,k] >= 0 , name= "Con9")
          
#Constraint(14)
for i in range(N) :
    for j in range(M) :
          model.addConstr(quicksum(X[i,j,k] for k in range(K)) <= V[i,j]*t[i,j], name= "Con10")
          
#Constraint(15)
for i in range(N) :
    for j in range(M) :
          model.addConstr(quicksum(Y[i,j,k] for k in range(K)) <= V[i,j]*Pai[i,j], name= "Con11")
     

# Objective - Total Electricity Cost
TEC = quicksum( p[i,j] * X[i,j,k] * c[k] for i in range(N) for j in range(M) for k in range(K))              
model.setObjective(TEC, GRB.MINIMIZE)


#%% Optimization

model.write('Final_Project.lp')
# model.Params.timelimit = 1000
model.optimize()    

#%% Results

X_sol = model.getAttr('x', X)
Y_sol = model.getAttr('x', Y)
U_sol = model.getAttr('x', U)
V_sol = model.getAttr('x', V)

ass = {}
for i in range(N):
    ass[i] = {}
keys = V_sol.keys()
for key in keys:
    ass[key[0]][key[1]] = V_sol[key]    
ass = pd.DataFrame(ass)

ass_per = {}
for i in range(N):
    ass_per[i] = {}
keys = Y_sol.keys()
for key in keys:
    if Y_sol[key] == 1:
        if key[1] not in ass_per[key[0]].keys():
            ass_per[key[0]][key[1]] = []
        ass_per[key[0]][key[1]].append(key[2])
    
            
ass_per = pd.DataFrame(ass_per)   
ass_per = ass_per.sort_index()     

#%%
orders = {}
orders_sort = [[0], [0,1], [1], [1,2], [2], [2,3], [3]]
for order in orders_sort:
    orders[str(order)] = []

for i in range(N):
    for j in range(M):
        if ass_per[i][j] is not np.nan:
            val = str(ass_per[i][j])
            orders[val].append(i)

for order in orders_sort:
    orders[str(order)].sort()
    
orders_sort_all = []
for order in orders_sort:
    for col in orders[str(order)]:
        orders_sort_all.append(col)
        
ass_per_ord = ass_per[orders_sort_all]

#%%

orders = {}
orders_sort = [[0], [0,1], [1], [1,2], [2], [2,3], [3]]
for order in orders_sort:
    orders[str(order)] = {}
    for j in range(M):
        orders[str(order)][j] = []

for i in range(N):
    for j in range(M):
        if ass_per[i][j] is not np.nan:
            val = str(ass_per[i][j])
            orders[val][j].append(i)
            
ass_per_ord2 = pd.DataFrame(orders)

#%%

orders = {}
orders_sort = [[0], [0,1], [1], [1,2], [2], [2,3], [3]]
for order in orders_sort:
    orders[str(order)] = {}
    for j in range(M):
        orders[str(order)][j] = {}


for i in range(N):
    for j in range(M):
        if ass_per[i][j] is not np.nan:
            ks = ass_per[i][j]
            val = str(ks)
            xs = []
            for k in ks:
                xs.append(round(X_sol[(i, j, k)], 2))
            orders[val][j][i] = xs
            
ass_per_ord3 = pd.DataFrame(orders)
        
        
#%% Check N Jobs Assigned
temp = 0
for j in ass_per_ord3.columns:
    for i in range(ass_per_ord3.shape[0]):
        temp += len(ass_per_ord3[j][i])
        
print('Jobs assigned : ', temp)

#%% Gantt Chart

j = 0
s = {'[0]' : 0 , '[0, 1]' : 6, '[1]' : 6, '[1, 2]' : 12 , '[2]' : 12, '[2, 3]' : 22 , '[3]' : 22 }

plt.figure(dpi=1500)

for j in range(M):
    row = ass_per_ord3.loc[j]
    cross = 0
    for k_ in row.keys():
        if len(row[k_]) == 0 :
            cross = 0
        else:
            sk = s[k_]
            if len(ast.literal_eval(k_)) == 1:
                sk += cross
            elif len(ast.literal_eval(k_)) == 2:
                if len(row[k_]) != 1:
                    print('error!')
                else:
                    sk -= list(row[k_].values())[0][0]
                    cross = list(row[k_].values())[0][1]
            x = sk
            for ass in row[k_].keys():
                i = ass
                if len(row[k_][i]) == 1:
                    time = row[k_][i][0]
                    
                elif len(row[k_][i]) == 2:
                    time = sum(row[k_][i])
                else:
                    print('error') 
                y = j +1
                plt.barh( y = y, width = time, left = x)
                plt.text( x = x , y = y , s= f'{i+1}', size = 'xx-small')
                x += time
                
plt.title('Optimized Solution Gantt Chart')

plt.xticks(np.arange(0, 25, 1), size = 'small') 
plt.xlabel('Time')

# plt.yticks(job) 
plt.yticks(np.arange(1, 18, 1)) 
plt.ylabel('Machine')

plt.savefig('甘特圖.png')  
plt.show()

