import numpy as np
from pandas import read_excel
import matplotlib.pyplot as plt
from os import path,getcwd


#Read Data
TMPath = path.abspath(getcwd()) + r"\Selling Price.xlsx"
X = read_excel(TMPath,usecols=[0]).squeeze("columns")
Y = read_excel(TMPath,usecols=[1]).squeeze("columns")


#Define Variables
N = len(X) ; Alpha = 0.01
X ,Y = np.array(X) , np.array(Y) #To Get Vectorization Benefits
X = np.column_stack((np.ones(N),X)) #Add Theta 0
Y = Y[:,np.newaxis] #To Turn it To (N,1) Array to Avoid Broadcasting
Theta=np.zeros((1,2))


#Show Data Sample
fig= plt.figure(0,dpi=140,figsize=[6,5])
ax=plt.axes(xlabel="Population",ylabel="Profit",title="Predicted Profit")
fig.add_axes(ax)
ax.scatter(X[:,1],Y,label="Training Set")


#Define Cost Function
def ComputeCost(X,Y,Theta,N):
    Cost = np.power(np.dot(X,Theta.T)-Y,2)
    return np.sum(Cost) / (2*N)
CostLog = [ComputeCost(X,Y,Theta,N)]
print(f"Initial Cost :{CostLog[0]}")


#Gradient Descent Function
def GD_OneIterator(X,Y,Theta,Alpha,N):
    Error = (np.dot(X,Theta.T)-Y)*X
    Error = np.sum(Error,0)
    return Theta - Alpha*Error/N


#Run Gradient Descent N Times
for i in range(1,1000):
    Theta=GD_OneIterator(X,Y,Theta,Alpha,N)
    CostLog.append(ComputeCost(X,Y,Theta,N))
    if CostLog[i-1]-CostLog[i] <=0.001:break 
print(f"New Cost :{CostLog[-1]}")

#Update Figure
x = np.linspace(X[:,1:].min(), X[:,1:].max(),N)
y = np.poly1d(*np.flip(Theta))
ax.plot(x,y(x),"r",label="Best Fit Line")
ax.legend(loc=2)

#Plot Learning Rate 
newfig= plt.figure(1,dpi=140,figsize=[6,5])
newax=plt.axes(xlabel="Iterations",ylabel="Cost Function",title="Learning Rate")
newfig.add_axes(newax)
plt.plot(CostLog, 'r')