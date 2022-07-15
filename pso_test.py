from custom_pso import pso
import numpy as np
import matplotlib.pyplot as plt
import time

obstacle=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,1,1,2,1,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

s=[]

for i in range(21):
    for j in range(21):
        if obstacle[i][j]!=0:
            s.append([i,j])

def f(x):
    x1=x[0]
    y1=x[1]
    r1=x[2]
    x2=x[3]
    y2=x[4]
    r2=x[5]

    cost=r1**2+r2**2

    for i in range(21):
        for j in range(21):
            if obstacle[i][j]==0:
                d1=(i-x1)**2+(j-y1)**2-r1**2
                d2=(i-x2)**2+(j-y2)**2-r2**2
                
                if d1<0 or d2<0:
                    cost+=1

    return cost

def con(x):
    x11=x[0]
    x12=x[1]
    r1=x[2]

    x21=x[3]
    x22=x[4]
    r2=x[5]

    cons=[]
    
    for si in s:
        con1=-(si[0]-x11)**2-(si[1]-x12)**2+r1**2 
        con2=-(si[0]-x21)**2-(si[1]-x22)**2+r2**2
        con_i=max(con1,con2)
        cons.append(con_i)   
    

    return cons

lb=[0,0,1,0,0,1]
ub=[21,21,21,21,21,21]

ti=time.time()

xopt, fopt=pso(f,lb,ub,f_ieqcons=con,maxiter=10000,minstep=1e-8,min_iter=1,swarmsize=100)

tf=time.time()

print("total time:",tf-ti)

x1=xopt[0]
y1=xopt[1]
r1=xopt[2]

x2=xopt[3]
y2=xopt[4]
r2=xopt[5]

plt.rcParams["figure.figsize"] = (10,10)
ax = plt.gca()
ax.cla() # clear things for fresh plot

for si in s:
    ci=plt.Circle((si[0],si[1]),.1,color='g')
    ax.add_patch(ci)

a1=plt.Circle((x1,y1),.5,color='b')
a2=plt.Circle((x1,y1),r1,color='b',fill=False)
ax.add_patch(a1)
ax.add_patch(a2)

a1=plt.Circle((x2,y2),.5,color='b')
a2=plt.Circle((x2,y2),r2,color='b',fill=False)
ax.add_patch(a1)
ax.add_patch(a2)

ax.set_xlim((-2,23))
ax.set_ylim((-2,23))
ax.grid()
plt.show()