import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

#[6,2],[13,5],[10,11],[16,5]
obstacle=np.array([[0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
                   [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                   [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                   [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                   [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

n=5

s=[]

for i in range(21):
    for j in range(21):
        if obstacle[i][j]!=0:
            s.append([i,j])
    

s=np.array(s)
x_mean=np.mean(s[:,0])
y_mean=np.mean(s[:,1])

#cycles=[[] for _ in range(n)]

o1=[]
o2=[]

for si in s:
    if si[0]>x_mean and si[1]<y_mean:
        o1.append(si)
    elif si[0]<x_mean and si[1]>y_mean:
        o2.append(si)

o1=np.array(o1)
o2=np.array(o2)

o1_x_mean=np.mean(o1[:,0])
o1_y_mean=np.mean(o1[:,1])

o2_x_mean=np.mean(o2[:,0])
o2_y_mean=np.mean(o2[:,1])

r=0

for si in s:
    d=np.sqrt((x_mean-si[0])**2+(y_mean-si[1])**2)
    if d>r:
        r=d 

opti=ca.Opti()

X=opti.variable(3,n)

cost=np.pi*(X[2,0]**2+X[2,1]**2+X[2,2]**2+X[2,3]**2+X[2,4]**2)#+1*((X[2,0]-X[2,1])**2+(X[2,1]-X[2,2])**2+(X[2,0]-X[2,2])**2)

for i in range(21):
    for j in range(21):
        si=[i,j]
        if obstacle[i][j]!=0:
            opti.subject_to(ca.fmax((-(si[0]-X[0,0])**2-(si[1]-X[1,0])**2+X[2,0]**2),
                            ca.fmax((-(si[0]-X[0,1])**2-(si[1]-X[1,1])**2+X[2,1]**2),
                            ca.fmax((-(si[0]-X[0,2])**2-(si[1]-X[1,2])**2+X[2,2]**2),
                            ca.fmax((-(si[0]-X[0,3])**2-(si[1]-X[1,3])**2+X[2,3]**2),
                                    (-(si[0]-X[0,4])**2-(si[1]-X[1,4])**2+X[2,4]**2)))))>=0)


opti.subject_to(X[2,0]>=0)
opti.subject_to(X[2,1]>=0)
opti.subject_to(X[2,2]>=0)
opti.subject_to(X[2,3]>=0)
opti.subject_to(X[2,4]>=0)

opti.minimize(cost)

ipopt_options = {
    'verbose': False,
    "ipopt.tol": 1e-4,
    "ipopt.acceptable_tol": 1e-4,
    "ipopt.max_iter": 1000,
    "ipopt.warm_start_init_point": "yes",
    "ipopt.print_level": 0,
    "print_time": True
    }

opti.solver('ipopt', ipopt_options)

"""o1_x_mean=x_mean
o2_x_mean=x_mean
o1_y_mean=y_mean
o2_y_mean=y_mean"""
#[6,2],[13,5],[10,11],[16,5]
initial_guess=np.array([[2,2,5,11,16],[6,10,13,10,5],[5,5,5,5,5]])

opti.set_initial(X,initial_guess)
sol=opti.solve()
res=sol.value(X)

print("Max R:",r)
print("Optimization Result: {}".format(res))

plt.rcParams["figure.figsize"] = (10,10)
ratio=28500
ax = plt.gca()
ax.cla() # clear things for fresh plot

for si in s:
    ci=plt.Circle((si[0],si[1]),.5,color='g')
    ax.add_patch(ci)

a1=plt.Circle((x_mean,y_mean),.5,color='r')
a2=plt.Circle((x_mean,y_mean),r,color='r',fill=False)
ax.add_patch(a1)
ax.add_patch(a2)

a1=plt.Circle((res[0,0],res[1,0]),.5,color='b')
a2=plt.Circle((res[0,0],res[1,0]),res[2,0],color='b',fill=False)
ax.add_patch(a1)
ax.add_patch(a2)

a1=plt.Circle((res[0,1],res[1,1]),.5,color='c')
a2=plt.Circle((res[0,1],res[1,1]),res[2,1],color='c',fill=False)
ax.add_patch(a1)
ax.add_patch(a2)

a1=plt.Circle((res[0,2],res[1,2]),.5,color='yellow')
a2=plt.Circle((res[0,2],res[1,2]),res[2,2],color='yellow',fill=False)
ax.add_patch(a1)
ax.add_patch(a2)

a1=plt.Circle((res[0,3],res[1,3]),.5,color='orange')
a2=plt.Circle((res[0,3],res[1,3]),res[2,3],color='orange',fill=False)
ax.add_patch(a1)
ax.add_patch(a2)

a1=plt.Circle((res[0,4],res[1,4]),.5,color='black')
a2=plt.Circle((res[0,4],res[1,4]),res[2,4],color='black',fill=False)
ax.add_patch(a1)
ax.add_patch(a2)

#[6,2],[13,5],[10,11],[16,5]
a1=plt.Circle((2,6),.1,color='blue')
a2=plt.Circle((5,13),.1,color='yellow')
a3=plt.Circle((11,10),.1,color='orange')
a4=plt.Circle((16,5),.1,color='black')
a5=plt.Circle((2,10),.1,color='c')
ax.add_patch(a1)
ax.add_patch(a2)
ax.add_patch(a3)
ax.add_patch(a4)
ax.add_patch(a5)

ax.set_xlim((-2,23))
ax.set_ylim((-2,23))
ax.grid()
plt.show()