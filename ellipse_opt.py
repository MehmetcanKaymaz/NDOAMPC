import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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
                   [0,0,0,0,0,0,0,1,1,1,2,1,1,1,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

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


#Find center and radius
s=[]

for i in range(21):
    for j in range(21):
        if obstacle[i][j]!=0:
            s.append([i,j])
    

s=np.array(s)
x_mean=np.mean(s[:,0])
y_mean=np.mean(s[:,1])


r=0

for si in s:
    d=np.sqrt((x_mean-si[0])**2+(y_mean-si[1])**2)
    if d>r:
        r=d 


#Optimization
opti=ca.Opti()

p=opti.variable(2,1)
a=opti.variable(2,1)
psi=opti.variable(1,1)

p2=opti.variable(2,1)
a2=opti.variable(2,1)
psi2=opti.variable(1,1)

p3=opti.variable(2,1)
a3=opti.variable(2,1)
psi3=opti.variable(1,1)

cost=np.pi*(a[0]*a[1]+a2[0]*a2[1]+a3[0]*a3[1])

#n=1
"""for si in s:
    sc=np.array(si).reshape((2,1))
    opti.subject_to((((sc[0]-p[0])*ca.cos(psi)-(sc[1]-p[1])*ca.sin(psi))**2/a[0]**2)+
                    (((sc[0]-p[0])*ca.sin(psi)+(sc[1]-p[1])*ca.cos(psi))**2/a[1]**2)-1<=0)"""
#n=2
"""for si in s:
    sc=np.array(si).reshape((2,1))
    opti.subject_to(ca.fmin((((sc[0]-p[0])*ca.cos(psi)-(sc[1]-p[1])*ca.sin(psi))**2/a[0]**2)+
                    (((sc[0]-p[0])*ca.sin(psi)+(sc[1]-p[1])*ca.cos(psi))**2/a[1]**2)-1,
                    (((sc[0]-p2[0])*ca.cos(psi2)-(sc[1]-p2[1])*ca.sin(psi2))**2/a2[0]**2)+
                    (((sc[0]-p2[0])*ca.sin(psi2)+(sc[1]-p2[1])*ca.cos(psi2))**2/a2[1]**2)-1)<=0)"""

#n=3
for si in s:
    sc=np.array(si).reshape((2,1))
    opti.subject_to(ca.fmin((((sc[0]-p[0])*ca.cos(psi)-(sc[1]-p[1])*ca.sin(psi))**2/a[0]**2)+
                    (((sc[0]-p[0])*ca.sin(psi)+(sc[1]-p[1])*ca.cos(psi))**2/a[1]**2)-1,
                    ca.fmin((((sc[0]-p2[0])*ca.cos(psi2)-(sc[1]-p2[1])*ca.sin(psi2))**2/a2[0]**2)+
                    (((sc[0]-p2[0])*ca.sin(psi2)+(sc[1]-p2[1])*ca.cos(psi2))**2/a2[1]**2)-1,
                    (((sc[0]-p3[0])*ca.cos(psi3)-(sc[1]-p3[1])*ca.sin(psi3))**2/a3[0]**2)+
                    (((sc[0]-p3[0])*ca.sin(psi3)+(sc[1]-p3[1])*ca.cos(psi3))**2/a3[1]**2)-1))<=0)


opti.subject_to(a[0]>0)
opti.subject_to(a[1]>0)

opti.subject_to(a2[0]>0)
opti.subject_to(a2[1]>0)

opti.subject_to(a3[0]>0)
opti.subject_to(a3[1]>0)

opti.minimize(cost)


ipopt_options = {
    'verbose': False,
    "ipopt.tol": 1e-4,
    "ipopt.acceptable_tol": 1e-4,
    "ipopt.max_iter": 10000,
    "ipopt.warm_start_init_point": "yes",
    "ipopt.print_level": 0,
    "print_time": True
    }

opti.solver('ipopt', ipopt_options)

opti.set_initial(p,[x_mean,y_mean])
opti.set_initial(a,[r,r])
opti.set_initial(psi,0)

opti.set_initial(p2,[x_mean,y_mean])
opti.set_initial(a2,[r,r])
opti.set_initial(psi2,0)

opti.set_initial(p3,[x_mean,y_mean])
opti.set_initial(a3,[r,r])
opti.set_initial(psi3,0)

sol=opti.solve()

res_p=opti.value(p)
res_a=opti.value(a)
res_psi=opti.value(psi)

res_p2=opti.value(p2)
res_a2=opti.value(a2)
res_psi2=opti.value(psi2)

res_p3=opti.value(p3)
res_a3=opti.value(a3)
res_psi3=opti.value(psi3)

print("guess 1 pose:{} rads:{}  psi:{}".format(res_p,res_a,res_psi))
print("guess 2 pose:{} rads:{}  psi:{}".format(res_p2,res_a2,res_psi2))
print("guess 3 pose:{} rads:{}  psi:{}".format(res_p3,res_a3,res_psi3))


plt.rcParams["figure.figsize"] = (10,10)
ax = plt.gca()
ax.cla() # clear things for fresh plot

#Draw points
for si in s:
    ci=plt.Circle((si[0],si[1]),.1,color='g')
    ax.add_patch(ci)

#Center and min radius
a1=plt.Circle((x_mean,y_mean),.5,color='r')
a2=plt.Circle((x_mean,y_mean),r,color='r',fill=False)
ax.add_patch(a1)
ax.add_patch(a2)

#Center of ellipies
a1=plt.Circle((res_p[0],res_p[1]),.5,color='b')
ax.add_patch(a1)

a1=plt.Circle((res_p2[0],res_p2[1]),.5,color='y')
ax.add_patch(a1)

a1=plt.Circle((res_p3[0],res_p3[1]),.5,color='black')
ax.add_patch(a1)

#Ellipies
a1=Ellipse(res_p, width=2*res_a[0], height=2*res_a[1], angle=(np.pi-res_psi)*180/np.pi,color='b',fill=False)
ax.add_artist(a1)

a1=Ellipse(res_p2, width=2*res_a2[0], height=2*res_a2[1], angle=(np.pi-res_psi2)*180/np.pi,color='y',fill=False)
ax.add_artist(a1)

a1=Ellipse(res_p3, width=2*res_a3[0], height=2*res_a3[1], angle=(np.pi-res_psi3)*180/np.pi,color='black',fill=False)
ax.add_artist(a1)

ax.set_xlim((-2,23))
ax.set_ylim((-2,23))
ax.grid()
plt.show()