import numpy as np
from quadcopter import Quadcopter
from controller import NMPC
import matplotlib.pyplot as plt
from planner import SimplePlanner,CAPlanner


"""quad=Quadcopter()
controller=NMPC()
planner=SimplePlanner()

ref=np.ones(3)

state_arr=[]

T=1
dt=quad.dt
N=int(T/dt)

for i in range(N):
    state=quad.states
    traj=planner.plan(x0=state[:3],ref=ref)
    u=controller.run(x0=state,ref=traj)
    quad.update(u=u)
    state_arr.append(state.copy())


state_arr=np.array(state_arr)
t=np.linspace(0,T,N)
plt.plot(t,state_arr[:,0],label='x')
plt.plot(t,state_arr[:,1],label='y')
plt.plot(t,state_arr[:,2],label='z')

plt.legend()
plt.show()"""


planner=CAPlanner(n_obstacle=2)
x0=np.zeros(6)
T=np.ones(1)
obs_states=[[5,5,5,0,0,0],[5,5,5,0,0,0]]

print(planner.plan(x0=x0,ref=T,obstacle_states=obs_states))