import numpy as np
from sim import Sim
from controller import NMPC

sim=Sim()
obstacle_states,obstacle_r=sim.get_obstacle_states()

quadcopter_r=sim.quadcopter_r

n_obstacle=len(obstacle_states)

controller=NMPC(n_obstacle=n_obstacle,obstacle_r=obstacle_r,quadcopter_r=quadcopter_r)

ref=np.array([5,0]).reshape((2,1))

dt=.01
T=1.5
N=int(T/dt)

for i in range(N):
    x=sim.simple_car.states
    ox,o_r=sim.get_obstacle_states()
    if len(ox)!=n_obstacle:
        n_obstacle=len(ox)
        #print("number of cureent obstacle:",n_obstacle)
        controller=NMPC(n_obstacle=n_obstacle,obstacle_r=o_r,quadcopter_r=quadcopter_r)
    u=controller.run_controller(x0=x,ref_states=ref,obstacle_states=ox)
    sim.update(u=u)


sim.vis(ref=ref,save=True)

