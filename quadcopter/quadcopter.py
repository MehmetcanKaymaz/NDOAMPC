import numpy as np
from quat import quat_mult,rotate_quad
from model_params import get_model_params

class Quadcopter:
    def __init__(self):
        self.info="3D quadcopter nonlinear dynamic model"
        
        self.states=np.array([.0,.0,.0,.0,.0,.0,1.,.0,.0,.0,.0,.0,.0])
        
        m,l,g,I,I_inv,T_max,T_min,omega_max,ctau,cd=get_model_params()

        self.m=m
        self.l=l
        self.g=g
        self.I=I
        self.I_inv=I_inv
        self.T_max=T_max
        self.T_min=T_min
        self.omega_max=omega_max
        self.ctau=ctau
        self.cd=cd

        self.dt=.01

    def update(self,u):
        state_dot=self.__update_state_dot(state=self.states,u=u)
        self.states+=state_dot*self.dt
    
    def __update_state_dot(self,state,u):
        v=state[3:6].reshape((3,1))
        q=state[6:10].reshape((4,1))
        w=state[-3:].reshape((3,1))

        p_dot=v
        
        v_dot=rotate_quad(q,np.array([0,0,np.sum(u)/self.m]).reshape((3,1)))+np.array([0,0,-self.g]).reshape((3,1))-v*self.cd
        
        q_dot=.5*quat_mult(q,np.array([0,w[0],w[1],w[2]],dtype='float32').reshape((4,1)))

        w_dot=np.matmul(self.I_inv,np.array([self.l*(u[0]-u[1]-u[2]+u[3]),self.l*(-u[0]-u[1]+u[2]+u[3]),self.ctau*(u[0]-u[1]+u[2]-u[3])]).reshape((3,1)))-np.cross(w.reshape((1,3)),np.matmul(self.I,w).reshape(1,3)).reshape((3,1))

        return np.concatenate((p_dot,v_dot,q_dot,w_dot), axis=0).reshape((13,))


        