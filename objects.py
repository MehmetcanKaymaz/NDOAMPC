import numpy as np

class DynamicObject:
    def __init__(self,velocity=[0,0],pose=[0,0],r=1,dt=0.01,acc_type=None,noise_var=1):
        self.info="Simple obstacle model"

        self.noise_var=noise_var
        self.acc_type=acc_type

        self.r=r

        self.states=np.array([pose[0],pose[1]])
        self.states_dot=np.array([velocity[0],velocity[1]])
        self.dt=dt

        self.pose_list=[]
        self.pose_list.append(self.position().copy())
            
    def update(self):
        self.__update_state_dot()
        self.states+=self.states_dot*self.dt
        self.pose_list.append(self.position().copy())

    def __update_state_dot(self):

        if self.acc_type=='gauss':
            noise=self.__get_rondom_noise()
            self.state_dot+=noise
        
    def __get_rondom_noise(self):
        return np.random.normal(0,self.noise_var,2)

    def position(self):
        return self.states[0:2]

class SimpleCarObject:
    def __init__(self,dt=0.01):
        self.info="Simple Car model"

        self.states=np.zeros(2)
        self.states_dot=np.zeros(2)
        self.dt=dt
        self.u=np.zeros(2)

        self.pose_list=[]
        self.pose_list.append(self.position().copy())
            
    def update(self,u):
        self.u=u
        self.__update_state_dot()
        self.states+=self.states_dot*self.dt
        self.pose_list.append(self.position().copy())

    def __update_state_dot(self):
        state_dot=np.array([self.u[0],self.u[1]])
        self.states_dot=state_dot

    def position(self):
        return self.states