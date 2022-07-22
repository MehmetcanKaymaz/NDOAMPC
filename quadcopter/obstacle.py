import numpy as np


class CircleObstacle:
    def __init__(self,x0=np.zeros(3),vel=np.zeros(3),r=1):
        self.info="Point mass 3D circle dynamic model"
        self.r=r
        self.At=np.array([[0,0,0,1,0,0],
                          [0,0,0,0,1,0],
                          [0,0,0,0,0,1],
                          [0,0,0,0,0,0],
                          [0,0,0,0,0,0],
                          [0,0,0,0,0,0]])
        self.dt=.01

        self.A=self.At*self.dt+np.eye(6)

        self.states=np.concatenate((x0,vel),axis=0).reshape((6,1))

    def update(self):
        self.states=np.matmul(self.A,self.states)

    def get_pose(self):
        return self.states[:3]


