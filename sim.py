import numpy as np
from objects import DynamicObject,SimpleCarObject
import matplotlib.pyplot as plt
import os
import cv2

class Sim:
    def __init__(self):
        self.simple_car=SimpleCarObject()

        self.obstacle_poses=[[5.,0.2],[4.5,.5],[4.,.6],[5.,-1.3],[4.5,-1.],[4.,-.7],[5.5,-1.],[5.5,-.5]]
        self.obstacle_vels=[[-2.,0.],[-2.,0.],[-2.,0.],[-2.0,0.],[-2.0,.0],[-2.0,.0],[-2.,0],[-2.,-0.5]]
        self.obstacle_r=[.3,.3,.3,.3,.3,.3,.3,.3]
        self.quadcopter_r=.2
        self.quadcopter_obs_r=1.

        self.obstacles=[]
        self.n_obstacle=len(self.obstacle_poses)

        for i in range(self.n_obstacle):
            self.obstacles.append(DynamicObject(pose=self.obstacle_poses[i],velocity=self.obstacle_vels[i],r=self.obstacle_r[i]))

    def update(self,u):
        self.simple_car.update(u=u)
        for obstacle in self.obstacles:
            obstacle.update()

    def get_obstacle_states(self):
        states=[]
        r=[]
        for obstacle in self.obstacles:
            distance=self.get_distance(obstacle=obstacle)
            if distance<self.quadcopter_obs_r:
                states.append(obstacle.states)
                r.append(obstacle.r)        
        return states,r
    
    def get_distance(self, obstacle):
        obstacle_position=obstacle.position()
        car_position=self.simple_car.position()
        distance=np.sqrt((car_position[0]-obstacle_position[0])**2+(car_position[1]-obstacle_position[1])**2)
        return distance

    def vis(self,ref,save=False):
        N=len(self.simple_car.pose_list)
        self.simple_car_locs=np.array(self.simple_car.pose_list)
        plt.rcParams["figure.figsize"] = (10,10)
        ax = plt.gca()
        ax.cla() 

        if save:
            os.system("mkdir videos/images")
        for i in range(N):
            for obstacle in self.obstacles:
                obstacle_locs=np.array(obstacle.pose_list)

                a2=plt.Circle((obstacle_locs[i,0],obstacle_locs[i,1]),obstacle.r,color='blue')

                ax.add_patch(a2)
            
            a2=plt.Circle((self.simple_car_locs[i,0],self.simple_car_locs[i,1]),self.quadcopter_r,color='darkorange')
            
            a3=plt.Circle((ref[0],ref[1]),.1,color='red')
            
            a4=plt.Circle((self.simple_car_locs[i,0],self.simple_car_locs[i,1]),1,color='red',fill=False)
            
            ax.add_patch(a2)
            ax.add_patch(a3)
            ax.add_patch(a4)

            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim([0,6])
            plt.ylim([-3,3])
            plt.grid()
            
            if save:
                plt.savefig("videos/images/image_{}.png".format(i))

            plt.pause(.000001)
            plt.cla()

        if save:
            video_name="videos/test6.avi"
            frame=cv2.imread("videos/images/image_0.png")
            h,w,l=frame.shape

            video=cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 15,(w,h))

            for i in range(N):
                video.write(cv2.imread("videos/images/image_{}.png".format(i)))

            video.release()
            os.system("rm -r videos/images")
