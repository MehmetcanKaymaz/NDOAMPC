import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class EllipseGenerator:
    def __init__(self, n=1, obstacle=None, soft_constraints=False, only_barriers=False, equal_area_cost=False):
        self.info = "2D ellipse generator"
        self.n = n
        self.obstacle = obstacle
        self.soft_constraints=soft_constraints
        self.soft_theta=.1

        self.s,self.circle=self.get_inner_points(self.obstacle)
        if only_barriers:
          self.s=self.get_barrier_points(self.obstacle)

        self.pose_results=None
        self.rads_results=None
        self.angle_results=None

        #Non-Convex Optimization 
        self.opti = ca.Opti()

        self.ellipse_poses = self.opti.variable(2,self.n)
        self.ellipse_rads = self.opti.variable(2,self.n)
        self.ellipse_angles = self.opti.variable(self.n)

        if self.soft_constraints:
          self.S=self.opti.variable(self.s.shape[0])


        self.cost = 0
        
        #Area cost
        for i in range(self.n):
            if self.n==1:
                self.cost += np.pi*self.ellipse_rads[0]*self.ellipse_rads[1]
            else:
                self.cost += np.pi*self.ellipse_rads[0,i]*self.ellipse_rads[1,i]
        
        #Soft cost
        if self.soft_constraints:
          for k in range(self.s.shape[0]):
            self.cost+=self.S[k]**2
        
        #Equal area cost
        if equal_area_cost:
          for i in range(self.n):
              for j in range(self.n):
                  if i!=j:
                      self.cost+=.2*np.pi*(self.ellipse_rads[0,i]*self.ellipse_rads[1,i]-self.ellipse_rads[0,j]*self.ellipse_rads[1,j])**2

        if self.n==1:
            index=0
            for si in self.s:
                sc=np.array(si).reshape((2,1))
                if self.soft_constraints:
                  self.opti.subject_to((((sc[0]-self.ellipse_poses[0])*ca.cos(self.ellipse_angles)-
                                        (sc[1]-self.ellipse_poses[1])*ca.sin(self.ellipse_angles))**2/self.ellipse_rads[0]**2)+
                                      (((sc[0]-self.ellipse_poses[0])*ca.sin(self.ellipse_angles)+
                                        (sc[1]-self.ellipse_poses[1])*ca.cos(self.ellipse_angles))**2/self.ellipse_rads[1]**2)-1-self.soft_theta*self.S[index]<=0)   
                else:  
                  self.opti.subject_to((((sc[0]-self.ellipse_poses[0])*ca.cos(self.ellipse_angles)-
                                        (sc[1]-self.ellipse_poses[1])*ca.sin(self.ellipse_angles))**2/self.ellipse_rads[0]**2)+
                                      (((sc[0]-self.ellipse_poses[0])*ca.sin(self.ellipse_angles)+
                                        (sc[1]-self.ellipse_poses[1])*ca.cos(self.ellipse_angles))**2/self.ellipse_rads[1]**2)-1<=0)            
                index+=1

        elif self.n==2:
            index=0
            for si in self.s:
                sc=np.array(si).reshape((2,1))
                if self.soft_constraints:
                  self.opti.subject_to(ca.fmin((((sc[0]-self.ellipse_poses[0,0])*ca.cos(self.ellipse_angles[0])-
                                                (sc[1]-self.ellipse_poses[1,0])*ca.sin(self.ellipse_angles[0]))**2/self.ellipse_rads[0,0]**2)+
                                              (((sc[0]-self.ellipse_poses[0,0])*ca.sin(self.ellipse_angles[0])+
                                                (sc[1]-self.ellipse_poses[1,0])*ca.cos(self.ellipse_angles[0]))**2/self.ellipse_rads[1,0]**2)-1,
                                              (((sc[0]-self.ellipse_poses[0,1])*ca.cos(self.ellipse_angles[1])-
                                                (sc[1]-self.ellipse_poses[1,1])*ca.sin(self.ellipse_angles[1]))**2/self.ellipse_rads[0,1]**2)+
                                              (((sc[0]-self.ellipse_poses[0,1])*ca.sin(self.ellipse_angles[1])+
                                                (sc[1]-self.ellipse_poses[1,1])*ca.cos(self.ellipse_angles[1]))**2/self.ellipse_rads[1,1]**2)-1)-self.soft_theta*self.S[index]<=0)  
                else:
                  self.opti.subject_to(ca.fmin((((sc[0]-self.ellipse_poses[0,0])*ca.cos(self.ellipse_angles[0])-
                                                (sc[1]-self.ellipse_poses[1,0])*ca.sin(self.ellipse_angles[0]))**2/self.ellipse_rads[0,0]**2)+
                                              (((sc[0]-self.ellipse_poses[0,0])*ca.sin(self.ellipse_angles[0])+
                                                (sc[1]-self.ellipse_poses[1,0])*ca.cos(self.ellipse_angles[0]))**2/self.ellipse_rads[1,0]**2)-1,
                                              (((sc[0]-self.ellipse_poses[0,1])*ca.cos(self.ellipse_angles[1])-
                                                (sc[1]-self.ellipse_poses[1,1])*ca.sin(self.ellipse_angles[1]))**2/self.ellipse_rads[0,1]**2)+
                                              (((sc[0]-self.ellipse_poses[0,1])*ca.sin(self.ellipse_angles[1])+
                                                (sc[1]-self.ellipse_poses[1,1])*ca.cos(self.ellipse_angles[1]))**2/self.ellipse_rads[1,1]**2)-1)<=0)                    
                index+=1

        elif self.n==3:
            for si in self.s:
                sc=np.array(si).reshape((2,1))
                self.opti.subject_to(ca.fmin((((sc[0]-self.ellipse_poses[0,0])*ca.cos(self.ellipse_angles[0])-
                                               (sc[1]-self.ellipse_poses[1,0])*ca.sin(self.ellipse_angles[0]))**2/self.ellipse_rads[0,0]**2)+
                                             (((sc[0]-self.ellipse_poses[0,0])*ca.sin(self.ellipse_angles[0])+
                                               (sc[1]-self.ellipse_poses[1,0])*ca.cos(self.ellipse_angles[0]))**2/self.ellipse_rads[1,0]**2)-1,
                                     ca.fmin((((sc[0]-self.ellipse_poses[0,1])*ca.cos(self.ellipse_angles[1])-
                                               (sc[1]-self.ellipse_poses[1,1])*ca.sin(self.ellipse_angles[1]))**2/self.ellipse_rads[0,1]**2)+
                                             (((sc[0]-self.ellipse_poses[0,1])*ca.sin(self.ellipse_angles[1])+
                                               (sc[1]-self.ellipse_poses[1,1])*ca.cos(self.ellipse_angles[1]))**2/self.ellipse_rads[1,1]**2)-1,
                                             (((sc[0]-self.ellipse_poses[0,2])*ca.cos(self.ellipse_angles[2])-
                                               (sc[1]-self.ellipse_poses[1,2])*ca.sin(self.ellipse_angles[2]))**2/self.ellipse_rads[0,2]**2)+
                                             (((sc[0]-self.ellipse_poses[0,2])*ca.sin(self.ellipse_angles[2])+
                                               (sc[1]-self.ellipse_poses[1,2])*ca.cos(self.ellipse_angles[2]))**2/self.ellipse_rads[1,2]**2)-1))<=0)

        elif self.n==4:
            for si in self.s:
                sc=np.array(si).reshape((2,1))
                self.opti.subject_to(ca.fmin((((sc[0]-self.ellipse_poses[0,0])*ca.cos(self.ellipse_angles[0])-
                                               (sc[1]-self.ellipse_poses[1,0])*ca.sin(self.ellipse_angles[0]))**2/self.ellipse_rads[0,0]**2)+
                                             (((sc[0]-self.ellipse_poses[0,0])*ca.sin(self.ellipse_angles[0])+
                                               (sc[1]-self.ellipse_poses[1,0])*ca.cos(self.ellipse_angles[0]))**2/self.ellipse_rads[1,0]**2)-1,
                                     ca.fmin((((sc[0]-self.ellipse_poses[0,1])*ca.cos(self.ellipse_angles[1])-
                                               (sc[1]-self.ellipse_poses[1,1])*ca.sin(self.ellipse_angles[1]))**2/self.ellipse_rads[0,1]**2)+
                                             (((sc[0]-self.ellipse_poses[0,1])*ca.sin(self.ellipse_angles[1])+
                                               (sc[1]-self.ellipse_poses[1,1])*ca.cos(self.ellipse_angles[1]))**2/self.ellipse_rads[1,1]**2)-1,
                                     ca.fmin((((sc[0]-self.ellipse_poses[0,2])*ca.cos(self.ellipse_angles[2])-
                                               (sc[1]-self.ellipse_poses[1,2])*ca.sin(self.ellipse_angles[2]))**2/self.ellipse_rads[0,2]**2)+
                                             (((sc[0]-self.ellipse_poses[0,2])*ca.sin(self.ellipse_angles[2])+
                                               (sc[1]-self.ellipse_poses[1,2])*ca.cos(self.ellipse_angles[2]))**2/self.ellipse_rads[1,2]**2)-1,
                                             (((sc[0]-self.ellipse_poses[0,3])*ca.cos(self.ellipse_angles[3])-
                                               (sc[1]-self.ellipse_poses[1,3])*ca.sin(self.ellipse_angles[3]))**2/self.ellipse_rads[0,3]**2)+
                                             (((sc[0]-self.ellipse_poses[0,3])*ca.sin(self.ellipse_angles[3])+
                                               (sc[1]-self.ellipse_poses[1,3])*ca.cos(self.ellipse_angles[3]))**2/self.ellipse_rads[1,3]**2)-1)))<=0)

        elif self.n==5:
            for si in self.s:
                sc=np.array(si).reshape((2,1))
                self.opti.subject_to(ca.fmin((((sc[0]-self.ellipse_poses[0,0])*ca.cos(self.ellipse_angles[0])-
                                               (sc[1]-self.ellipse_poses[1,0])*ca.sin(self.ellipse_angles[0]))**2/self.ellipse_rads[0,0]**2)+
                                             (((sc[0]-self.ellipse_poses[0,0])*ca.sin(self.ellipse_angles[0])+
                                               (sc[1]-self.ellipse_poses[1,0])*ca.cos(self.ellipse_angles[0]))**2/self.ellipse_rads[1,0]**2)-1,
                                     ca.fmin((((sc[0]-self.ellipse_poses[0,1])*ca.cos(self.ellipse_angles[1])-
                                               (sc[1]-self.ellipse_poses[1,1])*ca.sin(self.ellipse_angles[1]))**2/self.ellipse_rads[0,1]**2)+
                                             (((sc[0]-self.ellipse_poses[0,1])*ca.sin(self.ellipse_angles[1])+
                                               (sc[1]-self.ellipse_poses[1,1])*ca.cos(self.ellipse_angles[1]))**2/self.ellipse_rads[1,1]**2)-1,
                                     ca.fmin((((sc[0]-self.ellipse_poses[0,2])*ca.cos(self.ellipse_angles[2])-
                                               (sc[1]-self.ellipse_poses[1,2])*ca.sin(self.ellipse_angles[2]))**2/self.ellipse_rads[0,2]**2)+
                                             (((sc[0]-self.ellipse_poses[0,2])*ca.sin(self.ellipse_angles[2])+
                                               (sc[1]-self.ellipse_poses[1,2])*ca.cos(self.ellipse_angles[2]))**2/self.ellipse_rads[1,2]**2)-1,
                                     ca.fmin((((sc[0]-self.ellipse_poses[0,3])*ca.cos(self.ellipse_angles[3])-
                                               (sc[1]-self.ellipse_poses[1,3])*ca.sin(self.ellipse_angles[3]))**2/self.ellipse_rads[0,3]**2)+
                                             (((sc[0]-self.ellipse_poses[0,3])*ca.sin(self.ellipse_angles[3])+
                                               (sc[1]-self.ellipse_poses[1,3])*ca.cos(self.ellipse_angles[3]))**2/self.ellipse_rads[1,3]**2)-1,
                                             (((sc[0]-self.ellipse_poses[0,4])*ca.cos(self.ellipse_angles[4])-
                                               (sc[1]-self.ellipse_poses[1,4])*ca.sin(self.ellipse_angles[4]))**2/self.ellipse_rads[0,4]**2)+
                                             (((sc[0]-self.ellipse_poses[0,4])*ca.sin(self.ellipse_angles[4])+
                                               (sc[1]-self.ellipse_poses[1,4])*ca.cos(self.ellipse_angles[4]))**2/self.ellipse_rads[1,4]**2)-1))))<=0)

        elif self.n==6:
            for si in self.s:
                sc=np.array(si).reshape((2,1))
                self.opti.subject_to(ca.fmin((((sc[0]-self.ellipse_poses[0,0])*ca.cos(self.ellipse_angles[0])-
                                               (sc[1]-self.ellipse_poses[1,0])*ca.sin(self.ellipse_angles[0]))**2/self.ellipse_rads[0,0]**2)+
                                             (((sc[0]-self.ellipse_poses[0,0])*ca.sin(self.ellipse_angles[0])+
                                               (sc[1]-self.ellipse_poses[1,0])*ca.cos(self.ellipse_angles[0]))**2/self.ellipse_rads[1,0]**2)-1,
                                     ca.fmin((((sc[0]-self.ellipse_poses[0,1])*ca.cos(self.ellipse_angles[1])-
                                               (sc[1]-self.ellipse_poses[1,1])*ca.sin(self.ellipse_angles[1]))**2/self.ellipse_rads[0,1]**2)+
                                             (((sc[0]-self.ellipse_poses[0,1])*ca.sin(self.ellipse_angles[1])+
                                               (sc[1]-self.ellipse_poses[1,1])*ca.cos(self.ellipse_angles[1]))**2/self.ellipse_rads[1,1]**2)-1,
                                     ca.fmin((((sc[0]-self.ellipse_poses[0,2])*ca.cos(self.ellipse_angles[2])-
                                               (sc[1]-self.ellipse_poses[1,2])*ca.sin(self.ellipse_angles[2]))**2/self.ellipse_rads[0,2]**2)+
                                             (((sc[0]-self.ellipse_poses[0,2])*ca.sin(self.ellipse_angles[2])+
                                               (sc[1]-self.ellipse_poses[1,2])*ca.cos(self.ellipse_angles[2]))**2/self.ellipse_rads[1,2]**2)-1,
                                     ca.fmin((((sc[0]-self.ellipse_poses[0,3])*ca.cos(self.ellipse_angles[3])-
                                               (sc[1]-self.ellipse_poses[1,3])*ca.sin(self.ellipse_angles[3]))**2/self.ellipse_rads[0,3]**2)+
                                             (((sc[0]-self.ellipse_poses[0,3])*ca.sin(self.ellipse_angles[3])+
                                               (sc[1]-self.ellipse_poses[1,3])*ca.cos(self.ellipse_angles[3]))**2/self.ellipse_rads[1,3]**2)-1,
                                     ca.fmin((((sc[0]-self.ellipse_poses[0,4])*ca.cos(self.ellipse_angles[4])-
                                               (sc[1]-self.ellipse_poses[1,4])*ca.sin(self.ellipse_angles[4]))**2/self.ellipse_rads[0,4]**2)+
                                             (((sc[0]-self.ellipse_poses[0,4])*ca.sin(self.ellipse_angles[4])+
                                               (sc[1]-self.ellipse_poses[1,4])*ca.cos(self.ellipse_angles[4]))**2/self.ellipse_rads[1,4]**2)-1,
                                             (((sc[0]-self.ellipse_poses[0,5])*ca.cos(self.ellipse_angles[5])-
                                               (sc[1]-self.ellipse_poses[1,5])*ca.sin(self.ellipse_angles[5]))**2/self.ellipse_rads[0,5]**2)+
                                             (((sc[0]-self.ellipse_poses[0,5])*ca.sin(self.ellipse_angles[5])+
                                               (sc[1]-self.ellipse_poses[1,5])*ca.cos(self.ellipse_angles[5]))**2/self.ellipse_rads[1,5]**2)-1)))))<=0)

        else:
            print("n={} is not implemented!!!".format(self.n))

        for i in range(self.n):
            if self.n==1:
                self.opti.subject_to(self.ellipse_rads[0]>0)
                self.opti.subject_to(self.ellipse_rads[1]>0)
            else:
                self.opti.subject_to(self.ellipse_rads[0,i]>0)
                self.opti.subject_to(self.ellipse_rads[1,i]>0)
        
        self.opti.minimize(self.cost)

        self.ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 1000,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": True
            }

        self.opti.solver('ipopt', self.ipopt_options)

        pose_initial, rads_initial, angle_initial=self.get_initial()

        self.opti.set_initial(self.ellipse_poses,pose_initial)
        self.opti.set_initial(self.ellipse_rads,rads_initial)
        self.opti.set_initial(self.ellipse_angles,angle_initial)
    
    def solve_problem(self, verbose=True):
        sol=self.opti.solve()

        self.pose_results=self.opti.value(self.ellipse_poses)
        self.rads_results=self.opti.value(self.ellipse_rads)
        self.angle_results=self.opti.value(self.ellipse_angles)

        if verbose:
            print("Optimal ellipse positions are {}".format(self.pose_results))
            print("Optimal ellipse radiuses are {}".format(self.rads_results))
            print("Optimal ellipse angles are {}".format(self.angle_results))

        return self.pose_results,self.rads_results,self.angle_results

    def get_initial(self):
        x_mean=self.circle[0]
        y_mean=self.circle[1]
        r=self.circle[2]

        pose_initial=np.ones((2,self.n))
        pose_initial[0,:]=x_mean
        pose_initial[1,:]=y_mean

        rads_initial=np.ones((2,self.n))*r 

        angle_initial=np.zeros(self.n)

        #pose_initial=np.array([[3,10,18],[12,10,8]]) 

        return pose_initial, rads_initial, angle_initial 

    def get_inner_points(self,obstacle):
        s=[]
        for i in range(obstacle.shape[0]):
            for j in range(obstacle.shape[1]):
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

        return s,[x_mean,y_mean,r]
    
    def get_barrier_points(self,obstacle):
        s=[]
        barriers=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
        for i in range(obstacle.shape[0]):
            for j in range(obstacle.shape[1]):
                if obstacle[i][j]!=0:
                    statu=True
                    for barrier in barriers:
                        if obstacle[i+barrier[0]][j+barrier[1]]==0:
                            s.append([i,j])
                            break
        
        return np.array(s)

    def visualize(self):
        plt.rcParams["figure.figsize"] = (10,10)
        ax = plt.gca()
        ax.cla()

        for si in self.s:
            ci=plt.Circle((si[0],si[1]),.1,color='g')
            ax.add_patch(ci)

        a1=plt.Circle((self.circle[0],self.circle[1]),.5,color='r')
        a2=plt.Circle((self.circle[0],self.circle[1]),self.circle[2],color='r',fill=False)
        ax.add_patch(a1)
        ax.add_patch(a2)

        colors=['b','c','m','y','k','orange']
        if self.n==1:
            a1=plt.Circle((self.pose_results[0],self.pose_results[1]),.5,color=colors[0])
            ax.add_patch(a1)
            a1=Ellipse(self.pose_results,
                    width=2*self.rads_results[0],
                    height=2*self.rads_results[1],
                    angle=(np.pi-self.angle_results)*180/np.pi,
                    color=colors[0],
                    fill=False)
            ax.add_artist(a1)      
        else:
            for i in range(self.n):
                a1=plt.Circle((self.pose_results[0,i],self.pose_results[1,i]),.5,color=colors[i])
                ax.add_patch(a1)
                a1=Ellipse(self.pose_results[:,i],
                        width=2*self.rads_results[0,i],
                        height=2*self.rads_results[1,i],
                        angle=(np.pi-self.angle_results[i])*180/np.pi,
                        color=colors[i],
                        fill=False)
                ax.add_artist(a1)
        
        ax.set_xlim((-2,self.obstacle.shape[0]+2))
        ax.set_ylim((-2,self.obstacle.shape[1]+2))
        ax.grid()
        plt.show()
