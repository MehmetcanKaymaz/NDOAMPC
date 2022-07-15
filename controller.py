from casadi import MX, DM, vertcat, mtimes, Function, inv, cross, sqrt, norm_2
import casadi as ca
import numpy as np


class NMPC:
    def __init__(self,n_obstacle,obstacle_r,quadcopter_r):
        self.x_dim=2
        self.u_dim=2

        self.n_obstacle=n_obstacle
        self.obstacle_r=obstacle_r
        self.quadcopter_r=quadcopter_r
        self.epsilon=0.01

        self.Q = np.diag([1,1])
        self.R = np.diag([1,1])*0.001

        self.S=np.eye(self.n_obstacle)
        self.theta=.15
        
        self.N=20

        self.dt=.01

        self.A=np.eye(2)+np.array([[0,0],
                                   [0,0]])*self.dt
                
        self.B=np.array([[1,0],
                         [0,1]])*self.dt

        self._initDynamics()

    def _initDynamics(self,):


        umin = [-5, -5]
        umax = [5, 5]
        xmin = [-ca.inf,-ca.inf]
        xmax = [ca.inf,ca.inf]

        opti = ca.Opti()

        X = opti.variable(self.x_dim, self.N+1)
        
        self.P = opti.parameter(self.x_dim, 2)
        
        self.U = opti.variable(self.u_dim, self.N)

        if self.n_obstacle!=0:
            self.O=opti.parameter(2*self.n_obstacle,self.N+1)

            S=opti.variable(self.n_obstacle,self.N)

        cost = 0

        opti.subject_to(X[:, 0] == self.P[:, 0])

        for k in range(self.N):
            cost += (X[:, k+1]-self.P[:, 1]).T@self.Q@(X[:, k+1]-self.P[:, 1])
            if self.n_obstacle!=0:
                cost += (S[:,k].T@self.S@S[:,k])
            
            if k == 0:
                cost += 0
            else:
                cost += (self.U[:, k]-self.U[:, k-1]).T@self.R@(self.U[:, k]-self.U[:, k-1])

            if self.n_obstacle!=0:
                for i in range(self.n_obstacle):
                    opti.subject_to((X[:,k+1]-self.O[i*2:i*2+2,k+1]).T@(X[:,k+1]-self.O[i*2:i*2+2,k+1])+self.theta*S[i,k]>=self.obstacle_r[i]+self.quadcopter_r+self.epsilon)
                
            opti.subject_to(X[:, k+1] == self.A@X[:,k]+self.B@self.U[:,k])
            opti.subject_to(self.U[:, k] <= umax)
            opti.subject_to(self.U[:, k] >= umin)
            opti.subject_to(X[:, k+1] >= xmin)
            opti.subject_to(X[:, k+1] <= xmax)

        opti.minimize(cost)
        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 1000,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False
        }


        opti.solver('ipopt', ipopt_options)
        self.opti = opti

    def run_controller(self, x0, ref_states,obstacle_states):#

        p = np.concatenate((x0.reshape((2, 1)), ref_states), axis=1)

        if self.n_obstacle!=0:
            locs=np.zeros((2*self.n_obstacle,self.N+1))

            for i in range(self.n_obstacle):
                loc=self.__calculate_obstacle_loc(obstacle_states[i])
                locs[i*2:i*2+2,:]=loc

            
            self.opti.set_value(self.O,locs)
        
        self.opti.set_value(self.P, p)
        sol = self.opti.solve()
        u = sol.value(self.U)[:, 0]
        return u

    def __calculate_obstacle_loc(self,obstacle_states):
        locs=np.zeros((2,self.N+1))
        locs[:,0]=obstacle_states[:2]

        past_state=obstacle_states

        for i in range(self.N):
            next_state=np.matmul(self.A,past_state)
            locs[:,i+1]=next_state[:2]
            past_state=next_state
        
        return locs