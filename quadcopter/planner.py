import numpy as np
import casadi as ca

class SimplePlanner:
    def __init__(self):
        self.info="Simple trajectory planner for mpc test"

        self.N=20

    def plan(self,x0,ref):
        delta=(ref-x0)/self.N

        traj=np.ones((3,self.N))
        traj[:,0]=x0

        for i in range(1,self.N):
            traj[:,i]=traj[:,i-1]+delta
        
        total_traj=np.zeros((13,self.N))
        total_traj[:3,:]=traj
        return total_traj

class CAPlanner:
    def __init__(self,n_obstacle):
        self.info="Collusion avodiance path planner"

        self.n_obstacle=n_obstacle
        self.x_dim=3
        self.u_dim=3

        self.N=50
        self.dt=.01

        self.A=np.eye(3)
        self.B=np.eye(3)*self.dt

        self.A_obstacle=np.eye(6)+np.array([[0,0,0,1,0,0],
                                            [0,0,0,0,1,0],
                                            [0,0,0,0,0,1],
                                            [0,0,0,0,0,0],
                                            [0,0,0,0,0,0],
                                            [0,0,0,0,0,0]])*self.dt

        self.theta=.1

        u_min=[-1, -1,-1]
        u_max=[1, 1,1]
        x_min=[-ca.inf,-ca.inf,-ca.inf]
        x_max=[ca.inf,ca.inf,ca.inf]

        opti=ca.Opti()

        self.X=opti.variable(self.x_dim,self.N+1)
        self.X0=opti.parameter(self.x_dim)
        self.T=opti.parameter(self.x_dim)
        self.U=opti.variable(self.u_dim,self.N)

        self.O=opti.parameter(3*self.n_obstacle,self.N+1)
        self.S=opti.variable(self.n_obstacle,self.N)

        cost=(self.X[:,self.N]-self.T).T@(self.X[:,self.N]-self.T)

        opti.subject_to(self.X[:,0]==self.X0)

        for k in range(self.N):
            cost+=(self.S[:,k].T@self.S[:,k])

            for i in range(self.n_obstacle):
                opti.subject_to((self.X[:,k+1]-self.O[i*3:i*3+3,k+1]).T@(self.X[:,k+1]-self.O[i*3:i*3+3,k+1])+self.theta*self.S[i,k]>=1.1)

            opti.subject_to(self.X[:, k+1] == self.A@self.X[:,k]+self.B@self.U[:,k])
            opti.subject_to(self.U[:, k] <= u_max)
            opti.subject_to(self.U[:, k] >= u_min)
            opti.subject_to(self.X[:, k+1] >= x_min)
            opti.subject_to(self.X[:, k+1] <= x_max)

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

    def plan(self,x0,ref,obstacle_states):
        locs=self.__calculate_obstacle_loc(obstacle_states=obstacle_states)

        self.opti.set_value(self.X0,x0[:3])
        self.opti.set_value(self.O,locs)
        self.opti.set_value(self.T,ref)

        sol=self.opti.solve()
        X=sol.value(self.X)[:,1]

        return X

    
    def __calculate_obstacle_loc(self,obstacle_states):
        locs=np.zeros((3*self.n_obstacle,self.N+1))

        for i in range(self.n_obstacle):
            loc_i=np.zeros((6,self.N+1))
            locs[:,0]=obstacle_states[i]

            for j in range(self.N):
                loc_i[:,j+1]=np.matmul(self.A_obstacle,loc_i[:,j])
            
            locs[3*i:3*i+3,:]=loc_i[:3,:]
        
        return locs