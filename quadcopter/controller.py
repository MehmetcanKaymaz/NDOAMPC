import casadi as ca
from casadi import MX, DM, vertcat, mtimes, Function, inv, cross, sqrt, norm_2
import numpy as np
from scipy import sparse
from model_params import get_model_params
from quat import quat_mult_ca 
from quat import rotate_quat_ca  

class NMPC:
    def __init__(self):
        self.info="Nonlinear model predictive controller for quadcopter"

        self.x_dim=13
        self.u_dim=4
        self.N=20
        self.dt=.01

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

        self.Q=sparse.diags([1,1,1,0,0,0,0,0,0,0,0,0,0])
        #self.S=sparse.diags([1,1,1,0,0,0,0,0,0,0,0,0,0])
        self.R=sparse.eye(4)*.01    

        self._initDynamics()

    def _initDynamics(self,):
        #Dynamic Model
        p = MX.sym('p', 3)
        v = MX.sym('v', 3)
        q = MX.sym('q', 4)
        w = MX.sym('w', 3)
        T = MX.sym('thrust', 4)

        x = vertcat(p, v, q, w)
        u = vertcat(T)

        g = DM([0, 0, -self.g])

        x_dot = vertcat(
        v,
        rotate_quat_ca(q, vertcat(0, 0, (T[0]+T[1]+T[2]+T[3])/self.m)) + g - v * self.cd,
        0.5*quat_mult_ca(q, vertcat(0, w)),
        mtimes(self.I_inv, vertcat(
            self.l*(T[0]-T[1]-T[2]+T[3]),
            self.l*(-T[0]-T[1]+T[2]+T[3]),
            self.ctau*(T[0]-T[1]+T[2]-T[3]))
        -cross(w,mtimes(self.I,w)))
        )

        fx = Function('f',  [x, u], [x_dot], ['x', 'u'], ['x_dot'])         

        intg_options = {'tf': self.dt, 'simplify': True,
                        'number_of_finite_elements': 1}
        dae = {'x': x, 'p': u, 'ode': fx(x, u)}

        intg = ca.integrator('intg', 'rk', dae, intg_options)

        x_next = intg(x0=x, p=u)['xf']
        F = ca.Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])
        
        #Cost Functions
        Delta_x = ca.SX.sym("Delta_x", self.x_dim)
        Delta_u = ca.SX.sym("Delta_u", self.u_dim)

        cost_track = Delta_x.T @ self.Q @ Delta_x
        cost_u = Delta_u.T @ self.R @ Delta_u

        f_cost_track = ca.Function('cost_track', [Delta_x], [cost_track])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])       

        #Upper and lower bounds
        u_min=[self.T_min for _ in range(self.u_dim)]
        u_max=[self.T_max for _ in range(self.u_dim)]
        x_min=[-ca.inf for _ in range(self.x_dim)]
        x_max=[ca.inf for _ in range(self.x_dim)]

        #Optimization
        opti=ca.Opti()

        X=opti.variable(self.x_dim, self.N+1)
        self.X0=opti.parameter(self.x_dim)
        self.P=opti.parameter(self.x_dim, self.N)
        self.U=opti.variable(self.u_dim, self.N)

        cost=0

        opti.subject_to(X[:,0]==self.X0)

        for k in range(self.N):
            cost+=f_cost_track((X[:, k+1]-self.P[:,k]))
            if k!=0:
                cost+=f_cost_u((self.U[:, k]-self.U[:, k-1]))
            
            opti.subject_to(X[:, k+1]==F(X[:, k],self.U[:, k]))
            opti.subject_to(self.U[:, k] <= u_max)
            opti.subject_to(self.U[:, k] >= u_min)
            opti.subject_to(X[:, k+1] >= x_min)
            opti.subject_to(X[:, k+1] <= x_max)           


        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": True
        }

        opti.minimize(cost)
        opti.solver('ipopt', ipopt_options)

        self.opti=opti

    def run(self, x0, ref):
        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.P, ref)

        sol=self.opti.solve()
        u=sol.value(self.U)[:,0]

        return u