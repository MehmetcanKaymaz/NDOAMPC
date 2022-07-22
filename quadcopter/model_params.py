import numpy as np


def get_model_params():
    m=.85
    l=.14
    g=9.81
    I=np.array([[1e-3,0,0],[0,1e-3,0],[0,0,1.7e-3]])
    I_inv=np.linalg.inv(I)
    T_max=3.3
    T_min=0
    omega_max=3
    ctau=.05
    cd=0.0

    return m,l,g,I,I_inv,T_max,T_min,omega_max,ctau,cd