import numpy as np
from casadi import MX, DM, vertcat, mtimes, Function, inv, cross, sqrt, norm_2

def quat_mult(q1,q2):
    a1=q2[0] * q1[0] - q2[1] * q1[1] - q2[2] * q1[2] - q2[3] * q1[3]
    a2=q2[0] * q1[1] + q2[1] * q1[0] - q2[2] * q1[3] + q2[3] * q1[2]
    a3=q2[0] * q1[2] + q2[2] * q1[0] + q2[1] * q1[3] - q2[3] * q1[1]
    a4=q2[0] * q1[3] - q2[1] * q1[2] + q2[2] * q1[1] + q2[3] * q1[0]

    return np.array([a1,a2,a3,a4]).reshape(4,1)

def rotate_quad(q1,v1):
    v1_f=np.array([0,v1[0],v1[1],v1[2]],dtype='float32').reshape(4,1)
    q1_f=quat_mult(q1,v1_f)
    q2=np.array([q1[0],-q1[1],-q1[2],-q1[3]]).reshape(4,1)
    ans=quat_mult(q1_f,q2)

    return np.array([ans[1],ans[2],ans[3]]).reshape(3,1)


# For casadi
# Quaternion Multiplication
def quat_mult_ca(q1,q2):
    ans = vertcat(q2[0,:] * q1[0,:] - q2[1,:] * q1[1,:] - q2[2,:] * q1[2,:] - q2[3,:] * q1[3,:],
           q2[0,:] * q1[1,:] + q2[1,:] * q1[0,:] - q2[2,:] * q1[3,:] + q2[3,:] * q1[2,:],
           q2[0,:] * q1[2,:] + q2[2,:] * q1[0,:] + q2[1,:] * q1[3,:] - q2[3,:] * q1[1,:],
           q2[0,:] * q1[3,:] - q2[1,:] * q1[2,:] + q2[2,:] * q1[1,:] + q2[3,:] * q1[0,:])
    return ans

# Quaternion-Vector Rotation
def rotate_quat_ca(q1,v1):
    ans = quat_mult_ca(quat_mult_ca(q1, vertcat(0, v1)), vertcat(q1[0,:],-q1[1,:], -q1[2,:], -q1[3,:]))
    return vertcat(ans[1,:], ans[2,:], ans[3,:]) # to covert to 3x1 vec