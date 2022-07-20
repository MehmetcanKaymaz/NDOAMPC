import numpy as np
from cvxpy import Parameter, Variable, quad_form, Problem, Minimize, OSQP
import cvxpy as cp


x=Variable((2,1))
r=Variable((2,2))

print(quad_form(x,r).curvature)