# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:31:09 2017

@author: Dogan
"""
# Do not write bytecode to maintain clean directories
import sys
sys.path.append("C:\\Users\\Dogan\\Documents\\Casadi_Python")
sys.path.append("C:\\Users\\Dogan\\Dropbox\\COMSOL\\Flow_model")
sys.dont_write_bytecode = True

# Imports required packages.
from casadi import *
import numpy as NP
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.ticker import MaxNLocator
from scipy import linalg
from scipy import signal
from numpy import random
from scipy import io as sio
import matplotlib.pyplot as plt

import model_v1 as jet
import EKF_v1 as observer
# Import core code.
import core

### IMPORT MAT FILE###
mat=sio.loadmat('intermed.mat')
prevopt=mat['optimal']
Qx=mat['Qx']
Qz=mat['Qz']
Ru=mat['Ru']
Qt=mat['Qt']
N=(mat['N'])
xset=mat['xset'].flatten()
uset=mat['uset'].flatten()
zset=mat['zset'].flatten()

######################################## MODEL ########################
#######################################################################

[x, z, u, d, v, xdot, gn, I] = jet.model()

nx=int(x.dim()[0])
nz=int(z.dim()[0])
nu=int(u.dim()[0])
nv=int(v.dim()[0])

# test integrator
#x_0=[313.0/300.0, 313.0/300.0, 313.0/300.0, 0.0, 0.0, 0.0, (313.0+20.0)/300.0,  (295.0+20.0)/300.0]
x_0=[313.0/300.0, 0.0, (313.0+20.0)/300.0,  (295.0+20.0)/300.0]

Ik = I(x0=x_0,p=[3.75,15,2,4],z0=[3,3,3,3]);

#print(Ik['zf'])
#print(Ik['xf'])

######################## MPC ######################################
###################################################################



#uss=[4,15,2]
dss=4



#xss=Ik['xf'].full().flatten().tolist()
xss=mat['x0'].flatten().tolist()
uss=mat['u0'].flatten().tolist()
zss=mat['z0'].flatten().tolist()
dis=mat['dis'].flatten().tolist()
err=mat['error']
Pk_1=mat['Pk']
Qk=mat['Qk']
Rk=mat['Rk']



y_hat_k=[err.flatten().tolist()[2],err.flatten().tolist()[3],err.flatten().tolist()[4]]


[xss,dis,zss,Pk]=observer.EKF(x,z,u,d,v,xdot,gn,y_hat_k,Pk_1,Qk,Rk,xss,dis,zss,uss)


Qr=NP.diag([0.01,0.001,0.01])
# bounds on the control inputs
u_ub = [5,20,5]
u_lb = [3,10,1.5]

du_ub = [0.5,1,0.5]
du_lb = [-0.8,-1,-0.8]

#x_lb = [-inf, -inf, -inf, -inf, -inf]
x_lb = [0, 0, 0, 0, 0]
x_ub = [inf, inf, inf, inf, 10]

#z_lb = [-inf, -inf, -inf, -inf]
z_lb = [0, 0, 0, 0]
z_ub = [inf , inf, inf, inf]

#zss=[3,3,3,3]

#zss = core.rootFinder(alg, nz, args=[uss,dss])

#xss = core.rootFinder(diffs, nx, args=[zss,uss,dss], x0=x_0[0:nx-1])
### Build instance of MPC problem
# start with an empty NLP
q = []
q0 = []
lbq = []
ubq = []
J = 0
g = []
lbg = []
ubg = []
# "lift" initial conditions


# formulate the NLP
X0 = MX.sym('X0', nx)
q  += [X0]
qz = []
lbq += xss
ubq += xss
Zk = zss
q0 += xss
Xk = X0
U0 = uss

for k in range(N):
    # new NLP variable for the control
    Uk = MX.sym('U_' + str(k), nu)
    q   += [Uk]
    lbq += [u_lb[i] for i in range(nu)]
    ubq += [u_ub[i] for i in range(nu)]
    q0  += uss
    
 
    # next step dynamics 
    Fk = I(x0=Xk ,p=vertcat(Uk,4),z0=Zk)
    Xk_end = Fk['xf']
    Xk_end[2]=Xk_end[2]+dis[0]
    Xk_end[3]=Xk_end[3]
    Zk_end = Fk['zf']
    Zk_end[1]=Zk_end[1]+dis[1]

    #add to stage cost
    J = J + 0*(Xk[2]-xset[2])**2.0 + 30*(Xk[3]-xset[3])**2.0  + 0.00*(Zk[1]-zset[1])**2 #+ 0*core.mtimes((Uk-uss).T,Qr,(Uk-uss))
    #J=J+Fk['qf'] 
   # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k+1), nx)
    q += [Xk]
    lbq += [x_lb[i] for i in range(nx)]     
    ubq += [x_ub[i] for i in range(nx)]     
    q0 += xss
    
    Zk = MX.sym('Z_' + str(k+1), nz)
    q += [Zk]
    lbq += [z_lb[i] for i in range(nz)]     
    ubq += [z_ub[i] for i in range(nz)]     
    q0 += zss
    
    # Add equality constraint
#    g   += [Xk_end-Xk]
    g   += [Xk_end-Xk]
    lbg += [0]*nx
    ubg += [0]*nx
    
    g   += [Zk_end-Zk]
    lbg += [0]*nz
    ubg += [0]*nz
    
    g   += [Uk-U0]
    lbg += [du_lb[i] for i in range(nu)]
    ubg += [du_ub[i] for i in range(nu)]
    U0=Uk
    
#terminal cost
J = J + 0*(Xk[2]-1.04)**2 
print(J)
MySolver = "ipopt"
#MySolver = "sqpmethod"

sol_opts={}
#sol_opts['ipopt']={'linear_solver':'wsmp'} syntax to pass opts to IPOPT
#opts["linear_solver"] = "ma57"
#opts["hessian_approximation"] = "limited-memory"

if MySolver == "sqpmethod":
  sol_opts["qpsol"] = "qpoases"
  #sol_opts["qpsol_options"] = {"printLevel":"none"}
elif MySolver == 'ipopt':
  #sol_opts={'ipopt.hessian_approximation':'limited-memory','ipopt.fixed_variable_treatment':'relax_bounds' ,'ipopt.jacobian_approximation':'finite-difference-values'}
  sol_opts={'ipopt.hessian_approximation':'exact','ipopt.fixed_variable_treatment':'relax_bounds' ,'ipopt.tol':1e-2, 'ipopt.max_cpu_time':35}

if len(mat['optimal'])>2:
    q0=vertcat(xss,mat['optimal'][nx:])    
 
prob = {'f':J, 'x':vertcat(*q), 'g': vertcat(*g)}
solver = nlpsol('solver', MySolver, prob, sol_opts)
 
sol = solver(x0=q0, lbx=lbq, ubx=ubq, lbg=lbg, ubg=ubg)

#save_file.flush()
#save_file.close()

export_dict = {
    "optimal":sol['x'].full(),
    "error": err,
    "x0" : xss,
    "u0": uss,
    "z0": zss,
    "xset": xset,
    "uset":uset,
    "zset":zset,
    "dis":dis,
    "Pk":Pk,
    "Qx":Qx,
    "Qz":Qz,
    "Ru":Ru,
    "Qk":Qk,
    "Rk":Rk,
    "Qt":Qt,
    "N":N,
    }

sio.savemat('intermed', mdict=export_dict)
    
    
#sio.savemat('intermed.mat',{sol['x'].full().flatten()})
#print(sol['x'].full().flatten())
