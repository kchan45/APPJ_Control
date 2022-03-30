# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:06:50 2017

@author: Dogan
"""
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
import core


def EKF(x,z,u,d,v,xdot,gn,y_hat_k,Pk_1,Qk,Rk,xss,dis,zss,uss,delta):

    nx=int(x.dim()[0])
    nz=int(z.dim()[0])
    nu=int(u.dim()[0])
    nv=int(v.dim()[0])

    A1=Function('A1',[x,z,u,d],[jacobian(vertcat(xdot,v),vertcat(x,v))])
    B1=Function('B1',[x,z,u,d],[jacobian(vertcat(xdot,v),z)])
    C1=Function('C1',[x,z,u,d],[jacobian(gn,vertcat(x,v))])
    D1=Function('D1',[x,z,u,d],[jacobian(gn,z)])

    Alow1=[core.mtimes(-NP.linalg.inv(D1(xss,zss,uss,4)),C1(xss,zss,uss,4),A1(xss,zss,uss,4))]
    Alow2=[core.mtimes(-NP.linalg.inv(D1(xss,zss,uss,4)),C1(xss,zss,uss,4),B1(xss,zss,uss,4))]

    Aaug=vertcat(horzcat(A1(xss,zss,uss,4), B1(xss,zss,uss,4)),horzcat(Alow1[0], Alow2[0]))

    theta=linalg.expm(Aaug.full()*delta)
    gamma=vertcat(NP.eye(nx+nv),core.mtimes(linalg.inv(D1(xss,zss,uss,4)),C1(xss,zss,uss,4))).full()
    
    Hk=NP.array([(0, 0, 1, 0, 1 ,0 ,0, 0, 0, 0),
                 (0, 0, 0, 1, 0, 0, 0 ,0, 0, 0),
                 (0, 0, 0, 0, 0, 1, 0 ,1, 0, 0)])
    
    Pk_k_1=core.mtimes(theta,Pk_1,theta.T)+core.mtimes(gamma,Qk,gamma.T)
    
    Sk=core.mtimes(Hk,Pk_k_1,Hk.T)+Rk
    Kk=core.mtimes(Pk_k_1,Hk.T,NP.linalg.inv(Sk))
    
    aug_k=vertcat(xss,dis,zss)+core.mtimes(Kk,y_hat_k)
    Pk=core.mtimes(NP.eye(nx+nz+nv)-core.mtimes(Kk,Hk),Pk_k_1)
    
    xss=aug_k[0:nx].full().flatten().tolist()
    dis=aug_k[nx:nx+nv].full().flatten()
    zss=aug_k[nx+nv:].full().flatten().tolist()
    
    
    #print(NP.linalg.matrix_rank(control.ctrb(theta.T,Hk.T)))
    #print(NP.linalg.eigvals(theta-core.mtimes(Kk,Hk)))    
    return [xss,dis,zss,Pk]
    
def c_EKF(x,z,u,d,v,xdot,gn,y_hat_k,Pk_1,Qk,Rk,xss,dis,zss,uss,delta):

    nx=int(x.dim()[0])
    nz=int(z.dim()[0])
    nu=int(u.dim()[0])
    nv=int(v.dim()[0])
    
    P=MX.sym('P',nx+nv+nz,nx+nv+nz)
    
    A1=Function('A1',[x,z,u,d],[jacobian(vertcat(xdot,v),vertcat(x,v))])
    B1=Function('B1',[x,z,u,d],[jacobian(vertcat(xdot,v),z)])    
    C1=Function('C1',[x,z,u,d],[jacobian(gn,vertcat(x,v))])
    D1=Function('D1',[x,z,u,d],[jacobian(gn,z)])

    Alow1=[core.mtimes(-NP.linalg.inv(D1(xss,zss,uss,4)),C1(xss,zss,uss,4),A1(xss,zss,uss,4))]
    Alow2=[core.mtimes(-NP.linalg.inv(D1(xss,zss,uss,4)),C1(xss,zss,uss,4),B1(xss,zss,uss,4))]

    F = vertcat(horzcat(A1(xss,zss,uss,4), B1(xss,zss,uss,4)),horzcat(Alow1[0], Alow2[0])).full()

    gamma=vertcat(NP.eye(nx+nv),core.mtimes(linalg.inv(D1(xss,zss,uss,4)),C1(xss,zss,uss,4))).full()

 #   L = vertcat(NP.eye(nx+nv),core.mtimes(-linalg.inv(D1(xss,zss,uss,4)),C1(xss,zss,uss,4))).full()
    
    Hk = NP.array([(0, 0, 1, 0, 1 ,0 ,0, 0, 0, 0),
                  (0, 0, 0, 1, 0, 0, 0 ,0, 0, 0),
                   (0, 0, 0, 0, 0, 1, 0 ,1, 0, 0)])
    
    #K = core.mtimes(Pk_1,H.T,NP.linalg.inv(Rk))

    Diff_cov=core.mtimes(F,P)+core.mtimes(P,F.T)+core.mtimes(gamma,Qk,gamma.T)
    EKF_eqtn = {'x':reshape(P,100,1), 'ode':reshape(Diff_cov,100,1)}
    opts = {'tf':delta}

    EKF_I = integrator('EKF_I', 'cvodes', EKF_eqtn, opts)
   
    Covar = EKF_I(x0=reshape(Pk_1,100,1))
    Pk_k_1 = reshape(NP.array(Covar['xf'].full()),10,10)
    
    Sk = core.mtimes(Hk,Pk_k_1,Hk.T)+Rk
    
    Kk=core.mtimes(Pk_k_1,Hk.T,NP.linalg.inv(Sk))

    Pk=core.mtimes(NP.eye(nx+nz+nv)-core.mtimes(Kk,Hk),Pk_k_1)

    aug_k=vertcat(xss,dis,zss)+core.mtimes(Kk,y_hat_k)
    
    xss=aug_k[0:nx].full().flatten().tolist()
    dis=aug_k[nx:nx+nv].full().flatten()
    
    #print(NP.linalg.matrix_rank(control.ctrb(theta.T,Hk.T)))
    #print(NP.linalg.eigvals(theta-core.mtimes(Kk,Hk)))    
    return [xss,dis,zss,Pk]
    
def pole_place(x,z,u,d,v,xdot,gn,y_hat_k,Pk_1,Qk,Rk,xss,dis,zss,uss,delta):
    nx=int(x.dim()[0])
    nz=int(z.dim()[0])
    nu=int(u.dim()[0])
    nv=int(v.dim()[0])

    A1=Function('A1',[x,z,u,d],[jacobian(vertcat(xdot,v),vertcat(x,v))])
    B1=Function('B1',[x,z,u,d],[jacobian(vertcat(xdot,v),z)])
    C1=Function('C1',[x,z,u,d],[jacobian(gn,vertcat(x,v))])
    D1=Function('D1',[x,z,u,d],[jacobian(gn,z)])

    Alow1=[core.mtimes(-NP.linalg.inv(D1(xss,zss,uss,4)),C1(xss,zss,uss,4),A1(xss,zss,uss,4))]
    Alow2=[core.mtimes(-NP.linalg.inv(D1(xss,zss,uss,4)),C1(xss,zss,uss,4),B1(xss,zss,uss,4))]

    Aaug=vertcat(horzcat(A1(xss,zss,uss,4), B1(xss,zss,uss,4)),horzcat(Alow1[0], Alow2[0]))

    theta=linalg.expm(Aaug.full()*delta)
    gamma=vertcat(NP.eye(nx+nv),core.mtimes(linalg.inv(D1(xss,zss,uss,4)),C1(xss,zss,uss,4))).full()
    
    Hk=NP.array([(0, 0, 1, 0, 1 ,0 ,0, 0, 0, 0),
                 (0, 0, 0, 1, 0, 0, 0 ,0, 0, 0),
                 (0, 0, 0, 0, 0, 1, 0 ,1, 0, 0)])
    
    Pk_k_1=core.mtimes(theta,Pk_1,theta.T)+core.mtimes(gamma,Qk,gamma.T)
    
    Sk=core.mtimes(Hk,Pk_k_1,Hk.T)+Rk
    Kk=core.mtimes(Pk_k_1,Hk.T,NP.linalg.inv(Sk))
    
    aug_k=vertcat(xss,dis,zss)+core.mtimes(Kk,y_hat_k)
    Pk=core.mtimes(NP.eye(nx+nz+nv)-core.mtimes(Kk,Hk),Pk_k_1)
    
    xss=aug_k[0:nx].full().flatten().tolist()
    dis=aug_k[nx:nx+nv].full().flatten()
    zss=aug_k[nx+nv:].full().flatten().tolist()
    
    print(NP.linalg.eigvals(theta-core.mtimes(Kk,Hk)))
    print(theta)
    Kl=signal.place_poles(theta.T,Hk.T,NP.array([0.1, 0.3, 0.15, 0.22, 0.3, 0.36, 0.45, 0.5, 0.6, 0.7]))
    
    return Kl

def manual_obs(Lk,x,z,u,d,v,xdot,gn,y_hat_k,Pk_1,Qk,Rk,xss,dis,zss,uss,delta):
    
    Kk=Lk
    
    aug_k=vertcat(xss,dis,zss)+core.mtimes(Kk,y_hat_k)
    Pk=core.mtimes(NP.eye(nx+nz+nv)-core.mtimes(Kk,Hk),Pk_k_1)
    
    xss=aug_k[0:nx].full().flatten().tolist()
    dis=aug_k[nx:nx+nv].full().flatten()
    zss=aug_k[nx+nv:].full().flatten().tolist()
    
    return [xss,dis,zss,Pk]
