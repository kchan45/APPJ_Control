
# Imports.
from casadi import *
import casadi.tools as ctools
import numpy as NP
from numpy import random
import scipy.linalg
import matplotlib.pyplot as plt
from functools import reduce

def getCasadiFunc(f, varsizes, varnames=None, funcname="f", rk4=False,
                  Delta=1, M=1, scalar=True, casaditype=None):
    """
    Takes a function handle and turns it into a Casadi function.

    f should be defined to take a specified number of arguments and return a
    scalar, list, or numpy array. varnames, if specified, gives names to each
    of the inputs, but this is not required.

    sizes should be a list of how many elements are in each one of the inputs.

    This version is more general because it lets you specify arbitrary
    arguments, but you have to make sure you do everything properly.
    """
    # Pass the buck to the sub function.
    symbols = __getCasadiFunc(f, varsizes, varnames, funcname, scalar,
                              casaditype, allowmatrix=True)
    args = symbols["args"]
    fexpr = symbols["fexpr"]

    # Evaluate function and make a Casadi object.
    fcasadi = casadi.Function(funcname, args, [fexpr])

    # Wrap with rk4 if requested.
    if rk4:
        frk4 = util.rk4(fcasadi, args[0], args[1:], Delta, M)
        fcasadi = casadi.Function(funcname, args, [frk4])

    return fcasadi


def __getCasadiFunc(f, varsizes, varnames=None, funcname="f", scalar=True,
                    casaditype=None, allowmatrix=True):
    """
    Core logic for getCasadiFunc and its relatives.

    Returns a dictionary with entries fexpr, rawargs, args, XX, names, sizes:
    - rawargs is the list of raw arguments, each a numpy array of Casadi
      scalars if scalar=True, or a single Casadi symbolic matrix if
      scalar=False.
    - args is the same list, but with all arguments converted to a single
      Casadi symbolic matrix.
    - fexpr is the casadi expression resulting from evaluating f(*rawargs).
    - XX is either casadi.SX or casadi.MX depending on what was used to create
      rawargs and args.
    - names is a list of string names for each argument.
    - sizes is a list of one- or two-element lists giving the sizes.
    """
    # Check names.
    if varnames is None:
        varnames = ["x%d" % (i,) for i in range(len(varsizes))]
    else:
        varnames = [str(n) for n in varnames]
    if len(varsizes) != len(varnames):
        raise ValueError("varnames must be the same length as varsizes!")

    # Loop through varsizes in case some may be matrices.
    realvarsizes = []
    for s in varsizes:
        goodInput = True
        try:
            s = [int(s)]
        except TypeError:
            if allowmatrix:
                try:
                    s = list(s)
                    goodInput = len(s) <= 2
                except TypeError:
                    goodInput = False
            else:
                raise TypeError("Entries of varsizes must be integers!")
        if not goodInput:
            raise TypeError("Entries of varsizes must be integers or "
                "two-element lists!")
        realvarsizes.append(s)

    # Decide which Casadi type to use. XX is either casadi.SX or casadi.MX.
    if casaditype is None:
        casaditype = "SX" if scalar else "MX"
    XX = dict(SX=casadi.SX, MX=casadi.MX).get(casaditype, None)
    if XX is None:
        raise ValueError("casaditype must be either 'SX' or 'MX'!")

    # Now make the symbolic variables. How they are packaged depends on the
    # scalar option.
    if scalar:
        args = []
        for (name, size) in zip(varnames, realvarsizes):
            if len(size) == 2:
                thisarr = []
                for i in xrange(size[0]):
                    row = [XX.sym("%s_%d_%d" % (name, i, j)) for
                           j in xrange(size[1])]
                    thisarr.append(row)
            else:
                thisarr = [XX.sym("%s_%d" % (name, i)) for
                           i in xrange(size[0])]
            args.append(np.array(thisarr, dtype=object))
    else:
        args = [XX.sym(name, *size) for
                (name, size) in zip(varnames, realvarsizes)]
    catargs = [XX(a) for a in args]

    # Evaluate the function and return everything.
    fexpr = safevertcat(f(*args))
    return dict(fexpr=fexpr, args=catargs, rawargs=args, XX=XX, names=varnames,
                sizes=realvarsizes)


def mtimes(*args, **kwargs):
    """
    More flexible version casadi.tools.mtimes.

    Matrix multiplies all of the given arguments and returns the result. If any
    inputs are Casadi's SX or MX data types, uses Casadi's mtimes. Otherwise,
    uses a sequence of np.dot operations.

    Keyword arguments forcedot or forcemtimes can be set to True to pick one
    behavior or another.
    """
    # Get keyword arguments.
    forcemtimes = kwargs.pop("forcemtimes", None)
    forcedot = kwargs.pop("forcedot", False)
    if len(kwargs) > 0:
        raise TypeError("Invalid keywords: %s" % kwargs.keys())

    # Pick whether to use mul or dot.
    if forcemtimes:
        if forcedot:
            raise ValueError("forcemtimes and forcedot can't both be True!")
        useMul = True
    elif forcedot:
        useMul = False
    else:
        useMul = False
        symtypes = set(["SX", "MX"])
        for a in args:
            atype = getattr(a, "type_name", lambda : None)()
            if atype in symtypes:
                useMul = True
                break

    # Now actually do multiplication.
    ans = ctools.mtimes(args) if useMul else reduce(np.dot, args)
    return ans


def safevertcat(x):
    """
    Safer wrapper for Casadi's vertcat.

    the input x is expected to be an iterable containing multiple things that
    should be concatenated together. This is in contrast to Casadi 3.0's new
    version of vertcat that accepts a variable number of arguments. We retain
    this (old, Casadi 2.4) behavior because it makes it easier to check types.

    If a single SX or MX object is passed, then this doesn't do anything.
    Otherwise, if all elements are numpy ndarrays, then numpy's concatenate
    is called. If anything isn't an array, then casadi.vertcat is called.
    """
    symtypes = set(["SX", "MX"])
    xtype = getattr(x, "type_name", lambda : None)()
    if xtype in symtypes:
        val = x
    elif (not isinstance(x, np.ndarray) and
            all(isinstance(a, np.ndarray) for a in x)):
        val = np.concatenate(x)
    else:
        val = casadi.vertcat(*x)
    return val


def flattenlist(l,depth=1):
    """
    Flattens a nested list of lists of the given depth.

    E.g. flattenlist([[1,2,3],[4,5],[6]]) returns [1,2,3,4,5,6]. Note that
    all sublists must have the same depth.
    """
    for i in range(depth):
        l = list(itertools.chain.from_iterable(l))
    return l


def dlqr(A,B,Q,R,M=None):
    """
    Get the discrete-time LQR for the given system.

    Stage costs are

        x'Qx + 2*x'Mu + u'Qu

    with M = 0 if not provided.
    """
    # For M != 0, we can simply redefine A and Q to give a problem with M = 0.
    if M is not None:
        RinvMT = scipy.linalg.solve(R,M.T)
        Atilde = A - B.dot(RinvMT)
        Qtilde = Q - M.dot(RinvMT)
    else:
        Atilde = A
        Qtilde = Q
        M = np.zeros(B.shape)
    Pi = scipy.linalg.solve_discrete_are(Atilde,B,Qtilde,R)
    K = -scipy.linalg.solve(B.T.dot(Pi).dot(B) + R, B.T.dot(Pi).dot(A) + M.T)

    return [K, Pi]


def dlqe(A,C,Q,R):
    """
    Get the discrete-time Kalman filter for the given system.
    """
    P = scipy.linalg.solve_discrete_are(A.T,C.T,Q,R)
    L = scipy.linalg.solve(C.dot(P).dot(C.T) + R, C.dot(P)).T

    return [L, P]


def c2d(A, B, Delta, Bp=None, f=None, asdict=False):
    """
    Discretizes affine system (A, B, Bp, f) with timestep Delta.

    This includes disturbances and a potentially nonzero steady-state, although
    Bp and f can be omitted if they are not present.

    If asdict=True, return value will be a dictionary with entries A, B, Bp,
    and f. Otherwise, the return value will be a 4-element list [A, B, Bp, f]
    if Bp and f are provided, otherwise a 2-element list [A, B].
    """
    n = A.shape[0]
    I = np.eye(n)
    D = scipy.linalg.expm(Delta*np.vstack((np.hstack([A, I]),
                                     np.zeros((n, 2*n)))))
    Ad = D[:n,:n]
    Id = D[:n,n:]
    Bd = Id.dot(B)
    Bpd = None if Bp is None else Id.dot(Bp)
    fd = None if f is None else Id.dot(f)

    if asdict:
        retval = dict(A=Ad, B=Bd, Bp=Bpd, f=fd)
    elif Bp is None and f is None:
        retval = [Ad, Bd]
    else:
        retval = [Ad, Bd, Bpd]
    return retval


def getLinearizedModel(f,args,names=None,Delta=None,returnf=True,forcef=False):
    """
    Returns linear (affine) state-space model for f at the point in args.

    Note that f must be a casadi function (e.g., the output of getCasadiFunc).

    names should be a list of strings to specify the dictionary entry for each
    element. E.g., for args = [xs, us] to linearize a model in (x,u), you
    might choose names = ["A", "B"]. These entries can then be accessed from
    the returned dictionary to get the linearized state-space model.

    If "f" is not in the list of names, then the return dict will also include
    an "f" entry with the actual value of f at the linearization point. To
    disable this, set returnf=False.
    """
    # Decide names.
    if names is None:
        names = ["A"] + ["B_%d" % (i,) for i in range(1,len(args))]

    # Evaluate function.
    fs = np.array(f(*args))

    # Now do jacobian.
    jacobians = []
    for i in range(len(args)):
        jac = f.jacobian(i,0) # df/d(args[i]).
        jacobians.append(np.array(jac(*args)[0]))

    # Decide whether or not to discretize.
    if Delta is not None:
        (A, Bfactor) = c2d(jacobians[0],np.eye(jacobians[0].shape[0]),Delta)
        jacobians = [A] + [Bfactor.dot(j) for j in jacobians[1:]]
        fs = Bfactor.dot(fs)

    # Package everything up.
    ss = dict(zip(names,jacobians))
    if returnf and ("f" not in ss or forcef):
        ss["f"] = fs
    return ss


def rootFinder(g, N, args=None, x0=None, solver_type='ipopt', print_level=5, lbx=None, ubx=None):
    """
    Finds a root of a function g(x)=0 by solving NLP using initial guess x0.
    """
    x = MX.sym('x',N)
    if x0 is None:
        x0 = NP.zeros(N)
    if lbx is None:
        lbx = -inf*NP.ones(N)
    if ubx is None:
        ubx = inf*NP.ones(N)
    if lbx is float:
        lbx = lbx*NP.ones(N)
    if ubx is float:
        ubx = ubx*NP.ones(N)
    geval = g(x,*args)
    # define NLP with only constraints
    nlp = {'x':x, 'f':0, 'g':geval}
    # set options for NLP solver
    opts = {'ipopt.print_level':print_level}
    # build NLP solver "consistentIC"
    solver = nlpsol('solver', solver_type, nlp, opts)
    # solve NLP with some initial guess and ensuring g=0
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=NP.zeros(N), ubg=NP.zeros(N))
    # extract and return solution
    xroot = NP.squeeze(sol['x'])
    return xroot.tolist()


def linearMPC(MPC_dynamics, nx, nu, N, uss, xss, u_lb, u_ub, x_lb, x_ub, MySolver):
    """
    Builds instance of linear MPC problem (in deviation variables).
    Returns solver, initial conditions, and constraints on the decision variables.
    [Currently only supports bounds on inputs and states]

    MPC_dynamics must be a casadi function that returns 2 arguments:
    the next step dynamics and stage cost.
    """
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
    X0 = MX.sym('X0', nx)
    q += [X0]
    lbq += [0]*nx
    ubq += [0]*nx
    q0 += [0]*nx
    # formulate the QP
    Xk = X0
    for k in range(N):
        # new NLP variable for the control
        Uk = MX.sym('U_' + str(k), nu)
        q   += [Uk]
        lbq += [u_lb[i]-uss[i] for i in range(nu)]
        ubq += [u_ub[i]-uss[i] for i in range(nu)]
        q0  += [0]*nu
        # next step dynamics and stage cost
        Fk, Lk = MPC_dynamics(Xk,Uk)
        Xk_end = Fk
        J = J + Lk
        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), nx)
        q   += [Xk]
        lbq += [x_lb[i]-xss[i] for i in range(nx)]
        ubq += [x_ub[i]-xss[i] for i in range(nx)]
        q0  += [0]*nx
        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0]*nx
        ubg += [0]*nx
    # set solver options
    opts = {}
    if MySolver == "sqpmethod":
      opts["qpsol"] = "qpoases"
      opts["qpsol_options"] = {"printLevel":"none"}
    # create NLP solver for MPC problem
    prob = {'f':J, 'x':vertcat(*q), 'g':vertcat(*g)}
    solver = nlpsol('solver', MySolver, prob, opts)
    MPC = {}
    MPC["solver"] = solver
    MPC["lbq"] = lbq
    MPC["ubq"] = ubq
    MPC["q0"] = q0
    MPC["lbg"] = lbg
    MPC["ubg"] = ubg
    return MPC
