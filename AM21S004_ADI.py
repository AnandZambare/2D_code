import matplotlib.pyplot as plt
from numpy import *
import time
# Name - Anand Sanjeev Zambare
# Roll.no - AM21S004
# Foundations of CFD Computer Assignment 2
# Solving elliptic equation for finding the steady state temperature distribution in 2-D plate
stat_time = time.time()
W = 0.4                # width of the plate along Y
L = 0.3                # length of the plate along X
Nx = 31                # no. of nodes in the x direction
Ny = 41               # no. of nodes in the y direction
dx = L/(Nx-1)          # grid size in x direction
dy = W/(Ny-1)          # grid size in y direction
beta = dx/dy           # numerical constant used in formulation
X = arange(0, L+dx, dx)         # X vector
Y = arange(0, W+dy, dy)         # Y vector
XX, YY = meshgrid(X, Y)
Errormax = 0.01
# initial condition in the domain
T = zeros((Ny, Nx))    # Initialize Temperature in 2-D domain
# apply temperature conditions at Y = 0 and Y = 0.4 which is 40 and 10 deg C
# define a function to implement Boundary Conditions.


def BC(T1):
    T1[:, 0] = 0
    T1[:, Nx-1] = 0
    T1[0, :] = 10
    T1[Ny-1, :] = 40
    return T1


def error(T1, T2):
    err = 0
    # T2 is the latest value and T1 is the one iteration before value
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            err = err + (abs(T2[i, j] - T1[i, j]))
    return err


def Solver_1(Q, R):                      # P is coefficient matrix. Q is unknown vector and R is RHS which is known.
    m1 = len(Q)
    x_s = zeros(m1)                    # solution vector.
    x_s[0] = Q[0]
    x_s[m1-1] = Q[m1-1]
    s_s = zeros(m1-3)
    r_s = zeros(m1-3)
    s_s[0] = (1/(2*(1+(beta**2)))) - (2*(1+(beta**2)))
    for i in range(1, m1-3):
        s_s[i] = (-2*(1+(beta**2)))-(1/s_s[0])
    r_s[0] = (R[0]/(2*(1+(beta**2)))) + R[1]
    for i in range(1, m1-3):
        r_s[i] = R[i+1] - (r_s[i-1]/s_s[i-1])
    x_s[m1-2] = r_s[m1-4]/s_s[m1-4]
    for i in range(m1-3, 1, -1):
        x_s[i] = (r_s[i-2]-x_s[i+1])/(s_s[i-2])
    x_s[1] = (R[0]-x_s[2])/(-2*(1+(beta**2)))
    return x_s


def RHS_1(T00, T11, T22):
    m1 = len(T00)            # or len(T22)
    B11 = zeros(m1-2)
    for i in range(1, m1-1):
        B11[i-1] = (-1*(beta**2))*(T00[i]+T22[i])
    B11[0] = B11[0]-T11[0]
    B11[m1-3] = B11[m1-3]-T11[m1-1]
    return B11


def RHS_2(T00, T11, T22):
    m1 = len(T00)
    B22 = zeros(m1-2)
    for i in range(1, m1-1):
        B22[i-1] = -1*(T00[i]+T11[i])
    B22[0] = -1*(T00[0]+T11[0])-((beta**2)*T22[0])
    B22[m1-3] = -1*(T00[m1-2]+T11[m1-2])-((beta**2)*T22[m1-1])
    return B22


def Solver_2(Q, R):
    m1 = len(Q)
    y_s = zeros(m1)  # solution vector.
    y_s[0] = Q[0]
    y_s[m1-1] = Q[m1-1]
    s_s = zeros(m1-3)
    r_s = zeros(m1-3)
    s_s[0] = ((beta**4)/(2*(1+(beta**2))))-(2*(1+(beta**2)))
    for i in range(1, m1-3):
        s_s[i] = (-2*(1+(beta**2)))-((beta**4)/s_s[0])
    r_s[0] = ((R[0]*(beta**2))/(2*(1 + (beta ** 2)))) + R[1]
    for i in range(1, m1-3):
        r_s[i] = R[i+1]-((beta**2)*r_s[i-1]/s_s[i-1])
    y_s[m1-2] = r_s[m1-4]/s_s[m1-4]
    for i in range(m1-3, 1, -1):
        y_s[i] = (r_s[i-2]-((beta**2)*y_s[i+1]))/(s_s[i-2])
    y_s[1] = (R[0] - ((beta**2)*y_s[2]))/(-2*(1+(beta**2)))
    return y_s


T = BC(T)
T_prev = T.copy()
k = 1
while k < Ny-1:
    R = RHS_1(T[k-1, :], T[k+1, :], T[k, :])
    T[k, :] = Solver_1(T[k, :], R)
    k = k+1
T = BC(T)
k = 1
while k < Nx-1:
    R = RHS_2(T[:, k-1], T[:, k+1], T[:, k])
    T[:, k] = Solver_2(T[:, k], R)
    k = k+1
T = BC(T)
Error = error(T_prev, T)
iter = 1
while Error > Errormax:
    T_prev = T.copy()
    k = 1
    while k < Ny - 1:
        R = RHS_1(T[k-1, :], T[k, :], T[k+1, :])
        T[k, :] = Solver_1(T[k, :], R)
        k = k + 1
    T = BC(T)
    k = 1
    while k < Nx - 1:
        R = RHS_2(T[:, k - 1], T[:, k + 1], T[:, k])
        T[:, k] = Solver_2(T[:, k], R)
        k = k + 1
    T = BC(T)
    Error = error(T_prev, T)
    iter = 1 + iter
print(iter, Error)
plt.figure(1)
plt.contourf(XX, YY, T, cmap='jet')
plt.colorbar()
plt.show()