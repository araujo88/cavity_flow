# 2D Cavity-flow - explicit finite difference scheme
# Author: Leonardo Antonio de Araujo
# E-mail: leonardo.aa88@gmail.com
# Date: 20/05/2020

import numpy as np
import matplotlib.pyplot as plt

############################### Functions ####################################

def plot_contour(X,Y,Z,t):
    plt.contourf(X, Y, Z[:,:,int(t)], 20, cmap='gist_rainbow_r')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar();
    plt.show()
    
def plot_quiver(X,Y,u,v,t):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.quiver(X,Y,u[:,:,t],v[:,:,t], cmap='gist_rainbow_r', alpha=0.8)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
#    ax.axis([-0.2, 2.3, -0.2, 2.3])
    ax.set_aspect('equal')
    plt.show()    

# Computes LHS of poisson equation for pressure
def lhs(u,v,nx,ny,dx,dy):
    dudx = ddx(u,nx,ny,dx)
    dudy = ddy(u,nx,ny,dy)
    dvdx = ddx(v,nx,ny,dx)      
    dvdy = ddy(v,nx,ny,dy)
    return dudx**2+dvdy**2+2*dudy*dvdx

# Solves poisson equation
def poisson(f,nx,ny,dx,dy):
    p = np.zeros((nx,ny))
    for i in range(1,(nx-1)):
        for j in range (1,(ny-1)):
            p[i,j]=((dy**2)*(p[i+1,j]+p[i-1,j])+(dx**2)*(p[i,j+1]+p[i,j-1])-(dx**2)*(dy**2)*f[i,j])/(2*(dx**2+dy**2))
    return p

# Computes vorticity
def vort(u,v,nx,ny,dx,dy):
    dudy = ddy(u,nx,ny,dy)
    dvdx = ddx(v,nx,ny,dx)
    return dvdx-dudy

# Boundary conditions set-up
def boundary_conditions(T,T1,T2,T3,T4,nx,ny):
    for i in range(0,nx):
        T[i,0]=T1
        T[i,ny-1]=T2
    for j in range(0,ny):
        	T[0,j]=T3
        	T[nx-1,j]=T4

# Computes first derivative (x)
def ddx(T,nx,ny,dx):
    ddx = np.zeros((nx,ny))
    for i in range(1,(nx-1)):
        for j in range (1,(ny-1)):
            ddx[i,j]=(T[i+1,j]-T[i-1,j])/(2*dx) # ddx    
    return ddx

# Computes first derivative (y)
def ddy(T,nx,ny,dy):
    ddy = np.zeros((nx,ny))
    for i in range(1,(nx-1)):
        for j in range (1,(ny-1)):                         
            ddy[i,j]=(T[i,j+1]-T[i,j-1])/(2*dy) # ddy
    return ddy

# Computes second derivative (x)
def d2dx2(T,nx,ny,dx):
    d2dx2 = np.zeros((nx,ny))
    for i in range(1,(nx-1)):
        for j in range (1,(ny-1)):
            d2dx2[i,j]=(T[i+1,j]-2*T[i,j]+T[i-1,j])/dx**2 # d2dx2    
    return d2dx2
            
# Computes second derivative (y)
def d2dy2(T,nx,ny,dy):
    d2dy2 = np.zeros((nx,ny))
    for i in range(1,(nx-1)):
        for j in range (1,(ny-1)):                         
            d2dy2[i,j]=(T[i,j+1]-2*T[i,j]+T[i,j-1])/dy**2 # d2dy2
    return d2dy2

##############################################################################

# Physical parameters
nu = 0.01 # kinematic viscosity
Lx = 1 # length
Ly = 1 # width

# Numerical parameters
nx = 20 # number of points in x direction
ny = 20 # number of points in y direction
dt = 0.005 # time step
tf = 5 # final time
max_co = 1 # max Courant number

# Boundary conditions (Dirichlet)
u0=0; # internal field for u
v0=0; # internal field for v

u1=0; # bottom boundary condition
u2=1; # top boundary condition
u3=0; # right boundary condition
u4=0; # left boundary condition

v1=0;
v2=0;
v3=0;
v4=0;

# Generate 2D mesh
x = np.linspace(0, Lx, nx, endpoint=True)
y = np.linspace(0, Ly, ny, endpoint=True)
X, Y = np.meshgrid(x, y,indexing='ij')

# Computes cell length
dx = x[1]-x[0];
dy = y[1]-y[0];

# Maximum number of iterations
it_max = int(tf/dt)-1

# Courant numbers
r1 = u1*dt/(dx);
r2 = u1*dt/(dy);

if (r1 > max_co or r2 > max_co):
	raise TypeError('Unstable Solution!')

# Initialize variables
u = np.zeros((nx,ny,int(tf/dt))) # x-velocity
v = np.zeros((nx,ny,int(tf/dt))) # y-velocity
w = np.zeros((nx,ny,int(tf/dt))) # vorticity
psi = np.zeros((nx,ny,int(tf/dt))) # stream-function
p = np.zeros((nx,ny,int(tf/dt))) # pressure

# Initial condition
for i in range(0,nx-1):
    for j in range(1,ny-1):
        u[i,j,0] = u0
        v[i,j,0] = v0

# Boundary conditions set-up
boundary_conditions(u[:,:,:],u1,u2,u3,u4,nx,ny)
boundary_conditions(v[:,:,:],v1,v2,v3,v4,nx,ny)
w[:,:,0] = vort(u[:,:,0],v[:,:,0],nx,ny,dx,dy)
psi[:,:,0] = poisson(-w[:,:,0],nx,ny,dx,dy)

# Main time-loop
for t in range (0,it_max):	
 
       # Computes derivatives	
        dwdx=ddx(w[1:-1,1:-1,t],nx-2,ny-2,dx)
        dwdy=ddy(w[1:-1,1:-1,t],nx-2,ny-2,dy)
        d2wdx2=d2dx2(w[1:-1,1:-1,t],nx-2,ny-2,dx)
        d2wdy2=d2dy2(w[1:-1,1:-1,t],nx-2,ny-2,dy)
        
        # Computes vorticity
        w[1:-1,1:-1,t+1]=(-u[1:-1,1:-1,t]*dwdx-v[1:-1,1:-1,t]*dwdy+nu*(d2wdx2+d2wdy2))*dt+w[1:-1,1:-1,t]
        
        # Solves poisson equation for stream function
        psi[:,:,t+1] = poisson(-w[:,:,t+1],nx,ny,dx,dy)
        
        # Computes velocities
        dpsidx=ddx(psi[:,:,t+1],nx,ny,dx)
        dpsidy=ddy(psi[:,:,t+1],nx,ny,dy)
        u[1:-1,1:-1,t+1]=dpsidy[1:-1,1:-1]
        v[1:-1,1:-1,t+1]=-dpsidx[1:-1,1:-1]

        # Computes pressure
        dudx = ddx(u[:,:,t+1],nx,ny,dx)
        dudy = ddy(u[:,:,t+1],nx,ny,dy)
        dvdx = ddx(v[:,:,t+1],nx,ny,dx)      
        dvdy = ddy(v[:,:,t+1],nx,ny,dy)
        f = dudx**2+dvdy**2+2*dudy*dvdx
        p[:,:,t+1] = poisson(-f,nx,ny,dx,dy)
            
#        fig, ax = plt.subplots(figsize=(7,7))
#        ax.quiver(X,Y,u[:,:,t],v[:,:,t], cmap='gist_rainbow_r', alpha=0.8)
#        ax.xaxis.set_ticks([])
#        ax.yaxis.set_ticks([])
#        ax.set_aspect('equal')

        fig, ax = plt.subplots(figsize=(7,7))
        plt.contourf(X, Y, psi[:,:,int(t)], 20, cmap='gist_rainbow_r')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar();

        plt.savefig('figure-' + str(t) + '.png')
        plt.close()

        

#plot_contour(X,Y,psi,it_max)
#plot_quiver(X,Y,u,v,it_max)
#plot_contour(X,Y,p,it_max)
#plot_contour(X,Y,u,it_max)
#plot_contour(X,Y,v,it_max)
