# 2D lid-driven cavity-flow
# Author: Leonardo Antonio de Araujo
# E-mail: leonardo.aa88@gmail.com
# Date: 21/05/2020

import numpy as np
import matplotlib.pyplot as plt
from derivatives import Diff1, Diff2
from scipy import sparse
from scipy.sparse.linalg import inv
from fft_poisson import fft_poisson

############################### Functions ####################################

def plot_contour(X,Y,Z,t):
    plt.contourf(X, Y, Z[:,:,int(t)], 40, cmap='gist_rainbow_r')
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
    ax.set_aspect('equal')
    plt.show()

##############################################################################

# Physical parameters
Re = 100 # Reynolds number
Lx = 1 # length
Ly = 1 # width

# Numerical parameters
nx = 20 # number of points in x direction
ny = 20 # number of points in y direction
dt = 0.02 # time step
tf = 10 # final time
max_co = 1 # max Courant number
order = 2 # finite difference order for spatial derivatives

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

# Generates derivatives operators

d_x = Diff1(nx,order)/dx
d_y = Diff1(ny,order)/dy
d_x2 = Diff2(nx, order)/dx**2
d_y2 = Diff2(ny, order)/dy**2

I = np.eye(nx,ny) # identity matrix
DX = sparse.kron(d_x,I) # kronecker product
DY = sparse.kron(I,d_y)
DX2 = sparse.kron(d_x2,I)
DY2 = sparse.kron(I,d_y2)

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
dwdx = np.zeros((nx,ny,int(tf/dt)))
dwdy = np.zeros((nx,ny,int(tf/dt)))
d2wdx2 = np.zeros((nx,ny,int(tf/dt)))
d2wdy2 = np.zeros((nx,ny,int(tf/dt)))
dpsidx = np.zeros((nx,ny,int(tf/dt)))
dpsidy = np.zeros((nx,ny,int(tf/dt)))
dudx = np.zeros((nx,ny,int(tf/dt)))
dudy = np.zeros((nx,ny,int(tf/dt)))
dvdx = np.zeros((nx,ny,int(tf/dt))) 
dvdy = np.zeros((nx,ny,int(tf/dt)))

# Initial condition
for i in range(0,nx-1):
    for j in range(1,ny-1):
        u[i,j,0] = u0
        v[i,j,0] = v0

#dx2_dy2 = inv(DX2+DY2)
#psi[:,:,0] = np.reshape(dx2_dy2 @ np.reshape(-w[:,:,0],(nx*ny,1)),(nx,ny))
#plot_contour(X,Y,psi,0)

# Main time-loop
for t in range (0,it_max):
	# Boundary conditions
	for j in range(0,ny):
		u[0,j,t]=u3
		u[nx-1,j,t]=u4
		v[0,j,t]=v3
		v[nx-1,j,t]=v4
	for i in range(0,nx):
		u[i,0,t]=u1
		u[i,ny-1,t]=u2
		v[i,0,t]=v1
		v[i,ny-1,t]=v2
	dudy[:,:,t] = np.reshape(DY @ np.reshape(u[:,:,t],(nx*ny,1)),(nx,ny))
	dvdx[:,:,t] = np.reshape(DX @ np.reshape(v[:,:,t],(nx*ny,1)),(nx,ny))
	for j in range(0,ny):
		w[0,j,t]=dvdx[0,j,t]-dudy[0,j,t]
		w[nx-1,j,t]=dvdx[nx-1,j,t]-dudy[nx-1,j,t]
	for i in range(0,nx):
		w[i,0,t]=dvdx[i,0,t]-dudy[i,0,t]
		w[i,ny-1,t]=dvdx[i,ny-1,t]-dudy[i,ny-1,t]
	psi[:,:,t] = fft_poisson(-w[:,:,t],dx)

       # Computes derivatives	
	dwdx[:,:,t]=np.reshape(DX @ np.reshape(w[:,:,t],(nx*ny,1)),(nx,ny))
	dwdy[:,:,t]=np.reshape(DY @ np.reshape(w[:,:,t],(nx*ny,1)),(nx,ny))
	d2wdx2[:,:,t]=np.reshape(DX2 @ np.reshape(w[:,:,t],(nx*ny,1)),(nx,ny))
	d2wdy2[:,:,t]=np.reshape(DY2 @ np.reshape(w[:,:,t],(nx*ny,1)),(nx,ny))
        
        # Computes vorticity
	w[:,:,t+1]=(-u[:,:,t]*dwdx[:,:,t]-v[:,:,t]*dwdy[:,:,t]+(1/Re)*(d2wdx2[:,:,t]+d2wdy2[:,:,t]))*dt+w[:,:,t]

        # Solves poisson equation for stream function
	psi[:,:,t+1] = fft_poisson(-w[:,:,t+1],dx)
        
        # Computes velocities
	dpsidx[:,:,t+1] = np.reshape(DX @ np.reshape(psi[:,:,t+1],(nx*ny,1)),(nx,ny))
	dpsidy[:,:,t+1] = np.reshape(DY @ np.reshape(psi[:,:,t+1],(nx*ny,1)),(nx,ny))
	u[:,:,t+1] = dpsidy[:,:,t+1]
	v[:,:,t+1] = -dpsidx[:,:,t+1]

	# Checks continuity equation
	dudx[:,:,t+1] = np.reshape(DX @ np.reshape(u[:,:,t+1],(nx*ny,1)),(nx,ny))
	dvdy[:,:,t+1] = np.reshape(DY @ np.reshape(v[:,:,t+1],(nx*ny,1)),(nx,ny))
	continuity = dudx[:,:,t+1]+dvdy[:,:,t+1]
	print('Iteration: ' + str(t))
	print('Continuity max: ' + str(continuity.max()) + ' Continuity min: ' + str(continuity.min()))
	
        # Computes pressure
#	dudx = np.reshape(DX @ np.reshape(u[:,:,t+1],(nx*ny,1)),(nx,ny))
#	dudy = np.reshape(DY @ np.reshape(u[:,:,t+1],(nx*ny,1)),(nx,ny))
#	dvdx = np.reshape(DX @ np.reshape(v[:,:,t+1],(nx*ny,1)),(nx,ny))
#	dvdy = np.reshape(DY @ np.reshape(v[:,:,t+1],(nx*ny,1)),(nx,ny))
#	f = dudx**2+dvdy**2+2*dudy*dvdx
#	p[:,:,t+1] = fft_poisson(-f,dx)
            
#        fig, ax = plt.subplots(figsize=(7,7))
#        ax.quiver(X,Y,u[:,:,t],v[:,:,t], cmap='gist_rainbow_r', alpha=0.8)
#        ax.xaxis.set_ticks([])
#        ax.yaxis.set_ticks([])
#        ax.set_aspect('equal')

	fig, ax = plt.subplots(figsize=(7,7))
	plt.contourf(X, Y, psi[:,:,t], 40, cmap='gist_rainbow_r')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.colorbar();

	plt.title('Stream function - Re = ' + str(Re) + ' t = {:.2f}'.format((t)*dt))
	plt.savefig('figure-' + str(t) + '.png')
	plt.close()
        
#plot_contour(X,Y,w,it_max)
#plot_contour(X,Y,psi,it_max)
#plot_quiver(X,Y,u,v,it_max)
#plot_contour(X,Y,p,it_max)
#plot_contour(X,Y,u,it_max)
#plot_contour(X,Y,v,it_max)
