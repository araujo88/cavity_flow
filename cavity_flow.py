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
order = 4 # finite difference order for spatial derivatives

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
DX = sparse.kron(d_x,I) # kronecker product for sparse matrix
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
u = np.zeros((nx,ny)) # x-velocity
v = np.zeros((nx,ny)) # y-velocity
w = np.zeros((nx,ny)) # vorticity
psi = np.zeros((nx,ny)) # stream-function
p = np.zeros((nx,ny)) # pressure
dwdx = np.zeros((nx,ny))
dwdy = np.zeros((nx,ny))
d2wdx2 = np.zeros((nx,ny))
d2wdy2 = np.zeros((nx,ny))
dpsidx = np.zeros((nx,ny))
dpsidy = np.zeros((nx,ny))
dudx = np.zeros((nx,ny))
dudy = np.zeros((nx,ny))
dvdx = np.zeros((nx,ny)) 
dvdy = np.zeros((nx,ny))

# Initial condition
for i in range(0,nx-1):
    for j in range(1,ny-1):
        u[i,j] = u0
        v[i,j] = v0

#dx2_dy2 = inv(DX2+DY2)
#psi[:,:,0] = np.reshape(dx2_dy2 @ np.reshape(-w[:,:,0],(nx*ny,1)),(nx,ny))
#plot_contour(X,Y,psi,0)

# Main time-loop
for t in range (0,it_max):
	# Boundary conditions
	for j in range(0,ny):
		u[0,j]=u3
		u[nx-1,j]=u4
		v[0,j]=v3
		v[nx-1,j]=v4
	for i in range(0,nx):
		u[i,0]=u1
		u[i,ny-1]=u2
		v[i,0]=v1
		v[i,ny-1]=v2
	dudy = np.reshape(DY @ np.reshape(u,(nx*ny,1)),(nx,ny))
	dvdx = np.reshape(DX @ np.reshape(v,(nx*ny,1)),(nx,ny))
	for j in range(0,ny):
		w[0,j]=dvdx[0,j]-dudy[0,j]
		w[nx-1,j]=dvdx[nx-1,j]-dudy[nx-1,j]
	for i in range(0,nx):
		w[i,0]=dvdx[i,0]-dudy[i,0]
		w[i,ny-1]=dvdx[i,ny-1]-dudy[i,ny-1]
	psi = fft_poisson(-w,dx)

       # Computes derivatives	
	dwdx = np.reshape(DX @ np.reshape(w,(nx*ny,1)),(nx,ny))
	dwdy = np.reshape(DY @ np.reshape(w,(nx*ny,1)),(nx,ny))
	d2wdx2 = np.reshape(DX2 @ np.reshape(w,(nx*ny,1)),(nx,ny))
	d2wdy2 =np.reshape(DY2 @ np.reshape(w,(nx*ny,1)),(nx,ny))
        
        # Time-advancement (Euler)
	w=(-u*dwdx-v*dwdy+(1/Re)*(d2wdx2+d2wdy2))*dt+w

        # Solves poisson equation for stream function
	psi = fft_poisson(-w,dx)
        
        # Computes velocities
	dpsidx = np.reshape(DX @ np.reshape(psi,(nx*ny,1)),(nx,ny))
	dpsidy = np.reshape(DY @ np.reshape(psi,(nx*ny,1)),(nx,ny))
	u = dpsidy
	v = -dpsidx

	# Checks continuity equation
	dudx = np.reshape(DX @ np.reshape(u,(nx*ny,1)),(nx,ny))
	dvdy = np.reshape(DY @ np.reshape(v,(nx*ny,1)),(nx,ny))
	continuity = dudx+dvdy
	print('Iteration: ' + str(t))
	print('Continuity max: ' + str(continuity.max()) + ' Continuity min: ' + str(continuity.min()))
	
        # Computes pressure
#	dudx = np.reshape(DX @ np.reshape(u,(nx*ny,1)),(nx,ny))
#	dudy = np.reshape(DY @ np.reshape(u,(nx*ny,1)),(nx,ny))
#	dvdx = np.reshape(DX @ np.reshape(v,(nx*ny,1)),(nx,ny))
#	dvdy = np.reshape(DY @ np.reshape(v,(nx*ny,1)),(nx,ny))
#	f = dudx**2+dvdy**2+2*dudy*dvdx
#	p = fft_poisson(-f,dx)
            
#        fig, ax = plt.subplots(figsize=(7,7))
#        ax.quiver(X,Y,u,v, cmap='gist_rainbow_r', alpha=0.8)
#        ax.xaxis.set_ticks([])
#        ax.yaxis.set_ticks([])
#        ax.set_aspect('equal')

	fig, ax = plt.subplots(figsize=(7,7))
	plt.contourf(X, Y, psi, 40, cmap='gist_rainbow_r')
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
