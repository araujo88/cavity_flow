from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft,fft2,ifft2,ifft,irfft2,rfft2
import random as random
from mpl_toolkits.mplot3d import Axes3D

def dst(x,axis=-1):
    """Discrete Sine Transform (DST-I)

    Implemented using 2(N+1)-point FFT
    xsym = r_[0,x,0,-x[::-1]]
    DST = (-imag(fft(xsym))/2)[1:(N+1)]

    adjusted to work over an arbitrary axis for entire n-dim array
    """
    n = len(x.shape)
    N = x.shape[axis]
    slices = [None]*3
    for k in range(3):
        slices[k] = []
        for j in range(n):
            slices[k].append(slice(None))
    newshape = list(x.shape)
    newshape[axis] = 2*(N+1)
    xsym = np.zeros(newshape,np.float)
    slices[0][axis] = slice(1,N+1)
    slices[1][axis] = slice(N+2,None)
    slices[2][axis] = slice(None,None,-1)
    for k in range(3):
        slices[k] = tuple(slices[k])
    xsym[slices[0]] = x
    xsym[slices[1]] = -x[slices[2]]
    DST = fft(xsym,axis=axis)
    #print xtilde
    return (-(DST.imag)/2)[slices[0]]

def dst2(x,axes=(-1,-2)):
    return dst(dst(x,axis=axes[0]),axis=axes[1])

def idst2(x,axes=(-1,-2)):
    return dst(dst(x,axis=axes[0]),axis=axes[1])

def fft_poisson(f,h):

	m,n=f.shape
	f_bar=np.zeros([n,n])
	u_bar = f_bar			# make of the same shape
	u = u_bar

	f_bar=idst2(f)			# f_bar= fourier transform of f

	f_bar = f_bar * (2/n+1)**2  #Normalize
	#u_bar =np.zeros([n,n])
	pi=np.pi
	lam = np.arange(1,n+1)
	lam = -4/h**2 * (np.sin((lam*pi) / (2*(n + 1))))**2 #$compute $\lambda_x$
	#for rectangular domain add $lambda_y$
	for i in range(0,n):
		for j in range(0,n):
			u_bar[i,j] = (f_bar[i,j]) / (lam[i] + lam[j])

	u=dst2(u_bar)				#sine transform back
	u= u * (2/(n+1))**2			#normalize ,change for rectangular domain
	return u
