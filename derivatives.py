import numpy as np
import matplotlib.pyplot as plt

# Computes finite-difference matrices for the first derivative
def Diff1(n, o = 2): # Default order is set to second-order
	D=np.zeros((n,n))
	if o == 2: # Second-order
		D[0,0]=-1
		D[0,1]=1
		for i in range(1,(n-1)):
			D[i,i-1]=-0.5
			D[i,i]=0
			D[i,i+1]=0.5
		D[-1,-1]=D[0,1]
		D[-1,-2]=D[0,0]
		return D
	elif o == 4: # Fourth-order
		D[0,0]=-1
		D[0,1]=1
		D[1,0]=-0.5
		D[1,1]=0
		D[1,2]=0.5
		for i in range(2,(n-2)):
			D[i,i-2]=1/12
			D[i,i-1]=-2/3
			D[i,i]=0
			D[i,i+1]=2/3
			D[i,i+2]=-1/12
		D[-1,-1]=D[0,1]
		D[-1,-2]=D[0,0]
		D[-2,-1]=D[1,2]
		D[-2,-2]=D[1,1]
		D[-2,-3]=D[1,0]
		return D
	elif o == 6: # Sixth-order
		D[0,0]=-1
		D[0,1]=1
		D[1,0]=-0.5
		D[1,1]=0
		D[1,2]=0.5
		D[2,0]=1/12
		D[2,1]=-2/3
		D[2,2]=0
		D[2,3]=2/3
		D[2,4]=-1/12
		for i in range(3,(n-3)):
			D[i,i-3]=-1/60			
			D[i,i-2]=3/20
			D[i,i-1]=-3/4
			D[i,i]=0
			D[i,i+1]=3/4
			D[i,i+2]=-3/20
			D[i,i+3]=1/60
		D[-1,-1]=D[0,1]
		D[-1,-2]=D[0,0]
		D[-2,-1]=D[1,2]
		D[-2,-2]=D[1,1]
		D[-2,-3]=D[1,0]
		D[-3,-1]=D[2,4]
		D[-3,-2]=D[2,3]
		D[-3,-3]=D[2,2]
		D[-3,-4]=D[2,1]
		D[-3,-5]=D[2,0]
		return D

# Computes finite-difference matrices for the second derivative
def Diff2(n, o = 2): # Default order is set to second-order
	D=np.zeros((n,n))
	if o == 2: # Second-order
		D[0,0]=2 # Forward scheme (second-order)
		D[0,1]=-5
		D[0,2]=4
		D[0,3]=-1
		for i in range(1,(n-1)):
			D[i,i-1]=1
			D[i,i]=-2
			D[i,i+1]=1
		D[-1,-1]=D[0,0]
		D[-1,-2]=D[0,1]
		D[-1,-3]=D[0,2]
		D[-1,-4]=D[0,3]
		return D
	elif o == 4: # Fourth-order
		D[0,0]=2 # Forward scheme (second-order)
		D[0,1]=-5
		D[0,2]=4
		D[0,3]=-1
		D[1,0]=1 # Central scheme (second-order)
		D[1,1]=-2
		D[1,2]=1
		for i in range(2,(n-2)):
			D[i,i-2]=-1/12
			D[i,i-1]=4/3
			D[i,i]=-5/2
			D[i,i+1]=4/3
			D[i,i+2]=-1/12
		D[-1,-1]=D[0,0]
		D[-1,-2]=D[0,1]
		D[-1,-3]=D[0,2]
		D[-1,-4]=D[0,3]
		D[-2,-1]=D[1,0]
		D[-2,-2]=D[1,1]
		D[-2,-3]=D[1,2]
		return D
	elif o == 6: # Sixth-order
		D[0,0]=2 # Forward-scheme (second-order)
		D[0,1]=-5
		D[0,2]=4
		D[0,3]=-1
		D[1,0]=1 # Central-scheme (second-order)
		D[1,1]=-2
		D[1,2]=1
		D[2,0]=-1/12 # Central-scheme (fourth-order)
		D[2,1]=4/3
		D[2,2]=-5/2
		D[2,3]=4/3
		D[2,4]=-1/12
		for i in range(3,(n-3)):
			D[i,i-3]=1/90		
			D[i,i-2]=-3/20
			D[i,i-1]=3/2
			D[i,i]=-49/18
			D[i,i+1]=3/2
			D[i,i+2]=-3/20
			D[i,i+3]=1/90
		D[-1,-1]=D[0,0]
		D[-1,-2]=D[0,1]
		D[-1,-3]=D[0,2]
		D[-1,-4]=D[0,3]
		D[-2,-1]=D[1,0]
		D[-2,-2]=D[1,1]
		D[-2,-3]=D[1,2]
		D[-3,-1]=D[2,0]
		D[-3,-2]=D[2,1]
		D[-3,-3]=D[2,2]
		D[-3,-4]=D[2,3]
		D[-3,-5]=D[2,4]
		return D
