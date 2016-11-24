import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d.axes3d import Axes3D

def gauss2d(x,y,x0,y0,sx,sy,A):
	xledd = np.divide(np.square(x-x0),sx**2)
	yledd = np.divide(np.square(y-y0),sy**2)
	Z = A*np.exp(-0.5*(xledd+yledd))
	return(Z)

def make_atoms(x,y,m1=(-1,-1),m2=(3,3),s1=(2,2),s2=(1,1)):
	'''greates 2 gaussians, returns the convolved array Z. Can take tuple arguments for coordinates of mean and sigma in x and y directions for both gaussians''' 
	g1 = gauss2d(x,y,m1[0],m1[1],s1[0],s1[1],1)
	g2 = gauss2d(x,y,m2[0],m2[1],s2[0],s2[1],0.5)
	Z = np.add(g1,g2)
	return(Z)

def set_alpha(l,start=15,stop=35):
	'''function that sets alpha for lines or segments of lines in wireframe. returns array of alpha values. Start=Line where fade in starts. Stop=Line where fade in is complete'''
	alpha = np.ones(l)
	alpha[0:int(l/2)] = 0
	da = 1/(stop-start)
	a = 0+da
	indices = range(start,stop)
	for idx in indices:
		alpha[idx] = a
		a += da
	return(alpha)

#Configure figsize and dpi suitable for logo
fig = plt.figure(figsize=(200/60,200/60), dpi=60)

ax = fig.add_subplot(1, 1, 1, projection='3d')

##--config x-y plane

i0 = -5
i1 = i0+10
i_mid = (i0+i1)/2 
j0 = -5
j1 = j0+10
j_mid = (j0+j1)/2 

#---- Make surface plotted gauss ("experimental gauss")
i = np.arange(i0,i1, 0.05)
j = np.arange(j0,j1, 0.05)
X, Y = np.meshgrid(i, j)
Z = make_atoms(X,Y)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.bone,linewidth=0, antialiased=False,alpha=0.7,clim=(0,1.1))

#---- Make wireframe ("model part")
m = np.arange(i0, i1, 0.25)
l = m.size
n = np.arange(j0,j1, 0.25)
X, Y = np.meshgrid(m,n)
Z = make_atoms(X,Y)

#makes plot
alpha_list = set_alpha(l)
for k in np.arange(0,l,2):
	alpha = alpha_list[k]
	ax.plot(X[k,:], Y[k,:], Z[k,:],color='purple',alpha=alpha)
	for p in np.arange(0,l,2):
		ax.plot(X[k:k+3,p], Y[k:k+3,p], Z[k:k+3,p],color='pink',alpha=alpha)

ax.set_axis_off()

# Set angle from which saved figure is viewed
ax.view_init(elev=0, azim=150)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_zlim(0,1)

fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
fig.savefig('atomaplogo.png',transparent=True,bbox_inches='tight',pad_inches=0)
