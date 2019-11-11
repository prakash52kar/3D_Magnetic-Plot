from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy.linalg import norm
import time

t1=time.time()
n=20
r=2
# -------------- Poisition point of grid------------
x = np.array(np.linspace(-r,r,n))
y = np.array(np.linspace(-r,r,n))
z = np.array(np.linspace(-r,r,n))

def cart2pol(x,y):
    rho=np.sqrt(x**2+y**2)
    phi=np.arctan2(y,x)
    return(rho,phi)

Rho,Phi,Z=[cart2pol(x,y)[0],cart2pol(x,y)[1],z] # converting from cartesian to polar coordinates
# Creating All Unique position vectors
pos_vec=np.transpose([np.repeat(Rho,len(Z)*len(Phi)),np.tile(np.repeat(Phi,len(Z)),len(Rho)),np.tile(Z,len(Rho)*len(Phi))])
# Can be checked by printing above
print(pos_vec)

# Creating a loop of wire 
#   -------- Position Points of Loop------------
theta=np.linspace(0,np.pi*2,n) # n-points of wire loop in XY plane 
x_loop=r*np.sin(theta)
y_loop=r*np.cos(theta)
z_loop=np.zeros((1,n))
loop_vec=np.transpose(np.vstack((x_loop,y_loop,z_loop))) # all n position vectors of loop

# Can be seen by printing
print(loop_vec)
################plotting Section#############
# Ploting loop in 3D
fig=plt.figure()
ax=fig.gca(projection='3d')#axes3d(fig)
ax.plot(x_loop,y_loop,0,color='r')
plt.xlabel('x')
plt.ylabel('y')
################################################
#Ploting 
dist=np.hypot(*(loop_vec-pos_vec[0]).T) # Not understand much but...
# Above line creates an array of distance between each point in space...
# from every point of loop

# Can be seen by printing these
print(pos_vec[0])
print(dist)

Cross_product=np.cross(loop_vec,pos_vec[0]) # array of cross product loop points and position point 
# Check by printing
print(Cross_product)

B_point=np.empty((len(pos_vec),3))
i = 10                                          #Current (Amps) in the wire
mu = 1.26 * 10**(-6)                            #Magnetic constanti
for t in range(len(pos_vec)):
    B_field=mu/(4*np.pi)*i*Cross_product/(dist[:,None]**2)
    B_point[t]=np.sum(B_field,axis=0)

# Reshaping and ploting
    
roh,phi,ez=np.meshgrid(Rho,Phi,Z)   
Br=np.reshape(B_point[:,0],roh.shape)
Bphi=np.reshape(B_point[:,1],phi.shape)
Bz=np.reshape(B_point[:,2],roh.shape)

fig=plt.figure()
ax=fig.gca(projection='3d')
ax.quiver(x,y,z,Br,Bphi,Bz,length=1*100000)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Note I have to show the nloop in quiver plot only
# To see the time taken by program 
t2=time.time()
Elapsed=float(str(t2-t1))
print("Time of excecution (s):%.4f"%Elapsed)