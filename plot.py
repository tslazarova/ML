import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

n = 500 #number of values for clicks, discr

discrepancies = np.linspace(0,100,n,dtype=int)  # Y
clicks = np.linspace(0,100,n,dtype=int)         # X
[X, Y] = np.meshgrid(clicks, discrepancies) # return coordinate matrices from coordinate vectors

#Generating Z
Z = np.sqrt(Y*X) - 0.1*Y


lvl = [15,40] # !!! [15, 40] # Picked layers
#lvl = [*range(0,100,5)] # Uncomment to see all layers

# makes levels from vmin to vmax the different colors, before and after - same color
# vmin = 15
# vmax = 40

fig, ax = plt.subplots() # using the variable ax for single a Axes

#cs = plt.contourf(X,Y,Z,levels = lvl,cmap='coolwarm',extend='both')
# vmin, vmax included
cs = plt.contourf(X,Y,Z,levels = lvl,vmin=15,vmax=40,cmap='coolwarm',extend='both')

# Uncomment line 16, 19, 20, 26 to plot layers between 15 and 40

clb = fig.colorbar(cs)

ax.set_xlabel('Clicks')
ax.set_ylabel('Discrepancies')
ax.yaxis.set_major_formatter(StrMethodFormatter('%{x:,.0f}'))
plt.title('Z Contour Plot')

# Limit axis
# plt.xlim(0,100)
# plt.ylim(0,20)

# Scale x axis
#ax.set_xscale('log')
#plt.xscale('log')

# Ticks - values used to show specific points on the coordinate axis
plt.xticks(np.arange(0,105,5),rotation ='vertical') # clicks 
plt.yticks(np.arange(0,110,10))  # discrepancies
#plt.grid(b=None)
plt.show()

