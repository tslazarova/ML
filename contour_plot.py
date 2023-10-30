import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rnd
  
# feature_x = np.linspace(1, 100, 1000) # number spaces evenly w.r.t interval
# feature_y = np.linspace(1, 100, 1000) # geomspace

x = rnd.randint(100, size=(100))
y = rnd.randint(100, size=(100))

[X, Y] = np.meshgrid(x, y)# Return coordinate matrices from coordinate vectors.

fig, ax = plt.subplots(1, 1)

Z =0.55*(X ** 3) + 3*(Y **3) + X**2 + Y**2 # - 200*X + 200*Y

#Z = X ** 3
#Z = np.sqrt(X**2 + Y**2) + 5*X + 5*Y
#Z = np.sqrt(X**3 + Y**3)

lvl = [1,2,3,4,5,6,7,8]
cs = plt.contourf(X,Y,Z,levels = lvl,cmap='coolwarm',extend='both') #,hatches=['-','/','\\']) 
clb = fig.colorbar(cs)


# plots filled contour plot

ax.set_title('Filled Contour Plot')
ax.set_xlabel('Clicks_pub')
ax.set_ylabel('Discrepancies')


fig.colorbar(cs)

plt.show()
