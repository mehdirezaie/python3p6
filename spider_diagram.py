#
#	Hw 5 problem 3
#	c Mehdi Rezaie
#	mr095415@ohio.edu
#
#   graph V(r) for r < 4 ap
#   V^2(r) = GM r^2 / (ap^2 + r^2)^3/2
#
# to run the code, do
# $> python spider_diagram.py
from __future__ import division, print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def velocity(r):
    vmax = (2./3.**1.5)**0.5
    v = r/(r*r + 1.0)**0.75
    return v/vmax

def F(x, y):
    r = np.sqrt(x*x + y*y)
    phi = np.arctan(y/x)
    if x < 0.0:phi += np.pi     #arctan returns [-pi/2:pi/2] so we add pi by hand
    v = velocity(r)
    vcosp = v * np.cos(phi)
    return vcosp

#
#   plot V(R)
#
r = np.linspace(0.0, 4.0, 50)
v = velocity(r)
plt.subplot(2,1,1)
plt.plot(r, v, 'b-', label=r'$V(R)/V_{max}$')
plt.xlabel(r'$R/a_{p}$');plt.legend(frameon=False);plt.ylim([0., 1.02])

#
#   spider diagram
#
vdes = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
x = np.linspace(-4., 4., 100)
y = np.linspace(-4., 4., 100)
X, Y = np.meshgrid(x, y)
z = np.array([F(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = z.reshape(X.shape)

plt.subplot(2,1,2)
an = np.linspace(0.0, 2.*np.pi, 100)
CS = plt.contour(X, Y, Z, levels=vdes,colors='k')
plt.plot(4.*np.cos(an), 4.*np.sin(an), 'k-')
plt.clabel(CS, fontsize=9, inline=1)
plt.xlim([-4., 4.])
plt.ylim([-4., 4.])
plt.show()
