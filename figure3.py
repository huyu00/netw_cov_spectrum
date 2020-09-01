import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x, cdf_g, support_g)
from cov_spectrum_random_netw import J2C
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh
from numpy import sqrt, zeros, ones
import timeit

# created by Yu Hu (mahy@ust.hk), Aug 2020

plt.rcParams.update({'font.size': 18})


# Same bulk with div motif
g = 0.4
N = 400
J = randn(N,N)/sqrt(N)*g
kdivh = 0.25
kdiv = sqrt(kdivh**2/(1-kdivh**2)*(g**2/N))
b = randn(N) * kdiv
Jm = J + np.outer(ones(N),b)
x,px = pdf_g(g)
C = J2C(J)
Cm = J2C(Jm)
eig_C = eigvalsh(C)
eig_Cm = eigvalsh(Cm)
fig = plt.figure(figsize=(8,6))
plt.hist(eig_C, 40, density=True, label='N='+str(N));
plt.plot(x,px, linewidth=1.5, label='g='+str(g))
x_lim = plt.xlim()
plt.xlabel('cov eigenvalues')
plt.ylabel('probability')
plt.legend()
plt.title('Without motifs')
plt.tight_layout()
fig.savefig('./figure/figure3a_1.png', dpi=600)

eig_Cm_in = eig_Cm[1:-1]
fig = plt.figure(figsize=(8,6))
plt.hist(eig_Cm_in, 40, density=True, label='N='+str(N));
plt.xlim(x_lim)
plt.plot(x,px, linewidth=1.5, label='g='+str(g))
plt.xlabel('cov eigenvalues')
plt.ylabel('probability')
plt.legend()
plt.title('Bulk spectrum with diverging motifs')
plt.tight_layout()
fig.savefig('./figure/figure3a_2.png', dpi=600)

fig = plt.figure(figsize=(10,4))
plt.hist(eig_Cm_in, 40, density=True);
plt.scatter(eig_Cm[0], 0, 100, facecolors='none', edgecolors='m')
plt.scatter(eig_Cm[0], 0, 40, marker='.', color='b')
plt.scatter(eig_Cm[-1], 0, 100, facecolors='none', edgecolors='m')
plt.scatter(eig_Cm[-1], 0, 40, marker='.', color='b')
plt.plot(x,px, linewidth=1.5)
plt.ylim(bottom=0)
plt.xlabel('cov eigenvalues')
plt.ylabel('probability')
plt.title('With diverging motifs')
plt.tight_layout()
fig.savefig('./figure/figure3a_3.png', dpi=600)
