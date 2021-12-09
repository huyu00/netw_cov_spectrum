import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x, cdf_g, fit_cdf_g,
    pdf_g_a, pdf_g_a_x, fit_cdf_g_a, support_g, J_uv_outlier)
from cov_spectrum_random_netw import J2C
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh
from numpy import sqrt, zeros, ones, linspace
import timeit

plt.rcParams.update({'font.size': 18})

# created by Yu Hu (mahy@ust.hk)

# Same bulk with div motif
g = 0.4
N = 400
# # r_eig quantile
# ntrial = 1000
# reig = zeros(N*ntrial)
# for t in range(ntrial):
#     J = randn(N,N)/sqrt(N)
#     reig[t*N:(t+1)*N] = np.abs(eigvals(J))
# with open('./data/reig.npy','wb') as file:
#     np.save(file,reig)
reig = np.load('./data/reig.npy')
r_quantile = np.quantile(reig,0.995)
# r_quantile = 1.0238940717401241
print('0.995 quantile r:', r_quantile)
J = randn(N,N)/sqrt(N)*g
# kdivh = 0.25 # squared root of the kappa notations in text
kdivh = 0.45 # squared root of the kappa notations in text
kdiv = sqrt(kdivh**2/(1-kdivh**2)*(g**2/N)) # g is after removing div motifs
print('b entry var b/sqrt(N): ', kdiv*sqrt(N))
print('xuv, x=', N*kdiv)
b = randn(N) * kdiv  # squared root of the kappa notations in text
print('check', np.linalg.norm(b)*np.linalg.norm(ones(N)))
print('outlier theory:',  J_uv_outlier(g,N*kdiv))
print('outlier theory:',  J_uv_outlier(g,np.linalg.norm(b)*np.linalg.norm(ones(N))))


Jm = J + np.outer(ones(N),b)
x,px = pdf_g(g)
C = J2C(J)
Cm = J2C(Jm)
eig_C = eigvalsh(C)
eig_Cm = eigvalsh(Cm)
fig = plt.figure(figsize=(8,4))
plt.hist(eig_C, 40, density=True, label='N='+str(N));
plt.plot(x,px, linewidth=1.5, label='g='+str(g))
x_lim = plt.xlim()
plt.xlabel('cov eigenvalues')
plt.ylabel('probability')
plt.legend()
plt.title(r'Cov spectrum with $J$')
plt.tight_layout()
fig.savefig('./figure/figure3a_1_cov_J.png', dpi=600)

eig_J = eigvals(J)
eig_Jm = eigvals(Jm)
fig = plt.figure(figsize=(6,5))
ax = plt.subplot(111)
plt.scatter(eig_J.real, eig_J.imag, 15*ones(N))
ths = linspace(0,2*np.pi,400)
plt.plot(g*np.cos(ths), g*np.sin(ths), 'r-')
plt.plot(g*np.cos(ths)*r_quantile, g*np.sin(ths)*r_quantile, 'k--', linewidth=0.5)
plt.title(r'$J$ eigenvalues')
ax.set_aspect('equal')
plt.tight_layout()
fig.savefig('./figure/figure3a_Jeig.png', dpi=600)

fig = plt.figure(figsize=(6,5))
ax = plt.subplot(111)
plt.scatter(eig_Jm.real, eig_Jm.imag, 15*ones(N))
ths = linspace(0,2*np.pi,400)
plt.plot(g*np.cos(ths), g*np.sin(ths), 'r-')
plt.plot(g*np.cos(ths)*r_quantile, g*np.sin(ths)*r_quantile, 'k--', linewidth=0.5)
plt.title(r'$J+ e b^T$ eigenvalues')
ax.set_aspect('equal')
plt.tight_layout()
fig.savefig('./figure/figure3a_Jmeig.png', dpi=600)


eig_Cm_in = eig_Cm[1:-1]
fig = plt.figure(figsize=(8,4))
plt.hist(eig_Cm_in, 40, density=True, label='N='+str(N));
plt.xlim(x_lim)
plt.plot(x,px, linewidth=1.5, label='g='+str(g))
plt.xlabel('cov eigenvalues')
plt.ylabel('probability')
plt.legend()
# plt.title('Bulk spectrum with diverging motifs')
plt.title(r'Bulk cov spectrum with $J+ e b^T$')
plt.tight_layout()
fig.savefig('./figure/figure3a_2_cov_Jm_bulk.png', dpi=600)

fig = plt.figure(figsize=(10,4))
plt.hist(eig_Cm_in, 40, density=True);
# plt.scatter(eig_Cm[0], 0, 100, facecolors='none', edgecolors='m')
plt.scatter(eig_Cm[0], 0, 20, marker=2, color='b')
# plt.scatter(eig_Cm[-1], 0, 100, facecolors='none', edgecolors='m')
plt.scatter(eig_Cm[-1], 0, 20, marker=2, color='b')
plt.plot(x,px, linewidth=1.5)
plt.ylim(bottom=0)
plt.xlabel('cov eigenvalues')
plt.ylabel('probability')
# plt.title(r'Cov spectrum with $J+ e b^T$')
plt.tight_layout()
fig.savefig('./figure/figure3a_3_cov_Jm_all.png', dpi=600)
