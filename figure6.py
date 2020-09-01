import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x,
    dim_g_kre, support_g, pdf_g_kre, P_branch_g_kre,
    pdf_P_g_kre_x)
from cov_spectrum_random_netw import J2C, C2R, J_g_kre, mix_EI_netw, EI_netw, EI_netw_eqvar
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import sqrt, zeros, ones, eye, linspace, dot
import timeit

# created by Yu Hu (mahy@ust.hk), Aug 2020

plt.rcParams.update({'font.size': 18})

# ER network
g = 0.2
N = 400
p = 8/N
print('p:', p)
x,px = pdf_g(g, nx=2000)
J = np.random.binomial(1, p, (N,N))
w0 = g / sqrt(N*p*(1-p))
max_re = p*N*w0
print('max re:', max_re)
print(w0)
J = J * w0
C = J2C(J)
eig_C = eigvalsh(C)
fig = plt.figure(figsize=(12,6))
plt.hist(eig_C, 80, density=True, label=r'$N='+str(N)+',\;p='+str(p)+'$')
plt.plot(x,px, linewidth=2.5, label='Gaussian theory')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure6a.png', dpi=600, transparent=True)


# ER network, global inhibition (mean subtracted)
g = 0.4
N = 400
p = 0.1
x,px = pdf_g(g, nx=2000)
J = np.random.binomial(1, p, (N,N))
w0 = g / sqrt(N*p*(1-p))
J = J * w0
wi = 1*p*w0
J = J - ones((N,N))*wi
C = J2C(J)
eig_C = eigvalsh(C)
fig = plt.figure(figsize=(9,6))
plt.hist(eig_C, 60, density=True, label=r'$N='+str(N)+',\;p='+str(p)+'$')
plt.plot(x,px, linewidth=2.5, label='Gaussian theory')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure6a_2.png', dpi=600, transparent=True)

# ER network, global inhibition
g = 0.4
N = 400
p = 0.1
x,px = pdf_g(g, nx=2000)
J = np.random.binomial(1, p, (N,N))
w0 = g / sqrt(N*p*(1-p))
J = J * w0
wi = 1.5*p*w0
J = J - ones((N,N))*wi
C = J2C(J)
eig_C = eigvalsh(C)
fig = plt.figure(figsize=(8,6))
plt.hist(eig_C, 60, density=True, label=r'$N='+str(N)+',\;p='+str(p)+'$')
plt.plot(x,px, linewidth=2.5, label='Gaussian theory')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure6a_3.png', dpi=600, transparent=True)


# All inhibitory ER network
g = 0.4
N = 400
p = 0.1
x,px = pdf_g(g, nx=2000)
J = np.random.binomial(1, p, (N,N))
w0 = -g / sqrt(N*p*(1-p))
J = J * w0
C = J2C(J)
eig_C = eigvalsh(C)
fig = plt.figure(figsize=(8,6))
plt.hist(eig_C, 60, density=True, label=r'$N='+str(N)+',\;p='+str(p)+'$')
plt.plot(x,px, linewidth=2.5, label='Gaussian theory')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure6a_4.png', dpi=600, transparent=True)


# mixed EI network
g = 0.4
N = 400
pe = 0.025
pi = 0.075
x,px = pdf_g(g, nx=2000)
J = mix_EI_netw(N,pe,pi,g)
C = J2C(J)
eig_C = eigvalsh(C)
fig = plt.figure(figsize=(9,6))
plt.hist(eig_C, 60, density=True, label=r'$N='+str(N)+'$, mixed EI')
plt.plot(x,px, linewidth=2.5, label='Gaussian theory')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure6a_5.png', dpi=600, transparent=True)

# Dales law EI network, equal e,i
g = 0.4
N = 400
p = 0.1
fe = 0.5
J, para = EI_netw(N, fe, p, p, g, Ie=1) # Ie=1 is balanced
x,px = pdf_g(g, nx=2000)
C = J2C(J)
eig_C = eigvalsh(C)
fig = plt.figure(figsize=(9,6))
plt.hist(eig_C, 100, density=True, label=r'$N='+str(N)+'$, EI netw')
plt.plot(x,px, linewidth=2.5, label='Gaussian theory')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure6a_6_new.png', dpi=600, transparent=True)





# Dales law EI network, equal var
g = 0.4
N = 800
pi = 0.15
fe = 0.7
J, para = EI_netw_eqvar(N, fe, pi, g)
print(para)
pe = para[3]
x,px = pdf_g(g, nx=2000)
C = J2C(J)
eig_C = eigvalsh(C)
fig = plt.figure(figsize=(12,6))
plt.hist(eig_C[1:-1], 80, density=True, label=r'$N='+str(N)+'$, EI netw')
plt.plot(x,px, linewidth=2.5, label='Gaussian theory')
ti = 'g='+str(g)+', fe='+str(fe)+', pe='+str(round(pe*1000)/1000)+', pi='+str(pi)
plt.title(ti)
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure6f.png', dpi=600, transparent=True)
