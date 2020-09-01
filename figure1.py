import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x, cdf_g, fit_cdf_g)
from cov_spectrum_random_netw import J2C
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh
from numpy import sqrt
import timeit

# created by Yu Hu (mahy@ust.hk), Aug 2020

plt.rcParams.update({'font.size': 18})


# compare simulation and iid Gaussian theory
g_ls = [0.5]
N_ls = [100, 400, 800]
ifig = 0
for g in g_ls:
    for N in N_ls:
        ifig += 1
        J = g * randn(N,N)/sqrt(N)
        C = J2C(J)
        eig_C = eigvalsh(C)

        x, px = pdf_g(g, nx=1000)
        fig = plt.figure(figsize=(8,6))
        plt.plot(x, px, linewidth=1.5, label='theory')
        plt.hist(eig_C, 40, density=True, label='N='+str(N))
        plt.plot([x[0],x[-1]], [0,0], '.', markersize=10)
        plt.xlabel('cov eigenvalues')
        plt.ylabel('probabilty')
        plt.legend()
        plt.title('g='+str(g))
        plt.tight_layout()
        fig.savefig('./figure/figure1a_'+str(ifig)+'.png', dpi=600)

# different g
g_ls = [0.3,0.4,0.5,0.6,0.7]
x12_ls = []
x12c_ls = []
fig = plt.figure(figsize=(10,6))
for i,g in enumerate(g_ls):
    x, px = pdf_g(g, nx=1000)
    line = plt.plot(x, px, label='g='+str(g))
    x12_ls.append((x[0],x[-1]))
    x12c_ls.append(line[0].get_color())
for i,x12 in enumerate(x12_ls):
    plt.plot(x12, [0,0], '.', color=x12c_ls[i], markersize=10)
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.xlim([0,8])
plt.legend()
plt.tight_layout()
fig.savefig('./figure/figure1b.png', dpi=600)

# dimensionality and g, simulation
g_ls = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
N_ls = [100, 400]
ntrial = 16
t0 = timeit.default_timer()
fD = lambda g: (1-g**2)**2
D_th_ls = [fD(g) for g in g_ls]
fig = plt.figure(figsize=(8,6))
g_plot = np.linspace(0,1, 400)
plt.plot(g_plot, fD(g_plot), label='theory')
for N in N_ls:
    D_ls = np.zeros((len(g_ls), ntrial))
    for i,g in enumerate(g_ls):
        for t in range(ntrial):
            J = g * randn(N,N)/sqrt(N)
            C = J2C(J)
            D_ls[i,t] =(np.trace(C))**2 / np.trace(np.dot(C,C)) / N
    D_ls_mean = np.mean(D_ls, axis=1)
    D_ls_sd = np.std(D_ls, axis=1)
    plt.errorbar(g_ls, D_ls_mean, yerr=D_ls_sd,
        marker='o', linestyle='None', label='N='+str(N))
t1 = timeit.default_timer()
print('simulation time:', t1-t0)
plt.xlabel('g')
plt.ylabel('D/N')
plt.legend()
fig.savefig('./figure/figure1c.png', dpi=600)
