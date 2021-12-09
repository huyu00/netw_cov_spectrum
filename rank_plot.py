import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import *
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh
from numpy import sqrt
import timeit

plt.rcParams.update({'font.size': 18})

# created by Yu Hu (mahy@ust.hk)

# compare simulation and iid Gaussian theory
# g_ls = [0.5]
g_ls = [0.9]
# N_ls = [100, 400, 800]
N_ls = [100]
ifig = 0
t0 = timeit.default_timer()
for g in g_ls:
    for N in N_ls:
        ifig += 1
        J = g * randn(N,N)/sqrt(N)
        C = J2C(J)
        eig_C = eigvalsh(C)

        eig_C = -np.sort(-eig_C)
        x_ls = quantile_g(g,N)
        rank_ls = np.arange(N)+1
        fig = plt.figure(figsize=(7.5,6))
        plt.plot(rank_ls, eig_C,'o',fillstyle='none',label='N='+str(N),markersize=5)
        plt.plot(rank_ls, x_ls,'+',label='theory',markersize=5)
        plt.xlabel('rank')
        plt.ylabel('cov eigenvalue')
        plt.legend()
        plt.title('g='+str(g))
        plt.tight_layout()
        fig.savefig('./figure/rank_plot_'+str(ifig)+'.png', dpi=300)

        fig = plt.figure(figsize=(7.5,6))
        plt.loglog(rank_ls, eig_C,'o',fillstyle='none',label='N='+str(N),markersize=5)
        plt.loglog(rank_ls, x_ls,'+',label='theory',markersize=5)
        f_power = lambda p: (p*4*np.pi/3/sqrt(3))**(-3/2)
        x_power_ls = f_power(rank_ls/N)
        plt.loglog(rank_ls, x_power_ls,'r--',label='power law',linewidth=0.5)
        plt.xlabel('rank')
        plt.ylabel('cov eigenvalue')
        plt.legend()
        plt.title('g='+str(g))
        plt.tight_layout()
        fig.savefig('./figure/rank_plot_log_'+str(ifig)+'.png', dpi=300)
t1 = timeit.default_timer()
print('computing time:', t1-t0)
