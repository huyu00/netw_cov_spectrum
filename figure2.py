import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x, cdf_g, support_g)
from cov_spectrum_random_netw import J2C
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh
from numpy import sqrt
import timeit

# created by Yu Hu (mahy@ust.hk), Aug 2020

plt.rcParams.update({'font.size': 18})


# different g
g_ls = [0.6,0.7,0.8]
nx_ls = [4000,10000,40000]
ymax_ls = [1.5,1.5,1.7]
for i,g in enumerate(g_ls):
    fig = plt.figure(figsize=(16,6))
    x, px = pdf_g(g, nx=nx_ls[i])
    px2 = sqrt(3)/(2*np.pi) * x**(-5/3)
    plt.subplot(121)
    plt.plot(x, px, color='b', label='g='+str(g))
    plt.plot([x[0],x[-1]], [0,0], '.', markersize=10, color='b')
    plt.plot(x, px2, 'r--', label='power law')
    plt.ylim([0,ymax_ls[i]])
    plt.xlabel('cov eigenvalue $x$')
    plt.ylabel('probability $p(x)$')
    plt.legend()
    plt.subplot(122)
    plt.loglog(x,px)
    plt.loglog(x, px2, 'r--', label='power law')
    plt.xlabel(r'$\log(x)$')
    plt.ylabel(r'$\log(p(x))$')
    plt.tight_layout()
    fig.savefig('./figure/figure2a_'+str(i+1)+'.png', dpi=600)



# 2D colormap, linear scale
ng = 10 # increase to make smoother color map
nk = 200 # increase to make smoother color map
g_ls = np.linspace(0.7,0.97,ng)
k_ls = np.linspace(2.5,20,nk)
log_err = np.zeros((ng,nk))
g_bound = []
k_bound = []
t0 = timeit.default_timer()
for i,g in enumerate(g_ls):
    x12 = support_g(g)
    kmax = (x12[1]/x12[0])**(1/3)
    tf_in = k_ls < kmax
    xk1 = x12[0] * k_ls[tf_in]
    xk2 = x12[1] / (k_ls[tf_in]**2)
    px1 = pdf_g_x(xk1, g)
    px1b = sqrt(3)/(2*np.pi) * xk1**(-5/3)
    er1 = np.abs(np.log(px1b/px1))
    px2 = pdf_g_x(xk2, g)
    px2b = sqrt(3)/(2*np.pi) * xk2**(-5/3)
    er2 = np.abs(np.log(px2b/px2))
    er = np.maximum(er1, er2)
    er = np.maximum.accumulate(er[::-1])[::-1]
    log_err[i,tf_in] = er
    if k_ls[-1]>= kmax:
        g_bound.append(g)
        k_bound.append(kmax)
t1 = timeit.default_timer()
print('calculation time:', t1-t0)
fig = plt.figure(figsize=(10,6.25))
plt.imshow(log_err, extent=[k_ls[0],k_ls[-1],g_ls[0],g_ls[-1]],
    aspect='auto', origin='lower')
plt.plot(k_bound, g_bound, 'y--')
plt.colorbar()
plt.xlabel(r'$\min\left(x/x_{-}, \;\sqrt{x_{+}/x}\right)$')
plt.ylabel('g')
plt.title(r'$|\log(p(x)/\hat{p}(x))|$')
plt.tight_layout()
fig.savefig('./figure/figure2c_lin.png', dpi=600)
