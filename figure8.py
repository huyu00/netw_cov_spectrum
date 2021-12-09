import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x,
    pdf_ring_NN, pdf_ring_NN_a_d)
from cov_spectrum_random_netw import J2C, C2R
from numpy.random import randn, rand
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import sqrt, zeros, ones, eye, linspace, dot
import timeit

# created by Yu Hu (mahy@ust.hk), Aug 2020

plt.rcParams.update({'font.size': 18})

# Long range ring network
from numpy.fft import fftshift, fft
nx = 100
g = 0.5
x_i = linspace(0,1,nx+1)
x_i = x_i[:-1]
s1 = 0.05
s2 = 0.1
j_i = np.exp(-np.minimum(x_i, 1-x_i)**2 / 2/s1**2) - 0.5 * np.exp(-np.minimum(x_i, 1-x_i)**2 / 2/s2**2)
j_k = fft(j_i)
w0 =  g/np.max(j_k.real)
j_k = j_k*w0
c_k = 1 / np.abs(1 - j_k)**2

fig = plt.figure(figsize=(8,5))
plt.plot(x_i-0.5, fftshift(j_i)*w0)
plt.xlabel(r'$x_j-x_i$')
plt.ylabel(r'$J_{ij}$')
plt.xlim([-0.5,0.5])
plt.tight_layout()
fig.savefig('./figure/figure8a_profile.png', dpi=300)

fig = plt.figure(figsize=(8,6))
plt.hist(c_k, 80, density=False, label=r'$N='+str(nx)+'$')
# jk_inf
kmax = 20
import scipy.integrate as integrate
jf = lambda x: np.exp(-x**2/2/s1**2) - 0.5*np.exp(-x**2/2/s2**2)
j_k_inf = zeros(kmax)
for k in range(kmax):
    j_k_inf[k] = integrate.quad(lambda x: jf(x)*np.cos(2*np.pi*k*x), -0.5, 0.5)[0]
w0 =  g/np.max(j_k_inf)
j_k_inf = j_k_inf*w0
j_k_inf = -np.sort(-j_k_inf)
c_k_inf = 1 / np.abs(1 - j_k_inf)**2
plt.scatter(c_k_inf[:5], 4*ones(5), 50, marker='x', color='r', label='Fourier coefficient')
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.legend()
fig.savefig('./figure/figure8a_eig.png', dpi=300)


# NN ring 1D
# regular case
Jx = 0.4
Jy = 0.2
N = 400
th_k = np.linspace(0, 2*np.pi, N+1)
th_k = th_k[:-1]
j_k = (Jx+Jy)*np.cos(th_k) + (Jx-Jy)*np.sin(th_k)*1j
c_k = 1 / np.abs(1-j_k)**2  # eigenvalue of the covariance
x,px = pdf_ring_NN(Jx,Jy, nx=10000)
fig = plt.figure(figsize=(8,6))
plt.hist(c_k, 80, density=True, label=r'$N='+str(N)+'$')
line_plot = plt.plot(x[1:-1],px[1:-1], linewidth=1.5, label=r'$J_{i-1,i}='+str(Jx)+',\; J_{i+1,i}='+str(Jy)+'$')
yline = linspace(0,5,400)
plt.plot(x[-1]*ones(400), yline, color=line_plot[0].get_color(), linestyle='--', linewidth=2)
plt.plot(x[0]*ones(400), yline, color=line_plot[0].get_color(), linestyle='--', linewidth=2)
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim([0,2])
plt.legend()
fig.savefig('./figure/figure8b_regular.png', dpi=300)

# folded case
Jx = -1
Jy = 0.6
N = 1000
th_k = np.linspace(0, 2*np.pi, N+1)
th_k = th_k[:-1]
j_k = (Jx+Jy)*np.cos(th_k) + (Jx-Jy)*np.sin(th_k)*1j
c_k = 1 / np.abs(1-j_k)**2  # eigenvalue of the covariance
x,px = pdf_ring_NN(Jx,Jy, nx=10000)
fig = plt.figure(figsize=(8,6))
plt.hist(c_k, 200, density=True, label=r'$N='+str(N)+'$')
xmid = (1+abs(Jx+Jy))**(-2)
ix_mid = np.argmin((x - xmid)**2)
line_plot = plt.plot(x[1:ix_mid],px[1:ix_mid], linewidth=1.5, label=r'$J_{i-1,i}='+str(Jx)+',\; J_{i+1,i}='+str(Jy)+'$')
plt.plot(x[ix_mid+1:-1],px[ix_mid+1:-1], color=line_plot[0].get_color(), linewidth=1.5)
yline = linspace(0,10,400)
plt.plot(x[-1]*ones(400), yline, color=line_plot[0].get_color(), linestyle='--', linewidth=2)
plt.plot(x[0]*ones(400), yline, color=line_plot[0].get_color(), linestyle='--', linewidth=2)
yline = linspace(px[ix_mid+1],10,400)
plt.plot(xmid*ones(400), yline, color=line_plot[0].get_color(), linestyle='--', linewidth=2)
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.xlim([0,1.2])
plt.ylim([0,6])
plt.legend()
fig.savefig('./figure/figure8b_folded.png', dpi=300)





# # multi-dimensional ring
# a = 0.6 # total a, i.e. a_i*d
# d_ls = range(2,6)
# t0 = timeit.default_timer()
# nx = 400 # use 1000 for smoother curve
# x_ls = zeros((len(d_ls),nx))
# px_ls = zeros((len(d_ls),nx))
# for i, d in enumerate(d_ls):
#     fig = plt.figure(figsize=(8,6))
#     x,px = pdf_ring_NN_a_d(a, d, nx=nx)
#     x_ls[i,:] = x
#     px_ls[i,:] = px
#     if d>2:
#         plt.plot(x, px, color='b', linewidth=1.5, label=str(d)+'D ring')
#     else:
#         plt.plot(x, px, color='b', linewidth=1.5, label=str(d)+'D ring')
#         yline = linspace(0,3,400)
#         plt.plot(1*ones(400), yline, color='grey', linestyle='--', linewidth=1.5)
#         plt.ylim(top=1.5)
#     plt.xlabel('cov eigenvalues')
#     plt.ylabel('probabilty')
#     plt.legend()
#     plt.ylim(bottom=0)
#     plt.tight_layout()
#     fig.savefig('./figure/figure8c_'+str(d)+'.png', dpi=300)
# t1 = timeit.default_timer()
# print('calculation time:', t1-t0)
# with open('./data/figure8c.npz','wb') as file1:
#     np.savez(file1, x_ls, px_ls, d_ls, a, nx)

# plot from data
with np.load('./data/figure8c.npz') as file1:
    x_ls = file1['arr_0']
    px_ls = file1['arr_1']
    d_ls = file1['arr_2']
    a = file1['arr_3']
    nx = file1['arr_4']
for i, d in enumerate(d_ls):
    fig = plt.figure(figsize=(8,6))
    x = x_ls[i]
    px = px_ls[i]
    if d>2:
        plt.plot(x, px, color='b', linewidth=1.5, label=str(d)+'D ring')
    else:
        plt.plot(x, px, color='b', linewidth=1.5, label=str(d)+'D ring')
        yline = linspace(0,3,400)
        plt.plot(1*ones(400), yline, color='grey', linestyle='--', linewidth=1.5)
        plt.ylim(top=1.5)
    plt.xlabel('cov eigenvalues')
    plt.ylabel('probabilty')
    plt.legend()
    plt.ylim(bottom=0)
    plt.title('Dim=' + str(d))
    plt.tight_layout()
    fig.savefig('./figure/figure8c_'+str(d)+'.png', dpi=300)
