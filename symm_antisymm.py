import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x, cdf_g, fit_cdf_g,
    pdf_g_a, pdf_g_a_x, fit_cdf_g_a, dim_g_kre, support_g, support_g_kre,
     pdf_g_kre, P_branch_g_kre, pdf_P_g_kre_x,
     support_g_kre_normal, pdf_g_kre_normal, pdf_g_kre_normal_x,
     pdf_g_kre_normal_sim, r_theta_length)
from cov_spectrum_random_netw import J2C, J_g_kre
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh
from numpy import sqrt, zeros, ones, linspace
import timeit

from scipy.signal import find_peaks

plt.rcParams.update({'font.size': 18})


# created by Yu Hu (mahy@ust.hk)


# symm example
ng = 8
# g_ls = linspace(0.2,3, ng)
g_ls = [0.15,0.2,0.25,0.3,0.35]
x12_ls = []
x12c_ls = []
fig = plt.figure(figsize=(10,6))
for i, g in enumerate(g_ls):
    x,px = pdf_g_kre(g,1, nx=10000)
    line = plt.plot(x,px, linewidth=2, label='$g='+str(g)+'$')
    x12_ls.append((x[0],x[-1]))
    x12c_ls.append(line[0].get_color())
for i,x12 in enumerate(x12_ls):
    plt.plot(x12, [0,0], '.', color=x12c_ls[i], markersize=10)
plt.legend()
plt.xlim([0,5])
plt.ylim(bottom=0)
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.title('Random symmetric connectivity')
plt.tight_layout()
fig.savefig('./figure/symm_J.png', dpi=300)


# anti-symm example
ng = 8
# g_ls = linspace(0.2,3, ng)
g_ls = [0.2,0.4,0.6,1,2,3]
x12_ls = []
x12c_ls = []
fig = plt.figure(figsize=(10,6))
for i, g in enumerate(g_ls):
    x,px = pdf_g_kre(g,-1, nx=10000)
    line = plt.plot(x,px, linewidth=2, label='$g='+str(g)+'$')
    x12_ls.append((x[0],x[-1]))
    x12c_ls.append(line[0].get_color())
for i,x12 in enumerate(x12_ls):
    plt.plot(x12, [0,0], '.', color=x12c_ls[i], markersize=10)
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.title('Random antisymmetric connectivity')
plt.tight_layout()
yline=linspace(0,9,400)
plt.plot(x[-1]*ones(400), yline, color='grey', linestyle='--', linewidth='3')
plt.ylim(0,9)
fig.savefig('./figure/antisymm_J.png', dpi=300)
