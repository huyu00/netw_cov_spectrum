import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x, cdf_g, fit_cdf_g,
    pdf_g_a, pdf_g_a_x, fit_cdf_g_a, dim_g_kre, support_g, support_g_kre,
     pdf_g_kre, P_branch_g_kre, pdf_P_g_kre_x, pdf_g_kre_normal)
from cov_spectrum_random_netw import J2C, J_g_kre
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh
from numpy import sqrt, zeros, ones, eye, linspace
import timeit

from scipy.signal import find_peaks

plt.rcParams.update({'font.size': 18})

# created by Yu Hu (mahy@ust.hk)

# # verifying diagonal uniformity
# run_id = 14
# gr_ls = [0.5, 0.75]
# kre_ls = [-0.8,-0.4,0,0.4, 0.8]
# # N_ls = np.power(2, range(4,9))*25
# N_ls = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200]
# print(N_ls)
# ntrial = 100
# iplot = -1
# nplot = len(gr_ls)*len(kre_ls)
# ymean_ls = zeros((len(N_ls), nplot))
# ystd_ls = zeros((len(N_ls), nplot))
# t0 = timeit.default_timer()
# for gr in gr_ls:
#     for kre in kre_ls:
#         iplot += 1
#         g = gr/(1+kre)
#         dC_var_mat = zeros((len(N_ls), ntrial))
#         for i,N in enumerate(N_ls):
#             for t in range(ntrial):
#                 J = J_g_kre(N, g,kre)
#                 C = J2C(J)
#                 dC = np.diag(C)
#                 dC_var_mat[i,t] = np.var(dC)
#         ymean_ls[:,iplot] = np.mean(dC_var_mat, axis=1)
#         ystd_ls[:,iplot] = np.std(dC_var_mat, axis=1)
# t1 = timeit.default_timer()
# print('simulation time:', t1-t0)
# t01 = t1 - t0
# with open('./data/figure10a_'+str(run_id)+'.npz','wb') as file1:
#     np.savez(file1, gr_ls, kre_ls, N_ls, ntrial, ymean_ls, ystd_ls, t01)

# plot from data
run_id = 14
flag_lin_scale = False
with np.load('./data/figure10a_'+str(run_id)+'.npz') as file1:
    gr_ls = file1['arr_0']
    kre_ls = file1['arr_1']
    N_ls = file1['arr_2']
    ntrial = file1['arr_3']
    ymean_ls = file1['arr_4']
    ystd_ls = file1['arr_5']
    t01 = file1['arr_6']
print('runtime (min):', round(t01/60,1))
print(N_ls)
print('ntrial', ntrial)
fig = plt.figure(figsize=(20,6))
plt.subplot(121)
nplot = len(gr_ls)*len(kre_ls)
color_ls = plt.cm.hsv(linspace(0,1,nplot))
id_color = np.random.permutation(len(color_ls))# random coloring
iplot = -1
for gr in gr_ls:
    for kre in kre_ls:
        iplot += 1
        g = gr/(1+kre)
        xjitter = randn(len(N_ls))*0.03*N_ls
        tf_pos = ymean_ls[:,iplot] > ystd_ls[:,iplot] # since we are plotting in log-log
        if flag_lin_scale:
            ymean_ls[:, iplot] /= ymean_ls[0,iplot]
            ystd_ls[:, iplot] /= ymean_ls[0,iplot]
        # plt.errorbar(N_ls[tf_pos]+xjitter[tf_pos], ymean_ls[tf_pos,iplot], ystd_ls[tf_pos,iplot],
        #     color = color_ls[id_color[iplot]], label=r'$g_r='+str(gr)+',\; \hat{\kappa}_{re}='+str(kre)+'$')
        plotline1, caplines1, barlinecols1 = plt.errorbar(N_ls+xjitter, ymean_ls[:,iplot], ystd_ls[:,iplot],
                lolims=np.logical_not(tf_pos), color = color_ls[id_color[iplot]], label=r'$g_r='+str(gr)+',\; \hat{\kappa}_{re}='+str(kre)+'$')
        # c0 = np.mean(dC_var_mat, axis=1)[-1]* N_ls[-1]
        # plt.plot(N_ls, c0/N_ls, '--')
ax = (fig.axes)[0]
if not flag_lin_scale:
    ax.set_xscale("log")
    ax.set_yscale("log", nonposy='clip')
ax.set_xticks(N_ls)
ax.set_xticklabels([str(N) for N in N_ls], rotation = 45, ha="right")
# ax.get_xaxis().get_major_formatter().labelOnlyBase = False
plt.legend(loc='upper right', bbox_to_anchor=(1.75, 1))
plt.xlabel(r'$N$')
plt.ylabel('Diagonal variance')
plt.tight_layout()
fig.savefig('./figure/figure10a_'+str(run_id)+'.png', dpi=600)




# verifying J_ii=0 impact
N = 2000
g_ls = [0.3,0.4,0.5]
kre_ls = [-0.6,0,0.6]
iplot = 0
for i,g in enumerate(g_ls):
    for j,kre in enumerate(kre_ls):
        iplot += 1
        fig = plt.figure(figsize=(8,6))
        x, px = pdf_g_kre(g,kre, nx=4000)
        J = J_g_kre(N,g,kre)
        J[eye(N)>0] = 0
        C = J2C(J)
        eig_C = eigvalsh(C)
        plt.hist(eig_C, 80, density=True, label=r'$N='+str(N)+',\; J_{ii}=0$')
        label_plot = r'$g='+str(g)+',\; \hat{\kappa}_{re}='+str(kre)+'$'
        line = plt.plot(x,px, linewidth=1.5, label=label_plot)
        plt.plot([x[0],x[-1]], [0,0], '.', color=line[0].get_color(), markersize=10)
        plt.legend()
        plt.xlabel('cov eigenvalues')
        plt.ylabel('probabilty')
        plt.ylim(bottom=0)
        plt.tight_layout()
        fig.savefig('./figure/figure10b_'+str(iplot+1)+'.png', dpi=600, transparent=True)
