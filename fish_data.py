import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (mu_g, pdf_g, pdf_g_x, cdf_g, fit_cdf_g,
    pdf_g_a, pdf_g_a_x, fit_cdf_g_a, mu_g_kre, dim_g_kre, support_g_kre, support_g_a,
    support_MP,
    cdf_g_x, cdf_g, cdf_g_f, pdf_g_kre, pdf_g_f, D_KS_Px,
    D_CvM, cdf_g_a_x, fit_cdf_g_a0, fit_cdf_MP, cdf_MP_x, pdf_MP)
from cov_spectrum_random_netw import J2C, C2R, J_g_kre, mix_EI_netw, EI_netw, EI_netw_eqvar
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import sqrt, zeros, ones, eye, linspace, dot, logical_and
import timeit

plt.rcParams.update({'font.size': 18})

# created by Yu Hu (mahy@ust.hk)

# load matlab fish data
flag_exclude_PC1 = True
flag_use_a0 = True
flag_use_part_t = True
flag_MP_a0 = False
flag_load_fitting = True
flag_plot_outlier = False
niter = 1 # use niter=1 to get total eig cost
run_id = 12

from scipy.io import loadmat
mat = loadmat('./data/select_cluster_core_F9.mat')
M_cell = mat['M_cell'][0]
g_select = mat['g_select'][0]
# re-order according to left to right...
id_re_sort = [0,4,3,1,5,2]
g_select = g_select[id_re_sort]
print('cluster id:', g_select)
ng = len(g_select)
if flag_load_fitting:
    # np.savez(file1, a0_ls, ah_ls, gh_ls, s2h_ls, cost_ls, MP_ah_ls,
    # MP_s2h_ls, MP_cost_ls, n_out_ls, n_out_MP_ls)
    with np.load('./data/fish_data_run'+str(run_id)+'.npz') as file1:
        a0_ls = file1['arr_0']
        ah_ls = file1['arr_1']
        gh_ls = file1['arr_2']
        s2h_ls = file1['arr_3']
        cost_ls = file1['arr_4']
        MP_ah_ls = file1['arr_5']
        MP_s2h_ls = file1['arr_6']
        MP_cost_ls = file1['arr_7']
        n_out_ls = file1['arr_8']
        n_out_MP_ls = file1['arr_9']
else:
    a0_ls = zeros(ng)
    ah_ls = zeros(ng)
    gh_ls = zeros(ng)
    s2h_ls = zeros(ng)
    cost_ls = zeros(ng)
    n_out_ls = zeros(ng)
    MP_ah_ls = zeros(ng)
    MP_s2h_ls = zeros(ng)
    MP_cost_ls = zeros(ng)
    n_out_MP_ls = zeros(ng)
for i in range(ng):
    Mg = M_cell[id_re_sort[i]]
    if flag_use_part_t:
        Mg = Mg[:,600:]  # second segment of 1200 total
    n,T = Mg.shape
    # if T<n:
    #     n_use = int(round(T*0.7))
    #     Mg = Mg[:n_use,:]
    #     n = n_use
    #     flag_truncate_n = True
    # else:
    #     flag_truncate_n = False
    a0 = n/T
    a0_ls[i] = a0
    Cg = np.corrcoef(Mg)
    if a0>1:
        eig_Cg = eig_Cg[eig_Cg>0]
    eig_Cg = -np.sort(-eigvalsh(Cg)) # decending order
    if flag_exclude_PC1:
        eig_Cg = eig_Cg[1:]
        n = n-1
    if not flag_load_fitting:
        # fit g theory
        t0 = timeit.default_timer()
        if flag_use_a0:
            ah = a0
            tf_in = ones(n, dtype=bool)
            for t in range(niter):
                gh, s2h, cost = fit_cdf_g_a0(eig_Cg[tf_in], a0, cost='CvM')
                x12h = support_g_a(gh, a0)
                tf_in_new = eig_Cg <= x12h[1]*s2h
                if (tf_in_new == tf_in).all():
                    print('fitting converged at trial '+str(t+1))
                    break
                else:
                    tf_in = tf_in_new
        else:
            # not yet updated with iter...
            tf_in = ones(n, dtype=bool)
            for t in range(niter):
                gh, ah, s2h, cost = fit_cdf_g_a(eig_Cg[tf_in], cost='CvM')
                x12h = support_g_a(gh, ah)
                tf_in_new = eig_Cg <= x12h[1]*s2h
                if (tf_in_new == tf_in).all():
                    print('fitting converged at trial '+str(t+1))
                    break
                else:
                    tf_in = tf_in_new
        gh_ls[i] = gh
        ah_ls[i] = ah
        s2h_ls[i] = s2h
        # cost_ls[i] = D_CvM(eig_Cg/np.mean(eig_Cg), lambda x: cdf_g_a_x(x,gh,ah,normed=True))
        cost_ls[i] = cost
        n_out_ls[i] = n-np.sum(tf_in)
        n_out = n-np.sum(tf_in)
        t1 = timeit.default_timer()
        print('fitting g theory time:', t1-t0)
        print('gh:', gh)
        print('a0:', a0_ls[i], 'ah:', ah)
        print('s2h:', s2h)
        print('n out:', n_out_ls[i])

        # fit MP
        if flag_MP_a0:
            MP_ah = a0
            MP_ah_ls[i] = a0
            MP_s2h = np.mean(eig_Cg)/1
            MP_s2h_ls[i] = MP_s2h
            MP_cost_ls[i] = D_CvM(eig_Cg/np.mean(eig_Cg), lambda x: cdf_MP_x(x,a0))
        else:
            tf_in = ones(n, dtype=bool)
            for t in range(niter):
                MP_ah, MP_s2h, MP_cost = fit_cdf_MP(eig_Cg[tf_in], cost='CvM')
                x12h = support_MP(MP_ah)
                # tf_in_new = logical_and(eig_Cg <=x12h[1], eig_Cg >=x12h[0])
                tf_in_new = eig_Cg <= x12h[1]*MP_s2h
                if (tf_in_new == tf_in).all():
                    print('fitting converged at trial '+str(t+1))
                    break
                else:
                    tf_in = tf_in_new
            MP_ah_ls[i] = MP_ah
            MP_s2h_ls[i] = MP_s2h
            MP_cost_ls[i] = MP_cost
            n_out_MP_ls[i] = n-np.sum(tf_in)
            n_out_MP = n-np.sum(tf_in)
            # print('MP n out:', n_out_MP_ls[i])
    else:
        a0 = a0_ls[i]
        gh = gh_ls[i]
        ah = ah_ls[i]
        s2h = s2h_ls[i]
        cost = cost_ls[i]
        MP_ah = MP_ah_ls[i]
        MP_s2h = MP_s2h_ls[i]
        MP_cost = MP_cost_ls[i]
        n_out = n_out_ls[i]
        n_out_MP = n_out_MP_ls[i]
    fig = plt.figure(figsize=(9,4))
    # if flag_truncate_n:
    #     hist_label = r'$N^*='+str(n)+'$'
    # else:
    if flag_exclude_PC1:
        hist_label = r'$N='+str(n+1)+'$' # previously n=n-1...
    else:
        hist_label = r'$N='+str(n)+'$'
    if i==2:
        plt.hist(eig_Cg, 400, density=True, label=hist_label)
    else:
        plt.hist(eig_Cg, 200, density=True, label=hist_label)
    x, px = pdf_g_a(gh,ah, nx=2000)
    # line_label = r'fit $\hat{g}='+'{:.3f}'.format(round(gh, 2))+\
    #     '$, '+r'$\hat{\alpha}='+'{:.3f}'.format(round(ah, 2))+'$'
    line_label = r'random connectivity $\hat{g}='+'{:.3f}'.format(round(gh, 2))+'$'
    line = plt.plot(x*s2h, px/s2h, 'r', label=line_label, linewidth=1)
    plt.plot([x[0]*s2h,x[-1]*s2h], [0,0], '.', color=line[0].get_color(), markersize=10)
    if flag_plot_outlier and n_out>0:
        plt.scatter(eig_Cg[:n_out], 0.15*ones(n_out), 40, marker='x', color='m', label='outlier')
    x,px = pdf_MP(MP_ah, nx=2000)
    line_label = r'MP $\hat{\alpha}='+'{:.3f}'.format(round(MP_ah,2))+'$'
    line = plt.plot(x*MP_s2h, px/MP_s2h, 'y', label=line_label, linewidth=1)
    plt.legend()
    plt.xlabel('cov eigenvalues')
    plt.ylabel('probabilty')
    plt.ylim(bottom=0)
    if i==2:
        plt.ylim(top = 4)
    else:
        plt.ylim(top = min(3,plt.ylim()[1]))
    plt.tight_layout()
    fig.savefig('./figure/fish_data_run'+str(run_id)+'_id'+str(g_select[i])+'.png', dpi=300)

fig = plt.figure(figsize=(6,5.5))
plt.plot(cost_ls, MP_cost_ls, 'ko', markersize = 10)
plt.xlabel('time-sampled random connectivity theory')
plt.ylabel('Marchenko-Pastur law')
plt.axis('equal')
x12 = plt.xlim()
y12 = plt.ylim()
xline = linspace(min(x12[0],y12[0])*0.95, max(x12[1],y12[1])*1.05, 100)
plt.plot(xline, xline, 'r--')
plt.tight_layout()
fig.savefig('./figure/fish_data_run'+str(run_id)+'_error.png', dpi=300)
print('MP cost order:', np.argsort(MP_cost_ls)+1) # to label clusters
print('g-a cost order:', np.argsort(cost_ls)+1) # to label clusters

if not flag_load_fitting:
    with open('./data/fish_data_run'+str(run_id)+'.npz','wb') as file1:
        np.savez(file1, a0_ls, ah_ls, gh_ls, s2h_ls, cost_ls, MP_ah_ls,
        MP_s2h_ls, MP_cost_ls, n_out_ls, n_out_MP_ls)
