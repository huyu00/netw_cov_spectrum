import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x, cdf_g, fit_cdf_g,
    pdf_g_a, pdf_g_a_x, fit_cdf_g_a, dim_g_kre, support_g, pdf_g_kre, P_branch_g_kre,
    pdf_P_g_kre_x, fit_cdf_g, pdf_g_f, fit_cdf_g_f, fit_cdf_g_f0, fit_cdf_g_kre,
    C_u_outlier, J_uv_outlier, xmin_C_u_outlier, xmin_J_uv_outlier,
    xmin_J_uu_outlier, J_uu_outlier)
from cov_spectrum_random_netw import J2C, C2R, J_g_kre, mix_EI_netw, EI_netw, EI_netw_eqvar
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import sqrt, zeros, ones, eye, linspace, dot, outer
import timeit

plt.rcParams.update({'font.size': 18})
# plt.rcParams.update({'font.size': 16})

# created by Yu Hu (mahy@ust.hk)



# ER with global inhibition dominant (J+uu, x<0) outlier, error vs N
plot_id = 4
flag_load_data = True
flag_plot_power = True
if not flag_load_data:
    g = 0.4
    x1 = -4/3# N=400 value
    # p = 0.1
    # w0 = g / sqrt(n*p*(1-p))
    # wi = 1.5*p*w0
    # x1 = -wi*n + p*n*w0
    n_ls = np.array([400,800,1200,1600,2000])
    nn = len(n_ls)
    ntrial = 400
    x1p_ls = zeros((ntrial,nn))
    t0 = timeit.default_timer()
    for i,n in enumerate(n_ls):
        p = 4*x1**2/(4*x1**2+n*g**2)
        w0 = g / sqrt(n*p*(1-p))
        wi = 1.5*p*w0
        for t in range(ntrial):
            J = np.random.binomial(1, p, (n,n))
            J = J * w0
            J = J - wi*ones((n,n))
            C = J2C(J)
            eig_C = np.sort(eigvalsh(C))
            x1p_ls[t,i] = eig_C[0]
    t1 = timeit.default_timer()
    t01 = t1-t0
    print('simulation time: ', t01)
    with open('./data/figure_outlier_ER_subI_MSE_'+str(plot_id)+'.npz','wb') as file1:
        np.savez(file1,g,x1,n_ls,ntrial,x1p_ls,t01)
else:
    with np.load('./data/figure_outlier_ER_subI_MSE_'+str(plot_id)+'.npz') as file1:
        g = file1['arr_0']
        x1 = file1['arr_1']
        n_ls = file1['arr_2']
        ntrial = file1['arr_3']
        x1p_ls = file1['arr_4']
        t01 = file1['arr_5']
x1p_th = J_uu_outlier(g,x1)
fig = plt.figure(figsize=(8,6))
# plot from data
print('runtime (min):', round(t01/60,1))
print(n_ls)
print('ntrial', ntrial)
rmse1 = np.sqrt(np.mean((x1p_ls - x1p_th)**2, axis=0))
line1 = plt.loglog(n_ls, rmse1,'.-')
if flag_plot_power:
    beta = 1/2
    c0 = np.mean(rmse1 * n_ls**(beta))
    plt.loglog(n_ls, c0*n_ls**(-beta), '--', label=r'$O(N^{-\frac{1}{2}})$', linewidth=1,
        color=line1[0].get_color())
plt.legend()
plt.xlabel(r'$N$')
plt.ylabel('rMSE')
plt.tight_layout()
fig.savefig('./figure/figure_outlier_ER_subI_MSE_'+str(plot_id)+'.png', dpi=600)






# ER all E (J+uu, x>0) outlier, error vs N
plot_id = 4
flag_load_data = True
flag_plot_power = True
if not flag_load_data:
    g = 0.2
    x1 = 0.5714285714285715 # N=400 value
    # p=8/n
    # p = 0.1
    # w0 = g/sqrt(p*(1-p)*n)
    # x1 = n*p*w0
    n_ls = np.array([400,800,1200,1600,2000])
    nn = len(n_ls)
    ntrial = 2
    x1p_ls = zeros((ntrial,nn))
    t0 = timeit.default_timer()
    for i,n in enumerate(n_ls):
        p = x1**2/(x1**2+n*g**2)
        w0 = g / sqrt(n*p*(1-p))
        for t in range(ntrial):
            J = np.random.binomial(1, p, (n,n))
            J = J * w0
            C = J2C(J)
            eig_C = np.sort(eigvalsh(C))
            x1p_ls[t,i] = eig_C[-1]
    t1 = timeit.default_timer()
    t01 = t1-t0
    print('simulation time: ', t01)
    with open('./data/figure_outlier_ER_E_MSE_'+str(plot_id)+'.npz','wb') as file1:
        np.savez(file1,g,x1,n_ls,ntrial,x1p_ls,t01)
else:
    with np.load('./data/figure_outlier_ER_E_MSE_'+str(plot_id)+'.npz') as file1:
        g = file1['arr_0']
        x1 = file1['arr_1']
        n_ls = file1['arr_2']
        ntrial = file1['arr_3']
        x1p_ls = file1['arr_4']
        t01 = file1['arr_5']
x1p_th = J_uu_outlier(g,x1)
fig = plt.figure(figsize=(8,6))
# plot from data
print('runtime (min):', round(t01/60,1))
print(n_ls)
print('ntrial', ntrial)
rmse1 = np.sqrt(np.mean((x1p_ls - x1p_th)**2, axis=0))
line1 = plt.loglog(n_ls, rmse1,'.-')
if flag_plot_power:
    beta = 1/2
    c0 = np.mean(rmse1 * n_ls**(beta))
    plt.loglog(n_ls, c0*n_ls**(-beta), '--', label=r'$O(N^{-\frac{1}{2}})$', linewidth=1,
        color=line1[0].get_color())
plt.legend()
plt.xlabel(r'$N$')
plt.ylabel('rMSE')
plt.tight_layout()
fig.savefig('./figure/figure_outlier_ER_E_MSE_'+str(plot_id)+'.png', dpi=600)







# J + uu outlier, branch, simulation
plot_id = 3
flag_load_data = True
if not flag_load_data:
    nx1_sim = 10
    ntrial = 40
    n = 1000
    g_ls = [0.2,0.3,0.4,0.5]
    x1_max = 0.5 # abs
    ng = len(g_ls)
    x12p_ls = zeros((2,ng,nx1_sim,ntrial))  # neg, pos
    x1_sim_ls = zeros((2,ng,nx1_sim)) # neg, pos
    t0 = timeit.default_timer()
    for j,g in enumerate(g_ls):
        x1_sim = linspace(-x1_max,0, nx1_sim+1) # shift 1 from edge
        x1_sim_ls[0,j,:] = x1_sim[:-1]
        for i,x1 in enumerate(x1_sim_ls[0,j,:]):
            for t in range(ntrial):
                J = randn(n,n)/sqrt(n)*g
                u = randn(n)/sqrt(n)
                Ju = J + x1*outer(u,u)
                C = J2C(Ju)
                x12p_ls[0,j,i,t] = np.min(eigvalsh(C))
        x1_sim = linspace(0,x1_max, nx1_sim+1) # shift 1 from edge
        x1_sim_ls[1,j,:] = x1_sim[1:]
        for i,x1 in enumerate(x1_sim_ls[1,j,:]):
            for t in range(ntrial):
                J = randn(n,n)/sqrt(n)*g
                u = randn(n)/sqrt(n)
                Ju = J + x1*outer(u,u)
                C = J2C(Ju)
                x12p_ls[1,j,i,t] = np.max(eigvalsh(C))
    t1 = timeit.default_timer()
    t01 = t1-t0
    print(t01)
    with open('./data/figure_outlier_Juu_branch_sim_'+str(plot_id)+'.npz','wb') as file1:
        np.savez(file1,g_ls,x1_max,x12p_ls,x1_sim_ls,t01)
else:
    with np.load('./data/figure_outlier_Juu_branch_sim_'+str(plot_id)+'.npz') as file1:
        g_ls = file1['arr_0']
        x1_max = file1['arr_1']
        x12p_ls = file1['arr_2']
        x1_sim_ls = file1['arr_3']
        t01 = file1['arr_4']
        _,_,nx1_sim = x1_sim_ls.shape
        _,_,_,ntrial = x12p_ls.shape
    print('runtime:', t01)
    print('ntrial:', ntrial)
nx1_th = 1000
fig = plt.figure(figsize=(12,6))
plt1 = fig.add_subplot(121)
plt2 = fig.add_subplot(122)
x12p_th_ls = zeros((2,nx1_th))
x1p_ls = zeros(nx1_th)
for j,g in enumerate(g_ls):
    x12 = support_g(g)
    x12_min = xmin_J_uu_outlier(g)
    # x<0
    x1_ls = linspace(-x1_max,x12_min[0], nx1_th+1) # shift 1
    x1_ls = x1_ls[:-1]
    for i,x1 in enumerate(x1_ls):
        x1p_ls[i] = J_uu_outlier(g,x1)
    line = plt1.plot(x1_ls, x1p_ls,label='g='+str(g))
    # x>0
    x1_ls = linspace(x12_min[1],x1_max,nx1_th+1) # shift 1
    x1_ls = x1_ls[1:]
    for i,x1 in enumerate(x1_ls):
        x1p_ls[i] = J_uu_outlier(g,x1)
    line = plt2.plot(x1_ls, x1p_ls, label='g='+str(g),
        color=line[0].get_color())
    x1_pre = linspace(x12_min[0],0,100)
    plt1.plot(x1_pre, ones(100)*x12[0], '--', color=line[0].get_color())
    x1_pre = linspace(0,x12_min[1],100)
    plt2.plot(x1_pre, ones(100)*x12[1], '--', color=line[0].get_color())
    plt1.plot(x12_min[0],x12[0],'ks')
    plt2.plot(x12_min[1],x12[1],'ks')
    #simulation
    plt1.errorbar(x1_sim_ls[0,j,:], np.mean(x12p_ls[0,j,:,:],axis=-1),
        np.std(x12p_ls[0,j,:,:],axis=-1)/sqrt(ntrial), color=line[0].get_color(), ls='none') # bar=SEM
    plt1.plot(x1_sim_ls[0,j,:], np.mean(x12p_ls[0,j,:,:],axis=-1), '.', color=line[0].get_color())
    plt2.errorbar(x1_sim_ls[1,j,:], np.mean(x12p_ls[1,j,:,:],axis=-1),
        np.std(x12p_ls[0,j,:,:],axis=-1)/sqrt(ntrial), color=line[0].get_color(), ls='none') # bar=SEM
    plt2.plot(x1_sim_ls[1,j,:], np.mean(x12p_ls[1,j,:,:],axis=-1), '.', color=line[0].get_color())
plt1.legend()
plt1.set_xlabel(r'x')
plt1.set_ylabel('outlier')
plt1.set_title('Negative x')
# plt2.legend()
plt2.set_xlabel(r'x')
plt2.set_ylabel('outlier')
plt2.set_title('Positive x')
fig.tight_layout()
fig.savefig('./figure/figure_outlier_Juu_branch_sim_'+str(plot_id)+'.png', dpi=600)







# J + uu outlier, branch, theory
plot_id = 1
nx1 = 100
g_ls = [0.2,0.3,0.4,0.5]
x1_max = 0.5 #abs
x1p_ls = zeros(nx1)
fig = plt.figure(figsize=(12,6))
plt1 = fig.add_subplot(121)
plt2 = fig.add_subplot(122)
for g in g_ls:
    x12 = support_g(g)
    # x<0
    x12_min = xmin_J_uu_outlier(g)
    print(x12_min)
    x1_ls = linspace(-x1_max,x12_min[0], nx1+1) # shift 1
    x1_ls = x1_ls[:-1]
    for i,x1 in enumerate(x1_ls):
        x1p_ls[i] = J_uu_outlier(g,x1)
    line = plt1.plot(x1_ls, x1p_ls,label='g='+str(g))
    # x>0
    x1_ls = linspace(x12_min[1],x1_max,nx1+1) # shift 1
    x1_ls = x1_ls[1:]
    for i,x1 in enumerate(x1_ls):
        x1p_ls[i] = J_uu_outlier(g,x1)
    line = plt2.plot(x1_ls, x1p_ls, label='g='+str(g),
        color=line[0].get_color())
    plt1.plot(x12_min[0],x12[0],'ks')
    plt2.plot(x12_min[1],x12[1],'ks')
plt1.legend()
plt1.set_xlabel(r'x')
plt1.set_ylabel('outlier')
plt1.set_title('negative x')
plt2.legend()
plt2.set_xlabel(r'x')
plt2.set_ylabel('outlier')
plt2.set_title('positive x')
fig.tight_layout()
fig.savefig('./figure/figure_outlier_Juu_branch_th_'+str(plot_id)+'.png', dpi=600)









# Dales law EI (J+uv) outlier, error vs N
plot_id = 4
flag_load_data = True
flag_plot_power = True
if not flag_load_data:
    g = 0.4
    x1 = 2.666667 # according to n=400 figure
    fe = 0.5
    # p = 0.1
    # w0 = g/sqrt(p*(1-p)*n)
    # x1 = n*p*w0
    n_ls = np.array([400,800,1200,1600,2000])
    nn = len(n_ls)
    ntrial = 1000
    x1p_ls = zeros((ntrial,nn))
    x2p_ls = zeros((ntrial,nn))
    t0 = timeit.default_timer()
    for i,n in enumerate(n_ls):
        p = x1**2/(x1**2+n*g**2)
        for t in range(ntrial):
            J, para = EI_netw(n, fe, p, p, g, Ie=1) # Ie=1 is balanced
            C = J2C(J)
            eig_C = np.sort(eigvalsh(C))
            x1p_ls[t,i] = eig_C[0]
            x2p_ls[t,i] = eig_C[-1]
    t1 = timeit.default_timer()
    t01 = t1-t0
    print('simulation time: ', t01)
    with open('./data/figure_outlier_daleEI_MSE_'+str(plot_id)+'.npz','wb') as file1:
        np.savez(file1,g,x1,n_ls,ntrial,x1p_ls,x2p_ls,t01)
else:
    with np.load('./data/figure_outlier_daleEI_MSE_'+str(plot_id)+'.npz') as file1:
        g = file1['arr_0']
        x1 = file1['arr_1']
        n_ls = file1['arr_2']
        ntrial = file1['arr_3']
        x1p_ls = file1['arr_4']
        x2p_ls = file1['arr_5']
        t01 = file1['arr_6']
x12p = J_uv_outlier(g,x1)
x1p_th = x12p[0]
x2p_th = x12p[1]
fig = plt.figure(figsize=(8,6))
# plot from data
flag_lin_scale = False
print('runtime (min):', round(t01/60,1))
print(n_ls)
print('ntrial', ntrial)
rmse1 = np.sqrt(np.mean((x1p_ls - x1p_th)**2, axis=0))/x1p_th # relative
line1 = plt.loglog(n_ls, rmse1,'.-', label = 'Left outlier')
rmse2 = np.sqrt(np.mean((x2p_ls - x2p_th)**2, axis=0))/x2p_th # relative
line2 = plt.loglog(n_ls, rmse2,'.-', label = 'Right outlier')
if flag_plot_power:
    beta = 1/2
    c0 = np.mean(rmse1 * n_ls**(beta))
    plt.loglog(n_ls, c0*n_ls**(-beta), '--', label=r'$O(N^{-\frac{1}{2}})$', linewidth=1,
        color=line1[0].get_color())
    c0 = np.mean(rmse2 * n_ls**(beta))
    plt.loglog(n_ls, c0*n_ls**(-beta), '--', label=r'$O(N^{-\frac{1}{2}})$', linewidth=1,
        color=line2[0].get_color())
plt.legend()
plt.xlabel(r'$N$')
plt.ylabel('relative rMSE')
plt.tight_layout()
fig.savefig('./figure/figure_outlier_daleEI_MSE_'+str(plot_id)+'.png', dpi=600)







# Div motifs (J+uv) outlier, error vs N
plot_id = 4
flag_load_data = True
flag_plot_power = True
if not flag_load_data:
    g = 0.4
    x1 = 2.06559 # based on N=400
    # kdivh = 0.25
    # kdiv = sqrt(kdivh**2/(1-kdivh**2)*(g**2/n))
    # z12 = support_g(g)
    # x1 = int(round(z12[1]+2))
    # x1 = n*kdiv
    n_ls = np.array([400,800,1200,1600,2000])
    nn = len(n_ls)
    ntrial = 2
    x1p_ls = zeros((ntrial,nn))
    x2p_ls = zeros((ntrial,nn))
    t0 = timeit.default_timer()
    for i,n in enumerate(n_ls):
        kdiv = x1/n
        for t in range(ntrial):
            J = randn(n,n)/sqrt(n)*g
            b = randn(n) * kdiv
            Jm = J + np.outer(ones(n),b)
            C = J2C(Jm)
            eig_C = np.sort(eigvalsh(C))
            x1p_ls[t,i] = eig_C[0]
            x2p_ls[t,i] = eig_C[-1]
    t1 = timeit.default_timer()
    t01 = t1-t0
    print('simulation time: ', t01)
    with open('./data/figure_outlier_div_MSE_'+str(plot_id)+'.npz','wb') as file1:
        np.savez(file1,g,x1,n_ls,ntrial,x1p_ls,x2p_ls,t01)
else:
    with np.load('./data/figure_outlier_div_MSE_'+str(plot_id)+'.npz') as file1:
        g = file1['arr_0']
        x1 = file1['arr_1']
        n_ls = file1['arr_2']
        ntrial = file1['arr_3']
        x1p_ls = file1['arr_4']
        x2p_ls = file1['arr_5']
        t01 = file1['arr_6']

x12p = J_uv_outlier(g,x1)
x1p_th = x12p[0]
x2p_th = x12p[1]
fig = plt.figure(figsize=(8,6))
# plot from data
flag_lin_scale = False
print('runtime (min):', round(t01/60,1))
print(n_ls)
print('ntrial', ntrial)
rmse1 = np.sqrt(np.mean((x1p_ls - x1p_th)**2, axis=0))/x1p_th # relative
line1 = plt.loglog(n_ls, rmse1,'.-', label = 'Left outlier')
rmse2 = np.sqrt(np.mean((x2p_ls - x2p_th)**2, axis=0))/x2p_th # relative
line2 = plt.loglog(n_ls, rmse2,'.-', label = 'Right outlier')
if flag_plot_power:
    beta = 1/2
    c0 = np.mean(rmse1 * n_ls**(beta))
    plt.loglog(n_ls, c0*n_ls**(-beta), '--', label=r'$O(N^{-\frac{1}{2}})$', linewidth=1,
        color=line1[0].get_color())
    c0 = np.mean(rmse2 * n_ls**(beta))
    plt.loglog(n_ls, c0*n_ls**(-beta), '--', label=r'$O(N^{-\frac{1}{2}})$', linewidth=1,
        color=line2[0].get_color())
# ax = (fig.axes)[0]
# yt_ls = plt.yticks()
# ax.set_yticklabels([str(x) for x in yt_ls])
# y2 = ax.ytickslabels()
# print(y2)
# plt.yticks([0.02,0.04,0.05,0.1,0.2])
# plt.yticks_labels([0.02,0.04,0.05,0.1,0.2])
plt.legend()
plt.xlabel(r'$N$')
plt.ylabel('relative rMSE')
plt.tight_layout()
fig.savefig('./figure/figure_outlier_div_MSE_'+str(plot_id)+'.png', dpi=600)





# J + uv outlier, branch, simulation
plot_id = 7
flag_load_data = True
flag_subtract_edge = True
if not flag_load_data:
    nx1_sim = 10
    ntrial = 100
    n = 4000
    g_ls = [0.2,0.3,0.4,0.5]
    nx1_th = 1000
    x1_max = 1.5
    x1_low = 0.05
    ng = len(g_ls)
    x12p_ls = zeros((2,ng,nx1_sim,ntrial))
    x1_sim_ls = zeros((ng,nx1_sim))
    t0 = timeit.default_timer()
    for j,g in enumerate(g_ls):
        # x1_min = xmin_J_uv_outlier(g)
        x1_sim_ls[j,:] = linspace(x1_low, x1_max, nx1_sim) # shift 1 from edge
        # x1_sim_ls[j,:] = x1_sim[1:]
        for i,x1 in enumerate(x1_sim_ls[j,:]):
            for t in range(ntrial):
                J = randn(n,n)/sqrt(n)*g
                u = randn(n)/sqrt(n)
                v = randn(n)/sqrt(n)
                Ju = J + x1*outer(u,v)
                C = J2C(Ju)
                x12p_ls[0,j,i,t] = np.min(eigvalsh(C))
                x12p_ls[1,j,i,t] = np.max(eigvalsh(C))
    t1 = timeit.default_timer()
    t01 = t1-t0
    print(t01)
    with open('./data/figure_outlier_Juv_branch_sim_'+str(plot_id)+'.npz','wb') as file1:
        np.savez(file1,g_ls,x1_max,x1_low,nx1_th,x12p_ls,x1_sim_ls,t01)
else:
    with np.load('./data/figure_outlier_Juv_branch_sim_'+str(plot_id)+'.npz') as file1:
        g_ls = file1['arr_0']
        x1_max = file1['arr_1']
        x1_low = file1['arr_2']
        nx1_th = file1['arr_3']
        x12p_ls = file1['arr_4']
        x1_sim_ls = file1['arr_5']
        t01 = file1['arr_6']
        _,nx1_sim = x1_sim_ls.shape
        _,_,_,ntrial = x12p_ls.shape
    print('runtime:', t01)
    print('ntrial:', ntrial)
if not flag_subtract_edge:
    fig = plt.figure(figsize=(12,6))
    plt1 = fig.add_subplot(121)
    plt2 = fig.add_subplot(122)
    x12p_th_ls = zeros((2,nx1_th))
    for j,g in enumerate(g_ls):
        x12 = support_g(g)
        xmin12 = xmin_J_uv_outlier(g)
        # Left outlier
        x1_ls = linspace(xmin12[0], x1_max, nx1_th+1) # shift 1
        x1_ls = x1_ls[1:]
        for i,x1 in enumerate(x1_ls):
             x12p = J_uv_outlier(g,x1)
             x12p_th_ls[0,i] = x12p[0]
        line = plt1.plot(x1_ls, x12p_th_ls[0,:],label='g='+str(g))
        # Right outlier
        x1_ls = linspace(xmin12[1], x1_max, nx1_th+1) # shift 1
        x1_ls = x1_ls[1:]
        for i,x1 in enumerate(x1_ls):
             x12p = J_uv_outlier(g,x1)
             x12p_th_ls[1,i] = x12p[1]
        line = plt2.plot(x1_ls, x12p_th_ls[1,:],label='g='+str(g))
        x1_pre = linspace(0,xmin12[0],100)
        plt1.plot(x1_pre, ones(100)*x12[0], '--', color=line[0].get_color())
        x1_pre = linspace(0,xmin12[1],100)
        plt2.plot(x1_pre, ones(100)*x12[1], '--', color=line[0].get_color())
        plt1.plot(xmin12[0],x12[0],'ks', markersize=4)
        plt2.plot(xmin12[1],x12[1],'ks', markersize=4)
        #simulation
        plt1.errorbar(x1_sim_ls[j,:], np.mean(x12p_ls[0,j,:,:],axis=-1),
            np.std(x12p_ls[0,j,:,:],axis=-1)/sqrt(ntrial), color=line[0].get_color(), ls='none') # bar=SEM
        plt1.plot(x1_sim_ls[j,:], np.mean(x12p_ls[0,j,:,:],axis=-1), '.', color=line[0].get_color())
        plt2.errorbar(x1_sim_ls[j,:], np.mean(x12p_ls[1,j,:,:],axis=-1),
            np.std(x12p_ls[1,j,:,:],axis=-1)/sqrt(ntrial), color=line[0].get_color(), ls='none') # bar=SEM
        plt2.plot(x1_sim_ls[j,:], np.mean(x12p_ls[1,j,:,:],axis=-1), '.', color=line[0].get_color())
    plt1.legend()
    plt1.set_xlabel(r'|x|')
    plt1.set_ylabel('outlier')
    plt1.set_title('Left outlier')
    # plt2.legend()
    plt2.set_xlabel(r'|x|')
    plt2.set_ylabel('outlier')
    plt2.set_title('Right outlier')
    fig.tight_layout()
    fig.savefig('./figure/figure_outlier_Juv_branch_sim_'+str(plot_id)+'.png', dpi=600)
else:
    fig = plt.figure(figsize=(12,6))
    plt1 = fig.add_subplot(121)
    plt2 = fig.add_subplot(122)
    x12p_th_ls = zeros((2,nx1_th))
    for j,g in enumerate(g_ls):
        x12 = support_g(g)
        xmin12 = xmin_J_uv_outlier(g)
        # Left outlier
        x1_ls = linspace(xmin12[0], x1_max, nx1_th+1) # shift 1
        x1_ls = x1_ls[1:]
        for i,x1 in enumerate(x1_ls):
             x12p = J_uv_outlier(g,x1)
             x12p_th_ls[0,i] = x12p[0]
        line = plt1.plot(x1_ls, x12p_th_ls[0,:]-x12[0],label='g='+str(g))
        # Right outlier
        x1_ls = linspace(xmin12[1], x1_max, nx1_th+1) # shift 1
        x1_ls = x1_ls[1:]
        for i,x1 in enumerate(x1_ls):
             x12p = J_uv_outlier(g,x1)
             x12p_th_ls[1,i] = x12p[1]
        line = plt2.plot(x1_ls, x12p_th_ls[1,:]-x12[1],label='g='+str(g))
        x1_pre = linspace(0,xmin12[0],100)
        plt1.plot(x1_pre, ones(100)*0, '--', color=line[0].get_color())
        x1_pre = linspace(0,xmin12[1],100)
        plt2.plot(x1_pre, ones(100)*0, '--', color=line[0].get_color())
        plt1.plot(xmin12[0],0,'ks', markersize=4)
        plt2.plot(xmin12[1],0,'ks', markersize=4)
        #simulation
        plt1.errorbar(x1_sim_ls[j,:], np.mean(x12p_ls[0,j,:,:],axis=-1)-x12[0],
            np.std(x12p_ls[0,j,:,:],axis=-1)/sqrt(ntrial), color=line[0].get_color(), ls='none') # bar=SEM
        plt1.plot(x1_sim_ls[j,:], np.mean(x12p_ls[0,j,:,:],axis=-1)-x12[0], '.', color=line[0].get_color())
        plt2.errorbar(x1_sim_ls[j,:], np.mean(x12p_ls[1,j,:,:],axis=-1)-x12[1],
            np.std(x12p_ls[1,j,:,:],axis=-1)/sqrt(ntrial), color=line[0].get_color(), ls='none') # bar=SEM
        plt2.plot(x1_sim_ls[j,:], np.mean(x12p_ls[1,j,:,:],axis=-1)-x12[1], '.', color=line[0].get_color())
    plt1.legend()
    plt1.set_xlabel(r'|x|')
    plt1.set_ylabel('outlier location from the edge')
    plt1.set_title('Left outlier')
    # plt2.legend()
    plt2.set_xlabel(r'|x|')
    plt2.set_ylabel('outlier location from the edge')
    plt2.set_title('Right outlier')
    fig.tight_layout()
    fig.savefig('./figure/figure_outlier_Juv_branch_sim_'+str(plot_id)+'_subtract.png', dpi=600)






# J + uv outlier, branch, theory
plot_id = 1
nx1 = 100
g_ls = [0.2,0.3,0.4,0.5]
x1_max = 1.5
x1_low = 0.05
x12p_ls = zeros((2,nx1))
fig = plt.figure(figsize=(12,6))
plt1 = fig.add_subplot(121)
plt2 = fig.add_subplot(122)
for g in g_ls:
    x12 = support_g(g)
    # print('support',x12)
    x1_min = xmin_J_uv_outlier(g)
    x1_ls = linspace(x1_low,x1_max,nx1)
    for i,x1 in enumerate(x1_ls):
        x12p_ls[:,i] = J_uv_outlier(g,x1)
    line = plt1.plot(x1_ls, x12p_ls[0,:],label='xp1, g='+str(g))
    x00 = g**2/sqrt(1-g**2)
    print(x00)
    x1_ls = linspace(x1_min[1],x1_max,nx1)
    for i,x1 in enumerate(x1_ls):
        x12p_ls[:,i] = J_uv_outlier(g,x1)
    line = plt2.plot(x1_ls, x12p_ls[1,:], label='xp2, g='+str(g),
        color=line[0].get_color())
    plt2.plot(x1_min[1],x12[1],'k.')
plt1.legend()
plt1.set_xlabel(r'x')
plt1.set_ylabel('outlier')
plt1.set_title('Left outlier')
plt2.legend()
plt2.set_xlabel(r'x')
plt2.set_ylabel('outlier')
plt2.set_title('Right outlier')
fig.tight_layout()
fig.savefig('./figure/figure_outlier_Juv_branch_th_'+str(plot_id)+'.png', dpi=600)











# C + 2 outlier, error vs N
plot_id = 3
flag_load_data = True
flag_plot_power = True
if not flag_load_data:
    g = 0.6
    x1 = 17
    x2 = 15
    n_ls = [400,800,1200,1600,2000]
    nn = len(n_ls)
    ntrial = 1000
    x1p_ls = zeros((ntrial,nn))
    x2p_ls = zeros((ntrial,nn))
    t0 = timeit.default_timer()
    for i,n in enumerate(n_ls):
        for t in range(ntrial):
            J = randn(n,n)/sqrt(n)*g
            C = J2C(J)
            u = randn(n)/sqrt(n)
            Cu = C + outer(u,u)*x1
            u = randn(n)/sqrt(n)
            Cu = Cu + outer(u,u)*x2
            eig_Cu = -np.sort(-eigvalsh(Cu)) #decending
            x1p_ls[t,i] = eig_Cu[0]
            x2p_ls[t,i] = eig_Cu[1]
    t1 = timeit.default_timer()
    t01 = t1-t0
    print('simulation time: ', t01)
    with open('./data/figure_outlier_Cu2_MSE_'+str(plot_id)+'.npz','wb') as file1:
        np.savez(file1,g,x1,x2,n_ls,ntrial,x1p_ls,x2p_ls,t01)
else:
    with np.load('./data/figure_outlier_Cu2_MSE_'+str(plot_id)+'.npz') as file1:
        g = file1['arr_0']
        x1 = file1['arr_1']
        x2 = file1['arr_2']
        n_ls = file1['arr_3']
        ntrial = file1['arr_4']
        x1p_ls = file1['arr_5']
        x2p_ls = file1['arr_6']
        t01 = file1['arr_7']

x1p_th,_ = C_u_outlier(g,x1)
x2p_th,_ = C_u_outlier(g,x2)
fig = plt.figure(figsize=(8,6))
# plot from data
flag_lin_scale = False
print('runtime (min):', round(t01/60,1))
print(n_ls)
print('ntrial', ntrial)
rmse1 = np.sqrt(np.mean((x1p_ls - x1p_th)**2, axis=0))
line1 = plt.loglog(n_ls, rmse1,'.-', label = 'outlier 1')
rmse2 = np.sqrt(np.mean((x2p_ls - x2p_th)**2, axis=0))
line2 = plt.loglog(n_ls, rmse2,'.-', label = 'outlier 2')
                # lolims=np.logical_not(tf_pos), color = color_ls[id_color[iplot]],
                # label=r'$g_r='+str(gr)+',\; \hat{\kappa}_{re}='+str(kre)+'$')
if flag_plot_power:
    beta = 1/2
    c0 = np.mean(rmse1 * n_ls**(beta))
    plt.loglog(n_ls, c0*n_ls**(-beta), '--', label=r'$O(N^{-\frac{1}{2}})$', linewidth=1,
        color=line1[0].get_color())
    c0 = np.mean(rmse2 * n_ls**(beta))
    plt.loglog(n_ls, c0*n_ls**(-beta), '--', label=r'$O(N^{-\frac{1}{2}})$', linewidth=1,
        color=line2[0].get_color())
ax = (fig.axes)[0]
# if not flag_lin_scale:
#     ax.set_xscale("log")
#     ax.set_yscale("log", nonposy='clip')
# ax.set_xticks(n_ls)
# ax.set_xticklabels([str(n) for n in n_ls], rotation = 45, ha="right")
plt.legend()
plt.xlabel(r'$N$')
plt.ylabel('rMSE')
plt.tight_layout()
fig.savefig('./figure/figure_outlier_Cu2_MSE_'+str(plot_id)+'.png', dpi=600)







# C + 1 outlier, branch, simulation
plot_id = 8
flag_load_data = True
if not flag_load_data:
    nx1_sim = 10
    ntrial = 100
    n = 1000
    g_ls = [0.2,0.3,0.4,0.5]
    x1_max = 7
    ng = len(g_ls)
    x1p_ls = zeros((ng,nx1_sim,ntrial))
    x1_sim_ls = zeros((ng,nx1_sim))
    t0 = timeit.default_timer()
    for j,g in enumerate(g_ls):
        x1_sim = linspace(0, x1_max, nx1_sim+1) # shift 1 from edge
        x1_sim_ls[j,:] = x1_sim[1:]
        for i,x1 in enumerate(x1_sim_ls[j,:]):
            for t in range(ntrial):
                J = randn(n,n)/sqrt(n)*g
                C = J2C(J)
                u = randn(n)/sqrt(n)
                Cu = C + outer(u,u)*x1
                x1p_ls[j,i,t] = np.max(eigvalsh(Cu))
    t1 = timeit.default_timer()
    t01 = t1-t0
    print(t01)
    with open('./data/figure_outlier_Cu1_branch_sim_'+str(plot_id)+'.npz','wb') as file1:
        np.savez(file1,g_ls,x1_max,x1p_ls,x1_sim_ls,t01)
else:
    with np.load('./data/figure_outlier_Cu1_branch_sim_'+str(plot_id)+'.npz') as file1:
        g_ls = file1['arr_0']
        x1_max = file1['arr_1']
        x1p_ls = file1['arr_2']
        x1_sim_ls = file1['arr_3']
        t01 = file1['arr_4']
        _,nx1_sim = x1_sim_ls.shape
        _,_,ntrial = x1p_ls.shape
    print('runtime:', t01)
    print('ntrial:', ntrial)
nx1_th = 1000
fig = plt.figure(figsize=(8,6))
for j,g in enumerate(g_ls):
    x1_min = xmin_C_u_outlier(g)
    x1_th_ls = linspace(x1_min, x1_max, nx1_th+1) # shift 1 from edge
    x1_th_ls = x1_th_ls[1:]
    x1p_th_ls = zeros(nx1_th)
    for i,x1 in enumerate(x1_th_ls):
        x1p_th_ls[i],z4 = C_u_outlier(g,x1)
    line = plt.plot(x1_th_ls, x1p_th_ls,label='g='+str(g),linewidth=1.5)
    x12 = support_g(g)
    x1_pre = linspace(0,x1_min,100)
    plt.plot(x1_pre, ones(100)*x12[1], '--', color=line[0].get_color())
    x1p_th_sim = zeros(nx1_sim)
    plt.errorbar(x1_sim_ls[j,:], np.mean(x1p_ls[j,:,:],axis=-1),
        np.std(x1p_ls[j,:,:],axis=-1)/sqrt(ntrial), color=line[0].get_color(), ls='none') # bar=SEM
    plt.plot(x1_sim_ls[j,:], np.mean(x1p_ls[j,:,:],axis=-1), '.', color=line[0].get_color())
    plt.plot(x1_min,x12[1],'ks', markersize=4)
plt.legend()
plt.xlabel(r'x')
plt.ylabel('outlier')
plt.tight_layout()
fig.savefig('./figure/figure_outlier_Cu1_branch_sim_'+str(plot_id)+'.png', dpi=600)






# C + 1 outlier, branch, theory
plot_id = 3
nx1 = 1000
g_ls = [0.2,0.3,0.4,0.5]
x1_ls = linspace(0,7,nx1+1)
x1_ls = x1_ls[1:]
x1p_ls = zeros(nx1)
fig = plt.figure(figsize=(8,6))
for g in g_ls:
    x12 = support_g(g)
    print('support',x12)
    tf_valid = ones(nx1,dtype=bool)
    for i,x1 in enumerate(x1_ls):
        x1p_ls[i],z4 = C_u_outlier(g,x1)
        if abs(z4[0].imag)>1e-9:
            tf_valid[i] = False
    line = plt.plot(x1_ls[tf_valid], x1p_ls[tf_valid],label='g='+str(g))
    x1_min = xmin_C_u_outlier(g)
    plt.plot(x1_min, x12[1], '.', color=line[0].get_color())
plt.legend()
plt.xlabel(r'x')
plt.ylabel('outlier')
plt.tight_layout()
fig.savefig('./figure/figure_outlier_Cu1_branch_th_'+str(plot_id)+'.png', dpi=600)
