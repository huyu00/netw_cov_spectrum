import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x,
    dim_g_kre, support_g, support_g_kre,
     pdf_g_kre, P_branch_g_kre, pdf_P_g_kre_x,
     support_g_kre_normal, pdf_g_kre_normal, pdf_g_kre_normal_x,
     pdf_g_kre_normal_sim, r_theta_length)
from cov_spectrum_random_netw import J2C, J_g_kre
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh
from numpy import sqrt, zeros, ones, linspace
import timeit

from scipy.signal import find_peaks

# created by Yu Hu (mahy@ust.hk), Aug 2020

plt.rcParams.update({'font.size': 18})


# anti-symm example
ng = 8
g_ls = [0.2,0.4,0.6,1,2,3]
x12_ls = []
x12c_ls = []
fig = plt.figure(figsize=(8,6))
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
fig.savefig('./figure/figure5a.png', dpi=600)

# example bimodal
g = 1.1
kre = -0.99
N = 400
J = J_g_kre(N,g,kre)
C = J2C(J)
eig_C = eigvalsh(C)
x,px = pdf_g_kre(g, kre, nx = 10000)
fig = plt.figure(figsize=(8,6))
plt.hist(eig_C, 40, density=True, color='orange', label='N='+str(N))
t0 = timeit.default_timer()
ipeaks, _ = find_peaks(px)
t1 = timeit.default_timer()
print('finding peaks:', t1-t0)
line = plt.plot(x,px,'b', linewidth=2, label='$g='+str(g)+',\;\hat{\kappa}_{re}='+str(kre)+'$')
plt.plot([x[0],x[-1]], [0,0], '.', color=line[0].get_color(), markersize=10)
plt.plot(x[ipeaks], px[ipeaks], 'mo',  markersize=10, label='peak')
plt.legend()
plt.ylim(bottom=0)
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.tight_layout()
fig.savefig('./figure/figure5b_example_1b.png', dpi=600)


# curve along kre
kre = -0.99
ng = 100 # increase for smoother curve
g_ls = linspace(0, 10, ng+2)
g_ls = g_ls[g_ls>0]
g_ls = g_ls[g_ls*(1+kre)<1]
peak1_ls = []
peak2_ls = []
g1_ls = []
g2_ls = []
flag_branch1 = False
flag_branch2 = False
g_branch1 = 0
peak_branch1 = 0
peak_branch1_b = 0 # the other peak
g_branch2 = 0
peak_branch2 = 0
peak_branch2_b = 0
t0 = timeit.default_timer()
for i,g in enumerate(g_ls):
    x,px = pdf_g_kre(g, kre, nx = 4000)
    ipeaks, _ = find_peaks(px)
    ipeaks = np.sort(ipeaks)
    if len(ipeaks)>1:
        if not flag_branch1:
            g_branch1 = g
            peak_branch1 = x[ipeaks[0]]
            peak_branch1_b = x[ipeaks[1]]
            flag_branch1 = True
    elif flag_branch1 and (not flag_branch2):
        flag_branch2 = True
        g_branch2 = g_ls[i-1]
        peak_branch2 = peak1_ls[-1]
        peak_branch2_b = peak2_ls[-1]
    if not flag_branch2:
        peak1_ls.append(x[ipeaks[-1]])
        g1_ls.append(g)
    if flag_branch1:
        peak2_ls.append(x[ipeaks[0]])
        g2_ls.append(g)
t1 = timeit.default_timer()
print('finding peaks:', t1-t0)
fig = plt.figure(figsize=(8,6))
plt.plot(g1_ls, peak1_ls, 'b', zorder=2, linewidth=2, label='$\hat{\kappa}_{re}='+str(kre)+'$')
plt.plot(g2_ls, peak2_ls, color='orange', zorder=2, linewidth=2)
if flag_branch1:
    plt.plot(g_branch1, peak_branch1, 'r.', markersize=10, zorder=3)
    yline = linspace(peak_branch1,  peak_branch1_b, 100)
    plt.plot(g_branch1*ones(100), yline, '--', color = 'grey', zorder=1, linewidth=1)
if flag_branch2:
    plt.plot(g_branch2, peak_branch2, 'm.', markersize=10, zorder=3)
    yline = linspace(peak_branch2_b,  peak_branch2, 100)
    plt.plot(g_branch2*ones(100), yline, '--', color = 'grey', zorder=1, linewidth=1)
plt.legend()
plt.xlabel(r'$g$')
plt.ylabel(r'$x_{\max}$')
plt.xlim([0,10])
plt.ylim(bottom=0)
if flag_branch1:
    plt.title(str(g_branch1))
plt.tight_layout()
fig.savefig('./figure/figure5b_curve_g_'+str(kre)+'.png', dpi=600)
if flag_branch1:
    print('g branch point 1:', g_branch1)
if flag_branch2:
    print('g branch point 2:', g_branch2)




# # bimodal region in (gr, kre)
# run_id = 1
#
# ng = 10
# nkre = 10
# # For a smooth plot, use parameters below (run time 14.7 hrs on a regular desktop)
# # # ng = 800
# # # nkre = ng
#
# # zoomed in view
# g_ls = np.linspace(0,2,ng)
# kre_ls = np.linspace(-1,-0.9, nkre)
#
# # whole region
# # g_ls = np.linspace(0,2,ng)
# # kre_ls = np.linspace(-1,1, nkre)
#
# xmax_mat = np.zeros((ng,nkre)) # right peak if 2 peaks
# xmax2_mat = np.zeros((ng,nkre)) # the left peak if 2 peaks
# g_bound = []
# kre_bound = []
# g_branch1 = []
# kre_branch1 = []
# g_branch2 = []
# kre_branch2 = []
# t0 = timeit.default_timer()
# for j,kre in enumerate(kre_ls):
#     if kre>-1:
#         tf_branch1 = False
#         tf_branch2 = False
#         tf_bound = False
#         for i, g in enumerate(g_ls):
#             if g==0:
#                 xmax_mat[i,j] = 1
#             elif g*(1+kre)<1:
#                 x,px = pdf_g_kre(g, kre, nx = 10000)
#                 ipeaks, _ = find_peaks(px)
#                 ipeaks = np.sort(ipeaks)
#                 xmax_mat[i,j] = x[ipeaks[-1]]
#                 if len(ipeaks)>1:
#                     xmax2_mat[i,j] = x[ipeaks[0]]
#                 if (not tf_branch1) and len(ipeaks)>1:
#                     tf_branch1 = True
#                     g_branch1.append(g)
#                     kre_branch1.append(kre)
#                 if tf_branch1 and (not tf_branch2) and len(ipeaks)==1:
#                     tf_branch2 = True
#                     g_branch2.append(g)
#                     kre_branch2.append(kre)
#             elif not tf_bound:
#                 tf_bound = True
#                 g_bound.append(g)
#                 kre_bound.append(kre)
#     else:
#         xmax_mat[:,j] = 1
# t1 = timeit.default_timer()
# print('calculation time:', t1-t0)
# with open('./data/figure5c_'+str(run_id)+'.npz','wb') as file1:
#     np.savez(file1, xmax_mat, kre_ls, g_ls, kre_bound, g_bound,
#     kre_branch1, g_branch1, kre_branch2, g_branch2, xmax2_mat, t1-t0)
#
# plot from data
run_id = 17
flag_inset = True
if flag_inset:
    plt.rcParams.update({'font.size': 48})
    # plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
with np.load('./data/figure5c_'+str(run_id)+'.npz') as file1:
    xmax_mat = file1['arr_0']
    kre_ls = file1['arr_1']
    g_ls = file1['arr_2']
    kre_bound = file1['arr_3']
    g_bound = file1['arr_4']
    kre_branch1 = file1['arr_5']
    g_branch1 = file1['arr_6']
    kre_branch2 = file1['arr_7']
    g_branch2 = file1['arr_8']
    xmat2_mat = file1['arr_9']
    t01 = file1['arr_10']
print('runtime (hrs):', round(t01/3600,1))
if flag_inset:
    fig = plt.figure(figsize=(15,10.5))
else:
    fig = plt.figure(figsize=(10,7))
plt.imshow(xmax_mat, extent=[kre_ls[0],kre_ls[-1],g_ls[0],g_ls[-1]],
    aspect='auto', origin='lower')
if flag_inset:
    plt.plot(kre_bound, g_bound, 'y--', label=r'$g_c$', linewidth=5)
else:
    plt.plot(kre_bound, g_bound, 'y--', label=r'$g_c$', linewidth=3)
plt.plot(kre_branch1, g_branch1, 'r--', linewidth=3)
plt.plot(kre_branch2, g_branch2, 'm-', linewidth=3)
plt.colorbar()
plt.xlabel(r'$\hat{\kappa}_{re}$')
plt.ylabel(r'$g$')
if not flag_inset:
    plt.title(r'$x_{\max}$')
plt.legend()
plt.tight_layout()
fig.savefig('./figure/figure5c_'+str(run_id)+'.png', dpi=600, transparent=True)




plt.rcParams.update({'font.size': 18})
# example spectrum with normal matching
kre = -0.7
gr = 0.3
g = gr/(1+kre)
fig = plt.figure(figsize=(8,6))
x,px = pdf_g_kre(g,kre, nx=10000)
line_label = r'$g_r='+str(gr)+',\;\hat{\kappa}_{re}='+str(kre)+'$'
plt.plot(x,px,'b', linewidth=2, label=line_label)
xn, pxn = pdf_g_kre_normal(g,kre, nx=10000)
plt.plot(xn,pxn, 'm', linewidth=2, label='matching normal')
i_max = np.argmax(pxn)
plt.scatter(xn[i_max], pxn[i_max],20, 'm')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure5d.png', dpi=600)



# example spectrum with normal matching
plt.rcParams.update({'font.size': 26})
iplot = 0
kre_ls = [0.2,0.5]
# g = 1
# gr = g*(1+kre)
gr_ls = [0.4,0.6]
fig = plt.figure(figsize=(8,6))
for kre in kre_ls:
    for gr in gr_ls:
        iplot += 1
        g = gr/(1+kre)
        plt.clf()
        x,px = pdf_g_kre(g,kre, nx=10000)
        line_label = r'$g_r='+str(gr)+',\;\hat{\kappa}_{re}='+str(kre)+'$'
        line = plt.plot(x,px,'b', linewidth=2, label=line_label)
        plt.plot([x[0],x[-1]], [0,0], '.',color=line[0].get_color(), markersize=30)
        xn, pxn = pdf_g_kre_normal(g,kre, nx=10000)
        line = plt.plot(xn,pxn, 'm', linewidth=2, label='matching normal')
        plt.plot([xn[0],xn[-1]], [0,0], '.',color=line[0].get_color(), markersize=30)
        # i_max = np.argmax(pxn)
        # plt.scatter(xn[i_max], pxn[i_max],20, 'm')
        plt.legend()
        plt.xlabel('cov eigenvalues')
        plt.ylabel('probabilty')
        plt.ylim(bottom=0)
        plt.tight_layout()
        fig.savefig('./figure/figure5d_positive_kappa_'+str(iplot)+'.png',
            dpi=300, transparent=True)



# illustrating normal matching spectrum
kre = -0.7
gr = 0.3
g = gr/(1+kre)
a = g*(1+kre)
b = g*(1-kre)
r1 = 1-a
r2 = sqrt((b**2-a**2+1)*b**2 / (b**2-a**2))
nr = 10
r_ls = linspace(r1, r2, nr+2)
r_ls = r_ls[1:-1]
th_ls = linspace(0,2*np.pi, 400)
fig = plt.figure(figsize=(8,6))
plt.plot(1,0,'+')
plt.plot(a*np.cos(th_ls), b*np.sin(th_ls), 'k', linewidth=1.5, zorder=2)
for r in r_ls:
    plt.plot(r*np.cos(th_ls)+1, r*np.sin(th_ls), zorder=1)
plt.axis('equal')
fig.savefig('./figure/figure5_supp_e.png', dpi=600)




# # bimodal region for normal matching pdf
# run_id = 1
# ng = 40 # 400
# nkre = 20 # 200
# # for smooth plot use:
# # ng = 400
# # nkre = 200
# g_ls = np.linspace(0,2,ng)
# kre_ls = np.linspace(-1,1, nkre)
# xmax_mat = np.zeros((ng,nkre)) # right peak if 2 peaks
# xmax2_mat = np.zeros((ng,nkre)) # the left peak if 2 peaks
# g_bound = []
# kre_bound = []
# g_branch1 = []
# kre_branch1 = []
# g_branch2 = []
# kre_branch2 = []
# t0 = timeit.default_timer()
# for j,kre in enumerate(kre_ls):
#     if kre>-1:
#         tf_branch1 = False
#         tf_branch2 = False
#         tf_bound = False
#         for i, g in enumerate(g_ls):
#             if g==0:
#                 xmax_mat[i,j] = 1
#             elif g*(1+kre)<1:
#                 x12 = support_g_kre_normal(g,kre)
#                 if x12[1]>10: # adjust discretization for very long tail distributions
#                     x = linspace(x12[0],1, 4000)
#                     px = pdf_g_kre_normal_x(x,g,kre)
#                 else:
#                     x,px = pdf_g_kre_normal(g, kre, nx=4000)
#                 ipeaks, _ = find_peaks(px)
#                 ipeaks = np.sort(ipeaks)
#                 xmax_mat[i,j] = x[ipeaks[-1]]
#                 if len(ipeaks)>1:
#                     xmax2_mat[i,j] = x[ipeaks[0]]
#                 if (not tf_branch1) and len(ipeaks)>1:
#                     tf_branch1 = True
#                     g_branch1.append(g)
#                     kre_branch1.append(kre)
#                 if tf_branch1 and (not tf_branch2) and len(ipeaks)==1:
#                     tf_branch2 = True
#                     g_branch2.append(g)
#                     kre_branch2.append(kre)
#             elif not tf_bound:
#                 tf_bound = True
#                 g_bound.append(g)
#                 kre_bound.append(kre)
#     else:
#         xmax_mat[:,j] = 1
# t1 = timeit.default_timer()
# print('calculation time:', t1-t0)
# with open('./data/figure5e_'+str(run_id)+'.npz','wb') as file1:
#     np.savez(file1, xmax_mat, kre_ls, g_ls, kre_bound, g_bound,
#     kre_branch1, g_branch1, kre_branch2, g_branch2, xmax2_mat, t1-t0)
#
#
# plot from data
run_id = 8
flag_inset = False
if flag_inset:
    plt.rcParams.update({'font.size': 18})
    # plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
with np.load('./data/figure5e_'+str(run_id)+'.npz') as file1:
    xmax_mat = file1['arr_0']
    kre_ls = file1['arr_1']
    g_ls = file1['arr_2']
    kre_bound = file1['arr_3']
    g_bound = file1['arr_4']
    kre_branch1 = file1['arr_5']
    g_branch1 = file1['arr_6']
    kre_branch2 = file1['arr_7']
    g_branch2 = file1['arr_8']
    xmat2_mat = file1['arr_9']
    t01 = file1['arr_10']
print('runtime (hrs):', round(t01/3600,1))
if flag_inset:
    fig = plt.figure(figsize=(15,10.5))
else:
    fig = plt.figure(figsize=(10,7))
plt.imshow(xmax_mat, extent=[kre_ls[0],kre_ls[-1],g_ls[0],g_ls[-1]],
    aspect='auto', origin='lower')
if flag_inset:
    plt.plot(kre_bound, g_bound, 'y--', label=r'$g_c$', linewidth=5)
else:
    plt.plot(kre_bound, g_bound, 'y--', label=r'$g_c$', linewidth=3)
plt.plot(kre_branch1[kre_branch1<0], g_branch1[kre_branch1<0], 'r--', linewidth=3)
kre_max = 2*sqrt(2)-3
tf_1 = kre_ls<2*sqrt(2)-3 # g_kre_c < gc
g_non_diff = (1+kre_ls[tf_1])/(-4*kre_ls[tf_1])
plt.plot(kre_ls[tf_1], g_non_diff, 'k--', linewidth=2)
plt.colorbar()
plt.xlabel(r'$\hat{\kappa}_{re}$')
plt.ylabel(r'$g$')
if not flag_inset:
    plt.title(r'$x_{\max}$')
plt.legend()
plt.tight_layout()
fig.savefig('./figure/figure5e_'+str(run_id)+'.png', dpi=600, transparent=True)
