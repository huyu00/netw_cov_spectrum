import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x, cdf_g, fit_cdf_g,support_g_a,
    pdf_g_a, pdf_g_a_x, fit_cdf_g_a, dim_g_kre, support_g, pdf_g_kre, P_branch_g_kre,
    pdf_P_g_kre_x, fit_cdf_g, pdf_g_f, fit_cdf_g_f, fit_cdf_g_f0, fit_cdf_g_kre)
from cov_spectrum_random_netw import J2C, C2R, J_g_kre, mix_EI_netw, EI_netw, EI_netw_eqvar
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import sqrt, zeros, ones, eye, linspace, dot
import timeit

plt.rcParams.update({'font.size': 18})

# created by Yu Hu (mahy@ust.hk)

# how time-sampled edges change with a (beyond 1)
g_ls = [0.3,0.4,0.5,0.6,0.7]
na = 100
a_ls = linspace(0,4,na)
x1_a_g1 = lambda a: ((1+3*a)**(3/2)+1-9*a)*2/27
fig = plt.figure(figsize=(12,6))
plt1 = fig.add_subplot(121)
plt2 = fig.add_subplot(122)
for g in g_ls:
    x12_ls = zeros((2,na))
    for i,a in enumerate(a_ls):
        x12_ls[:,i] = support_g_a(g,a)
    plt1.plot(a_ls, x12_ls[0,:], label=r'$g='+str(g)+'$', linewidth=1.5)
    plt2.plot(a_ls, x12_ls[1,:], label=r'$g='+str(g)+'$',linewidth=1.5)
plt1.plot(a_ls, x1_a_g1(a_ls), 'k--', label=r'$g\rightarrow 1$', linewidth=1.5)
plt1.plot(1,0,'mo')
plt1.legend()
plt1.set_xlabel(r'$\alpha$')
# plt1.set_ylabel(r'$x_{-}$')
plt1.set_title('Left edge')
plt2.set_xlabel(r'$\alpha$')
# plt2.set_ylabel(r'$x_{+}$')
plt2.set_title('Right edge')
plt2.legend()
fig.tight_layout()
fig.savefig('./figure/figure7_g_a_edge.png', dpi=300)







# pdf of space-sampled iid Gaussian theory, log scale
g = 0.9
nx = 40000
f_ls = [0.1, 0.2, 0.4]
fig = plt.figure(figsize=(8,6))
x0_plot = 0.3
t0 = timeit.default_timer()
x, px = pdf_g(g, nx=nx)
x12_ls = []
x12c_ls = []
tf_plot = x>=x0_plot
line = plt.loglog(x[tf_plot],px[tf_plot], 'k',linewidth=1.5, label='non-sampled')
px2 = sqrt(3)/(2*np.pi) * x**(-5/3)
plt.loglog(x[tf_plot],px2[tf_plot], '--', linewidth=1.5, color = line[0].get_color())
x12_ls.append((x[0],x[-1]))
x12c_ls.append(line[0].get_color())
for f in f_ls:
    x, px = pdf_g_f(g,f, nx=nx)
    tf_plot = x>=x0_plot
    label_plot = r'$f=N_s/N='+str(f)+'$'
    line = plt.loglog(x[tf_plot],px[tf_plot], linewidth=1.5, label=label_plot)
    px2 = sqrt(3)/(2*np.pi) * x**(-5/3) * f**(-1/3)
    plt.loglog(x[tf_plot],px2[tf_plot], '--', linewidth=1.5, color = line[0].get_color())
    x12_ls.append((x[0],x[-1]))
    print(x[0])
    x12c_ls.append(line[0].get_color())
# for i,x12 in enumerate(x12_ls):
#     plt.plot(x12, [0,0], '.', color=x12c_ls[i], markersize=10)
t1 = timeit.default_timer()
print('computing pdf-g-f', t1-t0)
plt.title(r'$g='+str(g)+'$')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=1e-5)
plt.xlim(left=x0_plot)
plt.tight_layout()
fig.savefig('./figure/figure7c_log.png', dpi=300)






# pdf of time-sampled iid Gaussian theory, log scale
g = 0.8
# a_ls = [0.1, 0.3, 0.5, 0.7]
a_ls = [0.3, 0.6, 1, 1.5, 2]
# a_ls = [1, 1.5, 2, 3, 4]
flag_nonsample = True
flag_a1_line = False
nx = 40000
x0_plot = support_g(g)[0]/3

fig = plt.figure(figsize=(8,6))
t0 = timeit.default_timer()
x, px = pdf_g(g, nx=nx)
x12_ls = []
x12c_ls = []
if flag_nonsample:
    tf_plot = x>=x0_plot
    line = plt.loglog(x[tf_plot],px[tf_plot], 'k',linewidth=1.5, label='non-sampled')
    x12_ls.append((x[0],x[-1]))
    x12c_ls.append(line[0].get_color())
for a in a_ls:
    x, px = pdf_g_a(g,a, nx=nx)
    label_plot = r'$\alpha=N/M='+str(a)+'$'
    tf_plot = x>=x0_plot
    line = plt.loglog(x[tf_plot],px[tf_plot], linewidth=1.5, label=label_plot)
    x12_ls.append((x[0],x[-1]))
    x12c_ls.append(line[0].get_color())
    # if a == 1:
    #     px2 = sqrt(3)/(2*np.pi) * x[1:]**(-5/3)
    #     plt.loglog(x[1:],px2, '--', linewidth=1.5)
    # else:
    #     px2 = sqrt(3)/(2*np.pi) * x**(-5/3)
    #     if a>1:
    #         px2 *= a
    #     plt.loglog(x,px2, '--', linewidth=1.5)
    tf_plot = x>=x0_plot
    px2 = sqrt(3)/(2*np.pi) * x[tf_plot]**(-5/3)
    if a>1:
        px2 *= a
    if (not flag_a1_line) and a<=1:
        flag_a1_line = True
        plt.loglog(x[tf_plot],px2, 'k--', linewidth=1.5)
    elif a>1:
        plt.loglog(x[tf_plot],px2, '--', linewidth=1.5, color = line[0].get_color())

# for i,x12 in enumerate(x12_ls):
#     plt.plot(x12, [0,0], '.', color=x12c_ls[i], markersize=10)
t1 = timeit.default_timer()
print('computing pdf-g-a', t1-t0)
plt.title(r'$g='+str(g)+'$')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=1e-5)
plt.ylim(top=10)
plt.tight_layout()
fig.savefig('./figure/figure7a_ag1_log.png', dpi=300)




# pdf of time-sampled iid Gaussian theory
g = 0.4
a_ls = [0.1, 0.3, 0.5, 0.7]
# a_ls = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 4]
# a_ls = [1, 1.5, 2, 3, 4]
flag_nonsample = True

fig = plt.figure(figsize=(10,6))
t0 = timeit.default_timer()
x, px = pdf_g(g, nx=2000)
x12_ls = []
x12c_ls = []
if flag_nonsample:
    line = plt.plot(x,px, 'k--',linewidth=1.5, label='non-sampled')
    x12_ls.append((x[0],x[-1]))
    x12c_ls.append(line[0].get_color())
for a in a_ls:
    x, px = pdf_g_a(g,a, nx=2000)
    label_plot = r'$\alpha=N/M='+str(a)+'$'
    line = plt.plot(x,px, linewidth=1.5, label=label_plot)
    x12_ls.append((x[0],x[-1]))
    x12c_ls.append(line[0].get_color())
for i,x12 in enumerate(x12_ls):
    plt.plot(x12, [0,0], '.', color=x12c_ls[i], markersize=10)
t1 = timeit.default_timer()
print('computing pdf-g-a', t1-t0)
plt.title(r'$g='+str(g)+'$')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
# plt.ylim(top=2) # a>1 plot
plt.tight_layout()
# fig.savefig('./figure/figure7a_ag1.png', dpi=300)
fig.savefig('./figure/figure7a.png', dpi=300)








# Fitting to time-sampled iid Gaussian theory, unknown alpha and sigma
g = 0.4
a = 0.3
sigma2 = 1
N = 100
J = g * randn(N,N)/np.sqrt(N)
T = int(round(N / a))
Z = randn(N,T) / np.sqrt(T)
C = np.linalg.solve(np.eye(N) - J, Z)
C = np.dot(C, C.T) * sigma2
eig_C = eigvalsh(C)

t0 = timeit.default_timer()
gh, ah, s2h, cost = fit_cdf_g_a(eig_C, cost='CvM')
t1 = timeit.default_timer()
print('fitting time:', t1-t0)
print('g:', g, 'gh:', gh)
print('a:', a, 'ah:', ah)
print('sigma2:', sigma2, 's2h:', s2h)

fig = plt.figure(figsize=(8,6))
hist_label = r'$N='+str(N)+',\; g='+str(g)+'$, '+r'$\alpha='+str(a)+'$'
plt.hist(eig_C, 30, density=True, label=hist_label)
x, px = pdf_g_a(gh,ah, nx=2000)
line_label = r'fit $\hat{g}='+'{:.3f}'.format(round(gh, 2))+\
    '$, '+r'$\hat{\alpha}='+'{:.3f}'.format(round(ah, 2))+'$'
line = plt.plot(x*s2h, px/s2h, 'r', label=line_label, linewidth=2)
plt.plot([x[0]*s2h,x[-1]*s2h], [0,0], '.', color=line[0].get_color(), markersize=10)
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure7b', dpi=300)








# pdf of space-sampled iid Gaussian theory
g = 0.5
f_ls = [0.1, 0.3, 0.5, 0.7]
fig = plt.figure(figsize=(10,6))
t0 = timeit.default_timer()
x, px = pdf_g(g, nx=2000)
x12_ls = []
x12c_ls = []
line = plt.plot(x,px, 'k--',linewidth=1.5, label='non-sampled')
x12_ls.append((x[0],x[-1]))
x12c_ls.append(line[0].get_color())
for f in f_ls:
    x, px = pdf_g_f(g,f, nx=2000)
    label_plot = r'$f=N_s/N='+str(f)+'$'
    line = plt.plot(x,px, linewidth=1.5, label=label_plot)
    x12_ls.append((x[0],x[-1]))
    x12c_ls.append(line[0].get_color())
for i,x12 in enumerate(x12_ls):
    plt.plot(x12, [0,0], '.', color=x12c_ls[i], markersize=10)
t1 = timeit.default_timer()
print('computing pdf-g-f', t1-t0)
plt.title(r'$g='+str(g)+'$')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure7c.png', dpi=300)
