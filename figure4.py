import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import *
from cov_spectrum_random_netw import J2C, J_g_kre
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh
from numpy import sqrt, zeros, ones, linspace
import timeit

# created by Yu Hu (mahy@ust.hk), Aug 2020

plt.rcParams.update({'font.size': 18})

#compare with simulation
g = 0.3
kre = 0.4
x,px = pdf_g_kre(g,kre, nx=1000)
N = 400
J = J_g_kre(N,g,kre)
C = J2C(J)
eig_C = eigvalsh(C)
fig = plt.figure(figsize=(8,6))
plt.hist(eig_C, 40, density=True, label='N='+str(N))
line = plt.plot(x,px, linewidth=2, label='g='+str(g)+', $\hat{\kappa}_{re}='+str(kre)+'$')
plt.plot([x[0],x[-1]], [0,0], '.', color=line[0].get_color(), markersize=10)
plt.legend()
plt.ylim(bottom=0)
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.tight_layout()
fig.savefig('./figure/figure4a_1.png', dpi=600)

# compare with simulation
g = 0.3
kre = -0.4
x,px = pdf_g_kre(g,kre, nx=1000)
N = 400
J = J_g_kre(N,g,kre)
C = J2C(J)
eig_C = eigvalsh(C)
fig = plt.figure(figsize=(8,6))
plt.hist(eig_C, 40, density=True, label='N='+str(N))
line = plt.plot(x,px, linewidth=2, label='g='+str(g)+', $\hat{\kappa}_{re}='+str(kre)+'$')
plt.plot([x[0],x[-1]], [0,0], '.', color=line[0].get_color(), markersize=10)
plt.legend()
plt.ylim(bottom=0)
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.tight_layout()
fig.savefig('./figure/figure4a_2.png', dpi=600)


# fix g
g = 0.3
kre_ls = [-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7]
x12_ls = []
x12c_ls = []
fig = plt.figure(figsize=(10,6))
for i, kre in enumerate(kre_ls):
    x,px = pdf_g_kre(g,kre, nx=1000)
    line = plt.plot(x,px, linewidth=2, label=r'$\hat{\kappa}_{re}='+str(kre)+'$')
    x12_ls.append((x[0],x[-1]))
    x12c_ls.append(line[0].get_color())
for i,x12 in enumerate(x12_ls):
    plt.plot(x12, [0,0], '.', color=x12c_ls[i], markersize=10)
plt.ylim(bottom=0)
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.title('Fix $g='+str(g)+'$')
plt.tight_layout()
fig.savefig('./figure/figure4b_1.png', dpi=600)

# fix gr
gr = 0.3
kre_ls = [-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7]
x12_ls = []
x12c_ls = []
fig = plt.figure(figsize=(10,6))
for i, kre in enumerate(kre_ls):
    g = gr/(1+kre)
    x,px = pdf_g_kre(g,kre, nx=1000)
    line = plt.plot(x,px, linewidth=2, label=r'$\hat{\kappa}_{re}='+str(kre)+'$')
    x12_ls.append((x[0],x[-1]))
    x12c_ls.append(line[0].get_color())
for i,x12 in enumerate(x12_ls):
    plt.plot(x12, [0,0], '.', color=x12c_ls[i], markersize=10)
plt.ylim(bottom=0)
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.title('Fix $g_r='+str(gr)+'$')
plt.tight_layout()
fig.savefig('./figure/figure4b_2.png', dpi=600)


# power law as kre changes, near kre=1
# fix gr
gr = 0.995
kre_ls = [0.3,0.5,0.7,1]
flag_plot_powerlaw = True
flag_zoom = False
x12_ls = []
x12c_ls = []
if not flag_zoom:
    fig = plt.figure(figsize=(8,6))
else:
    fig = plt.figure(figsize=(8,6))
npoints = 4000
if not flag_zoom:
    ymin = 10**(-12.5)
else:
    ymin = 10**(-11)
for i, kre in enumerate(kre_ls):
    g = gr/(1+kre)
    x12 = support_g_kre(g,kre)
    if flag_zoom:
        x = np.exp(linspace(np.log(1e2), np.log(x12[1]),npoints+2))
        px = np.r_[pdf_g_kre_x(x[:-1],g,kre),ymin]
    else:
        x = np.exp(linspace(np.log(x12[0]), np.log(x12[1]),npoints+2))
        px = np.r_[ymin,pdf_g_kre_x(x[1:-1],g,kre),ymin]
    if kre<1 and kre>-1:
        line = plt.loglog(x,px, linewidth=1, label=r'$\kappa='+str(kre)+'$')
        tf_plot = x>=x[0]
        if i == 0 and flag_plot_powerlaw:
            px2 = sqrt(3)/(2*np.pi) * x[tf_plot]**(-5/3)*(1-kre)**(1/3)*(1+kre)
            plt.loglog(x[tf_plot],px2, '--', linewidth=1, color = line[0].get_color())
    elif kre==1:
        line = plt.loglog(x,px, linewidth=1,color='k', label=r'$\kappa='+str(kre)+'$')
        tf_plot = x>=x[0]
        if flag_plot_powerlaw:
            px2 = sqrt(2)/(np.pi) * x[tf_plot]**(-7/4)
            plt.loglog(x[tf_plot],px2, 'k--', linewidth=1)
plt.ylim(bottom=ymin)
# plt.ylim(top=10)
# plt.xlim(right=10**(2.5))
if not flag_zoom:
    plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
if not flag_zoom:
    plt.title('Fix $g_r='+str(gr)+'$')
plt.tight_layout()
fig.savefig('./figure/figure4b_powerlaw.png',
    dpi=300, transparent=True)




# power law as kre changes, near kre=-1
# fix gr
gr = 0.8
kre_ls = [-0.3,-0.5,-0.7,-0.9]
x12_ls = []
x12c_ls = []
fig = plt.figure(figsize=(8,6))
npoints = 4000
ymin = 10**(-7)
for i, kre in enumerate(kre_ls):
    g = gr/(1+kre)
    x12 = support_g_kre(g,kre)
    x = np.exp(linspace(np.log(x12[0]), np.log(x12[1]),npoints+2))
    px = np.r_[ymin,pdf_g_kre_x(x[1:-1],g,kre),ymin]
    if kre<1 and kre>-1:
        line = plt.loglog(x,px, linewidth=1.5, label=r'$\kappa='+str(kre)+'$')
        tf_plot = x>=x[0]
        if i == 0:
            # px2 = sqrt(3)/(2*np.pi) * x[tf_plot]**(-5/3)*(1-kre)**(1/3)*(1+kre)
            # plt.loglog(x[tf_plot],px2, '--', linewidth=1.5, color = line[0].get_color())
            xplot = np.exp(linspace(np.log(1e-2), np.log(10**(2.5)),400))
            px2 = sqrt(3)/(2*np.pi) * xplot**(-5/3)*(1-kre)**(1/3)*(1+kre)
            plt.loglog(xplot,px2, '--', linewidth=1, color = line[0].get_color())
plt.ylim(bottom=ymin)
# plt.ylim(top=10)
# plt.xlim(right=10**(2.5))
plt.legend(loc='upper right')
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.title('Fix $g_r='+str(gr)+'$')
plt.tight_layout()
fig.savefig('./figure/figure4b_powerlaw_negkappa.png',
    dpi=300, transparent=True)


# dimension as g, kre
ng = 400
nkre = 400
g_ls = np.linspace(0,1,ng+2)
g_ls = g_ls[1:-1]
kre_ls = np.linspace(-1,1,nkre+2)
kre_ls = kre_ls[1:-1]
D_mat = np.zeros((ng,nkre))
g_bound = []
kre_bound = []
t0 = timeit.default_timer()
for i,g in enumerate(g_ls):
    kre_max = 1/g -1
    tf_in = kre_ls < kre_max
    if sum(tf_in)>0:
        D_mat[i,tf_in] = dim_g_kre(g, kre_ls[tf_in])
    if kre_ls[-1] >= kre_max:
        g_bound.append(g)
        kre_bound.append(kre_max)
t1 = timeit.default_timer()
print('calculation time:', t1-t0)
fig = plt.figure(figsize=(10,6.25))
plt.imshow(D_mat, extent=[kre_ls[0],kre_ls[-1],g_ls[0],g_ls[-1]],
    aspect='auto', origin='lower')
plt.plot(kre_bound, g_bound, 'y--')
plt.colorbar()
# plt.xlabel(r'$\min\left(\frac{x}{x_{-}}, \sqrt{\frac{x_{+}}{x}}\right)$')
plt.xlabel(r'$\hat{\kappa}_{re}$')
plt.ylabel('g')
plt.title(r'$D/N$')
plt.tight_layout()
fig.savefig('./figure/figure4c_g.png', dpi=600)


# dimension as gr, kre
# curves
gr_ls = [0.2,0.4,0.6,0.8]
fig = plt.figure(figsize=(8,6))
for i,gr in enumerate(gr_ls):
    nkre = 400
    kre_ls = np.linspace(-1,1, nkre)
    tf_in = kre_ls>-1
    kre_ls_in = kre_ls[tf_in]
    g_ls = gr/(1+kre_ls_in)
    D_ls_in = [dim_g_kre(g_ls[i], kre_ls_in[i]) for i in range(sum(tf_in))]
    D_ls = np.r_[0,D_ls_in]
    plt.plot(kre_ls, D_ls,label='$g_r='+str(gr)+'$')
plt.xlabel(r'$\hat{\kappa}_{re}$')
plt.ylabel(r'$D/N$')
plt.axis('tight')
plt.xlim([-1,1])
plt.ylim([0,1])
plt.legend()
plt.tight_layout()
fig.savefig('./figure/figure4c_curve.png', dpi=600)
