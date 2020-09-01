import numpy as np
import matplotlib.pyplot as plt
from cov_spectrum_random_netw import (pdf_g, pdf_g_x, cdf_g, fit_cdf_g,
    support_g)
from cov_spectrum_random_netw import J2C, C2R
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import sqrt, zeros, ones, eye, linspace, dot, outer, logical_and, logical_not
import timeit

# created by Yu Hu (mahy@ust.hk), Aug 2020

plt.rcParams.update({'font.size': 18})


# Fitting iid Gaussian theory, unknown g and sigma
g = 0.6
s02 = 1
s12 = 17
s22 = 15
N = 200
J = g * randn(N,N)/np.sqrt(N)
u1 = randn(N)
u1 /= np.linalg.norm(u1)
u2 = randn(N)
u2 /= np.linalg.norm(u2)
C = J2C(J)*s02
C += outer(u1,u1)*s12 + outer(u2,u2)*s22
eig_C = np.sort(eigvalsh(C))

t0 = timeit.default_timer()
niter = 10
tf_in = ones(N, dtype=bool)
for t in range(niter):
    gh, s02h = fit_cdf_g(eig_C[tf_in], cost='CvM')
    x12h = support_g(gh)
    tf_in_new = logical_and(eig_C <=x12h[1], eig_C >=x12h[0])
    if (tf_in_new == tf_in).all():
        print('fitting converged at trial '+str(t+1))
        break
    else:
        tf_in = tf_in_new
t1 = timeit.default_timer()
print('fitting time:', t1-t0)
print('g:', g, 'gh:', gh)
print('s02:', s02, 's02h:', s02h)

fig = plt.figure(figsize=(9,4))
hist_label = r'$N='+str(N)+',\; g='+str(g)+'$'
plt.hist(eig_C, 150, density=True, label=hist_label)
x, px = pdf_g(gh, nx=2000)
line_label = r'fit $\hat{g}='+'{:.3f}'.format(round(gh, 2))+'$'
line = plt.plot(x*s02h, px/s02h, 'r', label=line_label, linewidth=1)
plt.plot([x[0]*s02h,x[-1]*s02h], [0,0], '.', color=line[0].get_color(), markersize=10)
tf_out = logical_not(tf_in)
plt.scatter(eig_C[tf_out], 0.15*ones(np.sum(tf_out)), 40, marker='x', color='m', label='outlier')
plt.legend()
plt.xlabel('cov eigenvalues')
plt.ylabel('probabilty')
plt.ylim(bottom=0)
plt.tight_layout()
fig.savefig('./figure/figure7_6', dpi=600, transparent=True)

fig = plt.figure(figsize=(8,5))
plt.plot(range(N), eig_C[::-1],'.-',markersize=10)
plt.tight_layout()
fig.savefig('./figure/figure7_6b', dpi=600, transparent=True)


# # test across trials
# ntrial = 10 #100
# g = 0.6
# s02 = 1
# s12 = 17
# s22 = 15
# N = 200
# print('ntrial:', ntrial)
# print('s1,s2:', s12, s22)
# t0 = timeit.default_timer()
# n_correct = 0
# n_correct1 = 0
# gh_ls = zeros(ntrial)
# for s in range(ntrial):
#     flag_converge = False
#     J = g * randn(N,N)/np.sqrt(N)
#     u1 = randn(N)
#     u1 /= np.linalg.norm(u1)
#     u2 = randn(N)
#     u2 /= np.linalg.norm(u2)
#     C = J2C(J)*s02
#     C += outer(u1,u1)*s12 + outer(u2,u2)*s22
#     eig_C = eigvalsh(C)
#     niter = 10
#     tf_in = ones(N, dtype=bool)
#     for t in range(niter):
#         gh, s02h = fit_cdf_g(eig_C[tf_in], cost='CvM')
#         x12h = support_g(gh)
#         tf_in_new = logical_and(eig_C <=x12h[1], eig_C >=x12h[0])
#         if (tf_in_new == tf_in).all():
#             # print('fitting converged at trial '+str(t+1))
#             flag_converge = True
#             break
#         else:
#             tf_in = tf_in_new
#     if flag_converge and tf_in[-1] == False and tf_in[-2] == False:
#         n_correct += 1
#     elif flag_converge and tf_in[-1] == False:
#         n_correct1 += 1
#     gh_ls[s] = gh
# t1 = timeit.default_timer()
# print('fitting time:', t1-t0)
# print('fraction correct:', n_correct/ntrial)
# print('fraction correct 1 outlier:', n_correct1/ntrial)
# print('sqrt(MSE):', sqrt(np.mean((gh_ls-g)**2)))
