import numpy as np
import matplotlib.pyplot as plt

from cov_spectrum_random_netw import *
from numpy.random import randn
from numpy.linalg import eigvals, eigvalsh
from numpy import sqrt, zeros, ones, eye, linspace, round
import timeit

plt.rcParams.update({'font.size': 26})

def corr_xy(x,y,nt_rad):
    # time-lagged correlation function, cxy=cov(x(t),y(t-tau))
    # nt_rad is the half window size
    c = zeros(2*nt_rad+1)
    nx = len(x)
    x -= np.mean(x)
    y -= np.mean(y)
    for i in range(-nt_rad,nt_rad+1):
        if i>=0 and i<nx:
            c[i+nt_rad] = np.mean(x[i:]*y[:(nx-i)])
        elif i<0 and -i<nx:
            c[i+nt_rad] = np.mean(x[:(nx+i)]*y[-i:])
    return c

def cov_XY_bin(X,Y,nt_bin):
    # covariance of the averaged activity over bin
    # c_ij=cov(x_i(t),y_j(t-tau))
    # X,Y are 2D array neuron*time
    # nt_bin is the bin/window size
    nx,nt = X.shape
    ny,nt2 = Y.shape
    assert nt==nt2
    nbin = int(np.floor(nt/nt_bin))
    Xbin = np.mean(np.reshape(X[:,:nbin*nt_bin],(nx,nbin,nt_bin)),axis=2)
    Ybin = np.mean(np.reshape(Y[:,:nbin*nt_bin],(nx,nbin,nt_bin)),axis=2)
    Xbin -= np.outer(np.mean(Xbin,axis=1),ones(nbin))
    Ybin -= np.outer(np.mean(Ybin,axis=1),ones(nbin))
    C = Xbin @ Ybin.T
    return C


# parameters for the pre-saved simulation data,
# run_id =
# '1': g = 0.4, sigma=0.5, alpha=0.1
# '2': g = 0.6, sigma=1, alpha=0.1
# '3': g = 0.8, sigma=1, alpha=0.1
# '4': g = 1.2, sigma=1, alpha=0.1
# Due to Github.com file size limit, the H matrix in the data
# is truncated to have T=100 instead of actual T=10000 used.
# This will affect the histogram to look coaser.


run_id = '3'
flag_load_data = True
flag_load_fit_g = True
if not flag_load_data:
    g = 0.4
    n = 400
    tau = 1 # neuronal time constant
    sigma = 0.5 # white noise sd
    flag_phi = 'tanh'
    if flag_phi=='lin':
        phi = lambda x:x
    elif flag_phi=='tanh':
        phi = lambda x: np.tanh(x) # activation function
    else:
        assert 0
    T = 10000  # simulation length
    T0 = 50 # warm up time
    ntrial = 4 # repeats of T-length simulations to avoid memory overflow
    T_total = T*ntrial
    dt = 0.01 # simulation time step
    n_Dt = 10
    # dt = 0.1 # simulation time step
    # n_Dt = 1
    Dt = n_Dt*dt # recorded time step, resolution of the correlation function
    t_bin = 10 # length of the time window for long-time/zero-frequency covariance
    t_corr_rad = 10 # half length of correlation function window
    np.random.seed(0)
    J = randn(n,n)/sqrt(n)*g


    # simulation
    import time
    np.random.seed(int(time.time()))
    nT = int(round(T/Dt))
    alpha = n/(T_total/(t_bin))
    nt_bin = int(round(t_bin/Dt))
    print('alpha:', alpha)
    nt_corr_rad = int(round(t_corr_rad/Dt))
    t_win_ls = np.arange(-nt_corr_rad,nt_corr_rad+1)*Dt
    # memory check
    m1 = 2*np.exp((np.log(n*nT*4)/np.log(2) - 30)*np.log(2))
    print('memory (GB):', round(m1,4))
    assert m1<8
    H = zeros((n,nT)) # activity sampled at Dt
    H_avg = zeros((n,nT)) # activity averaged over Dt bin
    h = zeros(n)
    h_avg = zeros(n)
    corr_all_h = zeros(2*nt_corr_rad+1)
    corr_all_r = zeros(2*nt_corr_rad+1)
    Ch = zeros((n,n))
    Cr = zeros((n,n))
    # simulating dynamics
    for i in range(int(round(T0/dt))):
        h += dt/tau*(-h + J @ phi(h)) + sqrt(dt/tau)*sigma*randn(n)
    time_sim = 0
    time_cov = 0
    for k in range(ntrial):
        t0 = timeit.default_timer()
        for i in range(nT):
            h_avg = 0
            for j in range(n_Dt):
                h += dt/tau*(-h + J @ phi(h)) + sqrt(dt/tau)*sigma*randn(n)
                h_avg += h
            h_avg /= n_Dt
            H[:,i] = np.copy(h)
            H_avg[:,i] = np.copy(h_avg)
        t1 = timeit.default_timer()
        time_sim += t1-t0
        # population correlation function and covariance, cij=cov(x_i(t),x_j(t-tau))
        h_avg = np.mean(H,axis=0)
        r_avg = np.mean(phi(H),axis=0)
        corr_all_h += corr_xy(h_avg,h_avg,nt_corr_rad)
        corr_all_r += corr_xy(r_avg,r_avg,nt_corr_rad)
        # covariance matrix
        t0 = timeit.default_timer()
        Ch += cov_XY_bin(H_avg,H_avg,nt_bin)
        Cr += cov_XY_bin(phi(H_avg),phi(H_avg),nt_bin)
        t1 = timeit.default_timer()
        time_cov += t1-t0
    corr_all_h /= ntrial
    corr_all_r /= ntrial
    Ch /= ntrial
    Cr /= ntrial
    print('simulation time:', round(time_sim,5))
    print('covariance time:', round(time_cov,5))
    t0 = timeit.default_timer()
    eig_Ch = eigvalsh(Ch)
    eig_Cr = eigvalsh(Cr)
    t1 = timeit.default_timer()
    print('eig time:', round(t1-t0,5))
    # save simulation data
    with open('./data/nonlinear_sim'+str(run_id)+'.npz','wb') as file1:
        np.savez(file1,g,n,tau,sigma,flag_phi,T,ntrial,dt,n_Dt,t_bin,
        t_corr_rad,J,Ch,Cr,corr_all_h,corr_all_r,H,H_avg,time_sim)
else:
    # load simulation data
    with np.load('./data/nonlinear_sim'+str(run_id)+'.npz') as file1:
        g = file1['arr_0']
        n = file1['arr_1']
        tau = file1['arr_2']
        sigma = file1['arr_3']
        flag_phi = file1['arr_4']
        T = file1['arr_5']
        ntrial = file1['arr_6']
        dt = file1['arr_7']
        n_Dt = file1['arr_8']
        t_bin = file1['arr_9']
        t_corr_rad = file1['arr_10']
        J = file1['arr_11']
        Ch = file1['arr_12']
        Cr = file1['arr_13']
        corr_all_h = file1['arr_14']
        corr_all_r = file1['arr_15']
        H = file1['arr_16']
        H_avg = file1['arr_17']
        time_sim = file1['arr_18']
    if flag_phi=='lin':
        phi = lambda x:x
    elif flag_phi=='tanh':
        phi = lambda x: np.tanh(x) # activation function
    else:
        assert 0
    T_total = T*ntrial
    Dt = n_Dt*dt
    nT = int(round(T/Dt))
    alpha = n/(T_total/(t_bin))
    nt_bin = int(round(t_bin/Dt))
    print('alpha:', alpha)
    print('T total:', T_total)
    print('simulation time:', round(time_sim,5))
    nt_corr_rad = int(round(t_corr_rad/Dt))
    t_win_ls = np.arange(-nt_corr_rad,nt_corr_rad+1)*Dt
    t0 = timeit.default_timer()
    eig_Ch = eigvalsh(Ch)
    eig_Cr = eigvalsh(Cr)
    t1 = timeit.default_timer()
    print('eig time:', round(t1-t0,5))



# plots
file_pre = './figure/nonlinear_sim_'+str(flag_phi)
file_pre += '_g'+str(g)+'_N'+str(n)+'_s'+str(int(sigma*10))
file_pre += '_T'+str(T_total)+'_bin'+str(t_bin)
file_pre += '_'

# combined histogram
plt.rcParams.update({'font.size': 26})
fig = plt.figure(1, figsize=(8,4))
plt.clf()
ax1 = plt.subplot(111)
hlin = np.reshape(H,[-1])
pbin,xbin = np.histogram(hlin,150, density=True)
xbin = (xbin[1:]+xbin[:-1])/2
dx = xbin[1]-xbin[0]
xbin = np.r_[xbin[0]-dx,xbin, xbin[-1]+dx]
pbin = np.r_[0,pbin,0]
plt.plot(xbin,pbin, linewidth=2,label='h')
rlin = np.reshape(phi(H),[-1])
pbin,xbin = np.histogram(rlin,150, density=True)
xbin = (xbin[1:]+xbin[:-1])/2
dx = xbin[1]-xbin[0]
xbin = np.r_[xbin[0]-dx,xbin, xbin[-1]+dx]
pbin = np.r_[0,pbin,0]
plt.plot(xbin,pbin,linestyle='--',linewidth=2,label='r')
plt.legend(loc='upper left')
color = 'k'
ax1.set_ylabel('probability', color=color)
ax1.tick_params(axis='y', labelcolor=color)
x0,x1 = [-2.5,2.5]
plt.xlim([x0,x1])
# nonlinear function overlay
ax2 = ax1.twinx()
color = 'r'
xls = linspace(x0,x1,400)
ax2.plot(xls,phi(xls), linewidth=4,color=color)
ax2.set_ylabel(r'$\phi(x)$', color=color)
ax2.tick_params(axis='y', labelcolor=color)
# effective g estimate
g_avg_dphi = np.mean(dphi(hlin))*g
plt.title(r'$g \langle \phi^\prime (h) \rangle = '+str(round(g_avg_dphi,2))+'$')
fig.tight_layout()
plt.savefig(file_pre+'hr_dist.png',
    dpi=300, transparent=True)
h_sd_avg = np.std(hlin)
r_sd_avg = np.std(rlin)


# activity trace
n_plot = 4
T_plot = 20
nT_plot = int(round(T_plot/Dt))
plt.figure(2,figsize=(10,6))
t_ls = Dt*np.arange(nT_plot)
plt.subplot(211)
for i in range(n_plot):
    plt.plot(t_ls, H[i,:nT_plot],label=str(i))
plt.xlabel('time')
plt.legend()
plt.ylabel('hi')
plt.subplot(212)
for i in range(n_plot):
    plt.plot(t_ls, phi(H[i,:nT_plot]),label=str(i))
plt.xlabel('time')
plt.legend()
plt.ylabel('ri')
plt.savefig(file_pre+'trace_example.png',
    dpi=300, transparent=True)

# correlation function
plt.figure(3)
plt.plot(t_win_ls, corr_all_h/corr_all_h[nt_corr_rad],linewidth=1,label='h')
plt.xlim([-10,10])
plt.ylim([-0.1,1.1])
plt.plot(t_win_ls, corr_all_r/corr_all_r[nt_corr_rad],linewidth=1,label='r')
plt.legend()
plt.xlabel('time')
plt.title('population correlation function (normalized)')
plt.savefig(file_pre+'corr_all.png',
    dpi=300, transparent=True)

# eigenvalues
plt.rcParams.update({'font.size': 30})
if not flag_load_fit_g:
    t0 = timeit.default_timer()
    gh_h, s2h_h, cost = fit_cdf_g_a0(eig_Ch, alpha, cost='CvM')
    gh_r, s2h_r, cost = fit_cdf_g_a0(eig_Cr, alpha, cost='CvM')
    t1 = timeit.default_timer()
    time_fit_gh = t1-t0
    with open('./data/nonlinear_sim'+run_id+'_fit.npz','wb') as file1:
        np.savez(file1,gh_h,s2h_h,gh_r,s2h_r,time_fit_gh)
else:
    with np.load('./data/nonlinear_sim'+run_id+'_fit.npz') as file1:
        gh_h = file1['arr_0']
        s2h_h = file1['arr_1']
        gh_r = file1['arr_2']
        s2h_r = file1['arr_3']
        time_fit_gh = file1['arr_4']
print('fit g-a0 time:', round(time_fit_gh,5))

# hist adjustment for better visualization
if g==0.8:
    nbin_hist = 120
elif g==1.2:
    nbin_hist = 200
else:
    nbin_hist = 60
plt.figure(4,figsize=(8,6))
plt.clf()
plt.hist(eig_Ch/np.mean(eig_Ch), nbin_hist, density=True, label='N='+str(n))
x, px = pdf_g_a(gh_h, alpha, nx=1000, normed=True)
line = plt.plot(x, px, linewidth=1.5, label='gh-a, gh='+str(round(gh_h,2)))
plt.plot([x[0],x[-1]], [0,0], '.', color=line[0].get_color(), markersize=5)
# # time-sampled theory
# x, px = pdf_g_a(g, alpha, nx=1000, normed=True)
# line = plt.plot(x, px, linestyle='--',linewidth=1.5, label=r'$g='+str(g)+'$')
# plt.plot([x[0],x[-1]], [0,0], '.', color=line[0].get_color(), markersize=10)
# # exact cov theory
# x, px = pdf_g(g, nx=1000, normed=True)
# line = plt.plot(x, px, linewidth=1.5, label='g theory')
# plt.plot([x[0],x[-1]], [0,0], '.', color=line[0].get_color(), markersize=10)
plt.legend()
plt.title('g='+str(g)+', sd='+str(round(h_sd_avg,2)))
plt.savefig(file_pre+'eig_h.png',
    dpi=300, transparent=True)
plt.figure(4,figsize=(8,6))
plt.clf()
plt.hist(eig_Cr/np.mean(eig_Cr), nbin_hist, density=True, label='N='+str(n))
x, px = pdf_g_a(gh_r, alpha, nx=1000, normed=True)
line = plt.plot(x, px, linewidth=1.5, label=r'$\hat{g}='+str(round(gh_r,2))+'$')
plt.plot([x[0],x[-1]], [0,0], '.', color=line[0].get_color(), markersize=10)
# # time-sampled theory
# x, px = pdf_g_a(g, alpha, nx=1000, normed=True)
# line = plt.plot(x, px, linestyle='--',linewidth=1.5, label=r'$g='+str(g)+'$')
# plt.plot([x[0],x[-1]], [0,0], '.', color=line[0].get_color(), markersize=10)
# # exact cov theory
# x, px = pdf_g(g, nx=1000, normed=True)
# line = plt.plot(x, px, linewidth=1.5, label='g theory')
# plt.plot([x[0],x[-1]], [0,0], '.', color=line[0].get_color(), markersize=10)
if g==1.2:
    plt.xlim([0,12]) # cropped view
plt.legend()
plt.title(r'$g='+str(g)+'$, $\sigma='+str(sigma)+'$')
plt.savefig(file_pre+'eig_r.png',
    dpi=300, transparent=True)
