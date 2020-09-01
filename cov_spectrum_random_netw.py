import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as opt

# created by Yu Hu (mahy@ust.hk), Aug 2020

# normed=True if the mean is scaled to 1 (for fitting to empirical eigs)
# iid Gaussian random connectivity
def mu_g(g):
    # mean of cov eigs
    return 1/(1-g**2)

def support_g(g, normed=False):
    # exact support
    z = 1. / g
    xz12 =  np.zeros(2)
    xz12[0] = 1./8/z**2 * (-1 + 20*z**2 + 8*z**4 - (1 + 8*z**2)**(3./2))
    xz12[1] = 1./8/z**2 * (-1 + 20*z**2 + 8*z**4 + (1 + 8*z**2)**(3./2))
    x12 = xz12 * g**2
    xi12 = 1. / x12
    xi12 = xi12[::-1]
    mu = mu_g(g)
    if normed:
        return xi12/mu
    else:
        return xi12

def pdf_g_x(x,g, normed=False):
    # return the prob density for a list of x
    x = np.array(x)
    mu = mu_g(g)
    if normed:
        x = x * mu
    n = np.size(x)
    p = np.zeros(n)
    xi12 = support_g(g)
    xz21 = 1/xi12 / g**2
    xz_all = 1/np.array(x) /g**2
    # inside the support
    tf_in = np.logical_and(xz_all>xz21[1], xz_all<xz21[0])
    xz = xz_all[tf_in]

    z = 1. / g
    A = (1-z**2) / xz
    B = z**2 / xz
    P = A - 1./3
    Q = B + A/3 - 2./27
    D = Q**2/4 + P**3/27
    S = (Q/2 + np.sqrt(D)) ** (1./3)
    T = (Q/2 - np.sqrt(D)) ** (1./3)
    p_nu = np.sqrt(3)/2 * (S - T) / np.pi

    p_nu /= g**2
    x_in = xz * g**2
    p_nui = p_nu * (x_in**2)
    p[tf_in] = p_nui
    if normed:
        return p*mu
    else:
        return p

def pdf_g(g, nx=1000, normed=False):
    # return a list of x and probability density
    # the endpoints gives the exact support (xi12)
    # the discritization (nx points) is based on evenly dividing 1/eig
    xi12 = support_g(g)
    xz21 = 1/xi12 / g**2
    xz = np.linspace(xz21[1], xz21[0], nx)
    x = 1 / (xz * g**2)
    x = x[::-1]
    p = pdf_g_x(x,g)
    mu = mu_g(g)
    if normed:
        return x/mu, p*mu
    else:
        return x, p

def cdf_g_x(x,g, normed=False):
    # cdf value at given point(s) x
    mu = mu_g(g)
    if normed:
        x = x * mu
    id_sort = np.argsort(x)
    x_sort = np.sort(x)
    x12 = support_g(g)
    m = np.size(x)
    P = np.zeros(m)
    for i,xi in enumerate(x_sort):
        if xi <= x12[0]:
            P[i] = 0
        elif xi >= x12[1]:
            P[i] = 1
        else:
            if i>0:
                P[i] = P[i-1] + integrate.quad(lambda x: pdf_g_x(x,g),
                max(x12[0], x_sort[i-1]), xi)[0]
            else:
                P[i] = integrate.quad(lambda x: pdf_g_x(x,g), x12[0], xi)[0]
    P_o = np.zeros(m) # original order
    P_o[id_sort] = P
    return P_o

def cdf_g(g, nx=40, normed=False):
    # x, cdf pairs spanning the support
    # the discritization (nx points) is based on evenly dividing 1/eig
    xi12 = support_g(g)
    xz21 = 1/xi12 / g**2
    xz = np.linspace(xz21[1], xz21[0], nx)
    x = 1 / (xz * g**2)
    x = x[::-1]
    P = cdf_g_x(x,g)
    mu = mu_g(g)
    if normed:
        return x/mu, P
    else:
        return x, P

def pdf2cdf(x, parameters, fmu, fsupport, fpdf_x, normed=False):
    # cdf value at given point(s) x
    # parameters is a tuple
    # fpdf_x is the pdf function for points x
    mu = fmu(*parameters)
    if normed:
        x = x * mu
    id_sort = np.argsort(x)
    x_sort = np.sort(x)
    x12 = fsupport(*parameters)
    m = np.size(x)
    P = np.zeros(m)
    for i,xi in enumerate(x_sort):
        if xi <= x12[0]:
            P[i] = 0
        elif xi >= x12[1]:
            P[i] = 1
        else:
            if i>0:
                P[i] = P[i-1] + integrate.quad(lambda x: fpdf_x(x,*parameters),
                max(x12[0], x_sort[i-1]), xi)[0]
            else:
                P[i] = integrate.quad(lambda x: fpdf_x(x,*parameters), x12[0], xi)[0]
    P_o = np.zeros(m) # original order
    P_o[id_sort] = P
    return P_o








# theory for Gaussian random connectivity with reciprocal motifs
def mu_g_kre(g,kre):
    # mean of cov eigs
    th = g**2*(1+kre)
    mu = 2*th - 1 + np.sqrt(1+4*(g**2 - th))
    mu /= 2*(g**2 - th**2)
    return mu

def dim_g_kre(g,kre):
    # relative dimension D/N
    mu = mu_g_kre(g,kre)
    th = g**2*(1+kre)
    D = mu*(2*g**2*mu+1) - 2*th*mu*(th*mu+1)
    D /= (th*mu+1)**2 * (g**2*mu+1)
    return D

def P_branch_g_kre(g,kre):
    # branch points of P = (1/g-J0)^T (1/g-J0))
    # use the uncorrelated case as reference
    # 4 points for 0<kre<1, 2 points otherwise
    # first 2 corresponds to the support of P
    if g * (1+kre) > 1:
        raise ValueError('unstable connectivity')
    z = 1/g
    if kre == 0:
        x12_C = support_g(g, normed=False)
        x12_P = 1/(g**2*x12_C)
        x12_P = x12_P[::-1]
        return x12_P
    elif kre ==1:
        return np.array([(z-2)**2, (z+2)**2])
    elif kre == -1:
        return np.array([(z**2), (z**2+4)])
    else:
        t = kre
        a = 1/g
        p = np.zeros(5)
        p[4] = 1
        p[3] = 5 + 3*t
        p[1] = 7 + 15*t + 9*t**2 + t**3 - 4*a**2*(1 + t)
        p[2] = 9 + 12*t + 3*t**2 - 2*a**2*(1 + t)
        p[0] = 2 + 6*t + 6*t**2 + 2*t**3 - 2*a**2*(1 + t)
        D_ls = np.roots(p)
        f_D = lambda d: -1/d + 1/(1+d) + a**2/(1+d*(1+t))**2
        x_ls = f_D(D_ls)
        if kre > 0:
            x_ls = np.sort(x_ls.real)
            y12 = np.zeros(4)
            y12[0] = x_ls[0]
            xmid = np.mean(x_ls[1:3])
            beta = 1 + kre
            eta = 1/g
            p = [xmid*beta**2, beta*(beta+2)*xmid, xmid*(2*kre+3) - eta**2 + beta**2,
            xmid - eta**2 + 2*beta, 1]
            r = np.roots(p)
            if np.sum(np.abs(r.imag)>1e-10) == 0: # test if two distributions cross
                y12[1] = x_ls[1]
            else:
                y12[1] = x_ls[2]
            y12[2:4] = np.sort(list(set(x_ls) - set(y12[:2])))
        else:
            i_ls = np.argsort(np.abs(x_ls.imag))
            y12 = np.sort(x_ls[i_ls[:2]].real)
        return y12

def support_g_kre(g,kre, normed=False):
    ybranch = P_branch_g_kre(g,kre)
    x12 = 1/ybranch[:2]/g**2
    x12 = x12[::-1]
    if normed:
        return x12/mu_g_kre(g,kre)
    else:
        return x12

def pdf_P_g_kre_x(x,g,kre):
    # P = (1/g-J0)^T (1/g-J0))
    # return the prob density for a list of x
    x = np.array(x)
    if np.size(x)>1:
        id_sort = np.argsort(x)
        x = x[id_sort]
    n = np.size(x)
    px = np.zeros(n)
    pxs = np.zeros(n)
    x12 = P_branch_g_kre(g,kre)
    eta = 1/g
    beta = 1+kre
    # inside the support
    tf_in = np.logical_and(x>x12[0], x<x12[1])
    if np.sum(tf_in) > 0:
        x_in = x[tf_in]
        p_in = np.zeros(len(x_in))
        nx_in = len(x_in)
        y4re = np.zeros((4,nx_in))
        y4im = np.zeros((4,nx_in))
        for i, xi in enumerate(x_in):
            p = [xi*beta**2, beta*(beta+2)*xi, xi*(2*kre+3) - eta**2 + beta**2,
                xi - eta**2 + 2*beta, 1]
            r = np.roots(p)
            y4re[:, i] = r.real
            y4im[:, i] = r.imag
        if kre<=0 or kre==1 or x12[2] >= x12[1]: # no crossing
            y = np.amax(y4im, axis=0)
        else:
            id_re_max = np.argmax(y4re, axis=0)
            y = np.abs(y4im[id_re_max, range(nx_in)])
        pxs[tf_in] = y/np.pi
    if np.size(x)>1:
        px[id_sort] = pxs
    return px


def pdf_g_symm_x(x,g, normed=False):
    x = np.array(x)
    mu = mu_g_kre(g,1)
    if normed:
        x = x * mu
    n = np.size(x)
    p = np.zeros(n)
    x12 = [(1+2*g)**(-2), (1-2*g)**(-2)]
    tf_in = np.logical_and(x>x12[0], x<x12[1])
    x0 = x[tf_in]
    p0 = np.sqrt((4*g**2-1)*x0 - 1 + 2*np.sqrt(x0)) / (4*np.pi*g**2* x0**2)
    p[tf_in] = p0
    if normed:
        return p*mu
    else:
        return p

def pdf_g_asymm_x(x,g, normed=False):
    # anti-symmetric random connectivity
    import math
    x = np.array(x)
    mu = mu_g_kre(g,-1)
    if normed:
        x = x * mu
    n = np.size(x)
    p = np.zeros(n)
    x12 = [1/(1+4*g**2), 1]
    tf_in = np.logical_and(x>x12[0], x<x12[1])
    x0 = x[tf_in]
    p0 = np.sqrt((4*g**2+1)*x0-1) / (2*np.pi*g**2*x0**2*np.sqrt(1-x0))
    p[tf_in] = p0
    if normed:
        p = p*mu
    p[x==x12[1]] = math.inf
    return p

def pdf_g_kre_x(x,g,kre, normed=False):
    if kre == 1:
        px = pdf_g_symm_x(x,g, normed)
    elif kre == -1:
        px = pdf_g_asymm_x(x,g, normed)
    elif kre == 0:
        px = pdf_g_x(x,g, normed)
    else:
        x = np.array(x)
        mu = mu_g_kre(g,kre)
        if normed:
            x = x * mu
        y = 1/(x*g**2)
        py = pdf_P_g_kre_x(y,g,kre)
        px = py/x**2/g**2
    if normed:
        return px*mu
    else:
        return px

def pdf_g_kre(g, kre, nx=1000, normed=False):
    # return a list of x and probability density
    # the endpoints gives the support
    # the discritization (nx points) is based on evenly dividing 1/eig
    ybranch = P_branch_g_kre(g,kre)
    y = np.linspace(ybranch[0], ybranch[1], nx)
    x = 1 / (y*g**2)
    x = x[::-1]
    p = pdf_g_kre_x(x,g,kre)
    mu = mu_g_kre(g,kre)
    if normed:
        return x/mu, p*mu
    else:
        return x, p

def r_theta_length(a, b, r):
    # for normal matching
    r1 = 1-a
    r3 = 1+a
    D = b**2 + (b**2-r**2) * (b**2 - a**2)
    assert(D>=0)
    x1 = (-a+np.sqrt(D)) / (b**2 - a**2)
    x2 = (-a-np.sqrt(D)) / (b**2 - a**2)
    v0 = np.pi*a*b
    if r > r3:
        th1 = np.arccos((a*x1 - 1)/r)
        th2 = np.arccos((a*x2 - 1)/r)
        return 2*(th2-th1)*r / v0
    else:
        th1 = np.arccos((a*x1 - 1)/r)
        return 2*(np.pi-th1)*r / v0

def support_g_kre_normal(g,kre):
    a = g*(1+kre)
    b = g*(1-kre)
    r1 = 1 - a
    if b**2 - a**2 - a > 0: # tangent condition
        r2 = np.sqrt((b**2-a**2+1)*b**2 / (b**2-a**2))
    else:
        r2 = 1 + a
    x12 = np.array([1/r2**2, 1/r1**2])
    return x12

def pdf_g_kre_normal_x(x,g,kre):
    # a normal connectivity matrix with matching eig distribution
    a = g*(1+kre)
    b = g*(1-kre)
    if kre == 1:
        px = pdf_g_symm_x(x,g)
    elif kre == -1:
        px = pdf_g_asymm_x(x,g)
    else:
        x12 = support_g_kre_normal(g,kre)
        px = np.zeros(len(x))
        tf_in = np.logical_and(x > x12[0], x < x12[1])
        if np.sum(tf_in)>0:
            arc_r_ls = [r_theta_length(a, b, 1/np.sqrt(xi)) for xi in x[tf_in]]
            arc_r_ls = np.array(arc_r_ls)
            px[tf_in] = arc_r_ls/2/x[tf_in]**(3/2)
    return px

def pdf_g_kre_normal(g,kre,nx=400):
    # a normal connectivity matrix with matching eig distribution
    x12 = support_g_kre_normal(g,kre)
    x = np.linspace(x12[0], x12[1], nx)
    px = pdf_g_kre_normal_x(x,g,kre)
    return x, px


def pdf_g_kre_normal_sim(n, g, kre):
    J = J_g_kre(n,g,kre)
    eig_J = np.linalg.eigvals(J)
    eig_Cn = np.abs(1-eig_J)**(-2)
    return eig_Cn

def cdf_g_kre_x(x,g,kre, normed=False):
    # cdf value at given point(s) x
    return pdf2cdf(x, (g,kre), mu_g_kre, support_g_kre, pdf_g_kre_x, normed)

def cdf_g_kre(g,kre, nx=40, normed=False):
    # x, cdf pairs spanning the support
    # the discritization (nx points) is based on evenly dividing 1/eig
    x12 = support_g_kre(g,kre)
    x = np.linspace(x12[0], x12[1], nx)
    P = cdf_g_kre_x(x,g,kre)
    mu = mu_g_kre(g,kre)
    if normed:
        return x/mu, P
    else:
        return x, P







# deterministic connectivity
def pdf_ring_NN(x,y, nx=1000):
    # x, y are left, right connections
    a = np.abs(x+y)
    b = np.abs(x-y)
    ab = b**2 - a**2 - a
    if ab < 0: # regular case
        x12 = [(1+a)**(-2), (1-a)**(-2)]
        x0 = np.linspace(x12[0], x12[1], nx)
        x = x0[1:-1]
        D = b**2*(1-a**2+b**2) + (a**2 - b**2)/x
        y = abs(a**2 - b**2)/(2*np.pi*x**2)/np.sqrt(D)/np.sqrt(
            (a**2-b**2 - np.sqrt(D)+a)*(a**2-b**2 + np.sqrt(D)-a))
    else: # folded case
        cmin = (b**2-a**2)/b**2/(b**2-a**2+1)
        x12 = [cmin, (1-a)**(-2)]
        x0 = np.linspace(x12[0], x12[1], nx)
        x = x0[1:-1]
        tf_1 = x <= (1+a)**(-2)
        D = b**2*(1-a**2+b**2) + (a**2 - b**2)/x
        y = abs(a**2 - b**2)/(2*np.pi*x**2)/np.sqrt(D)/np.sqrt(
            (a**2-b**2 - np.sqrt(D)+a)*(a**2-b**2 + np.sqrt(D)-a))
        D_1 = b**2*(1-a**2+b**2) + (a**2 - b**2)/x[tf_1]
        y[tf_1] += (b**2 - a**2)/(2*np.pi*x[tf_1]**2)/np.sqrt(D_1)/np.sqrt(
            (a**2-b**2 - np.sqrt(D_1)-a)*(a**2-b**2 + np.sqrt(D_1)+a))
    y0 = np.r_[0,y,0]
    return x0, y0

def pdf_ring_NN_a_2D(a, nx):
    # 2D symmetic case, a is the total connection i.e. a_i*2
    from scipy.special import ellipk
    a_i = a/2
    x12 = [(1+a)**(-2), (1-a)**(-2)]
    x0 = np.linspace(x12[0], x12[1], nx)
    x = x0[1:-1]
    y = 1 - (1 - 1/np.sqrt(x))**2/(4*a_i**2)
    y = ellipk(y) * x**(-3/2) /(2*a_i*np.pi**2)
    y0 = np.r_[0, y, 0]
    return x0, y0

def F_J0_d(x,d):
    from scipy.special import j0
    f_1 = lambda w: j0(w)**d * (np.cos(w*x))/np.pi
    return integrate.quad(f_1, 0, 1e3, limit=1000)[0]

def pdf_ring_NN_a_d(a, d, nx=1000):
    # symmetic case, a is the total connection i.e. a_i*d
    if d==1:
        return pdf_ring_NN(a/2,a/2, nx)
    elif d==2:
        return pdf_ring_NN_a_2D(a, nx)
    else:
        x12 = [(1+a)**(-2), (1-a)**(-2)]
        a_i = a/d
        x0 = np.linspace(x12[0], x12[1], nx)
        x = x0[1:-1]
        y = np.array([F_J0_d((1-1/np.sqrt(x1))/a_i, d) for x1 in x])
        y *= x**(-3/2)/(2*a_i)
        y0 = np.r_[0, y, 0]
        return x0, y0








# Fitting to empirical cov eigs
def D_KS(x, Fx):
    # x are samples, Fx is the function of theoretical cdf
    x = np.sort(x)
    n = np.size(x)
    Fn1 = np.arange(0, n)/n
    Fn2 = np.arange(1, n+1)/n
    F = Fx(x)
    D = max(np.max(np.abs(Fn1-F)), np.max(np.abs(Fn2-F)))
    return D

def D_KS_Px(x,Px, Fx):
    # x,Px are points on the cdf, Fx is the function of theoretical cdf
    n = np.size(x)
    F = Fx(x)
    D = np.max(np.abs(Px-F))
    return D

def D_CvM(x, Fx):
    # x are samples, Fx is the function of theoretical cdf
    x = np.sort(x)
    n = np.size(x)
    Fn = np.arange(1, n+1)/n - 1/(2*n)
    F = Fx(x)
    nD2 = np.sum((Fn-F)**2) + 1/(12*n)
    return np.sqrt(nD2/n)

def fit_cdf_g(x, cost='CvM'):
    # fitting to iid Gaussian J theory
    # x is a list of cov eigs
    mu_x = np.mean(x)
    x = x / mu_x
    L = {
    'CvM': lambda g: D_CvM(x, lambda x: cdf_g_x(x,g,normed=True)),
    'KS': lambda g: D_KS(x, lambda x: cdf_g_x(x,g,normed=True))
    }
    result = opt.minimize_scalar(L[cost], bounds=(0.01,0.99), method='Bounded')
    if not result.success:
        print('fitting unsuccessful')
    gh = result.x
    mu = mu_g(gh)
    sigma2 = mu_x/mu
    return gh, sigma2







# for finite size netw simulation
def relu(x):
    r = x * (x>=0)
    return r

def J2C(J):
    n,_ = J.shape
    C = np.eye(n) - J
    C = np.dot(C.T, C)
    C = np.linalg.inv(C)
    return C

def C2R(C):
    d = np.diag(C)
    d2 = 1/np.sqrt(d)
    R = np.dot(np.dot(np.diag(d2), C), np.diag(d2))
    return R

def J_g_kre(n, g, kre):
    # g < 1/(1+kre) for stability
    A = np.random.randn(n,n)/np.sqrt(2*n)
    if kre >=0:
        A = A + A.T
    else:
        A = A - A.T
    J = np.random.randn(n,n)/np.sqrt(n) * np.sqrt(1-abs(kre)) + np.sqrt(abs(kre)) * A
    J *= g
    return J

def mix_EI_netw(n, pe, pi, g):
    we = 1
    wi = we*pe/pi
    p = pe+pi
    assert(p<1)
    re = pe/p
    A = np.random.binomial(1,p, (n,n))
    Are = np.random.binomial(1,re, (n,n))
    Ae = A*Are # entrywise
    Ai = A*(1-Are)
    W = we*Ae - wi*Ai
    J  = W*g/np.sqrt(we**2*pe*n + wi**2*pi*n)
    return J

def EI_netw(n, fe, pe, pi, g, Ie=1):
    # Ie is the scale of balanced input: <1, inhibition dominance
    # Ie>1 is potentially unstable
    ne = int(round(n*fe))
    ni = n-ne
    fe = ne/n
    fi = 1 - fe
    we = 1
    wi = -we*pe*fe/(Ie*pi*fi)
    g0 = np.sqrt(n) * np.sqrt(fe*we**2*pe*(1-pe) + fi*wi**2*pi*(1-pi))
    we *= g/g0
    wi *= g/g0

    Je = we * np.random.binomial(1, pe, (n, ne))
    Ji = wi * np.random.binomial(1, pi, (n, ni))
    J = np.c_[Je, Ji]
    para = [ne, we, wi]
    return J, para

def EI_netw_eqvar(n, fe, pi, g):
    # requires fe>=0.5, slightly inhibitory dominant
    # this means pe <= pi
    ne = int(round(n*fe))
    ni = n-ne
    fe = ne/n
    fi = 1 - fe
    pe = pi*fi**2/fe**2
    we = 1
    wi = -we*np.sqrt(pe*(1-pe)/(pi*(1-pi)))
    assert(we*pe*fe + wi*pi*fi<=0)
    g0 = np.sqrt(n) * np.sqrt(fe*we**2*pe*(1-pe) + fi*wi**2*pi*(1-pi))
    we *= g/g0
    wi *= g/g0

    Je = we * np.random.binomial(1, pe, (n, ne))
    Ji = wi * np.random.binomial(1, pi, (n, ni))
    J = np.c_[Je, Ji]
    para = [ne, we, wi, pe, pi]
    return J, para
