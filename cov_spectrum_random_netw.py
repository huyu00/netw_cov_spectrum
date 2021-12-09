import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as opt

# created by Yu Hu (mahy@ust.hk)

# normed=True if the mean is scaled to 1 (for fitting to empirical eigs)
# iid Gaussian random connectivity
def mu_g(g):
    # mean of cov eigs
    return 1/(1-g**2)

def support_g(g, normed=False):
    # exact support
    x12 =  np.zeros(2)
    x12[0] = (2+5*g**2-g**4/4 - 1/4*g*(8+g**2)**(3/2))/(1-g**2)**3/2
    x12[1] = (2+5*g**2-g**4/4 + 1/4*g*(8+g**2)**(3/2))/(1-g**2)**3/2
    mu = mu_g(g)
    if normed:
        return x12/mu
    else:
        return x12

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

def quantile_g_p(p,g,lower_tail=True,normed=False):
    from scipy.optimize import root_scalar
    mu = mu_g(g)
    x12 = support_g(g)
    if lower_tail:
        f = lambda x: cdf_g_x([x],g) - p
        fp = lambda x: pdf_g_x(x,g)
    else:
        f = lambda x: 1 - cdf_g_x([x],g) - p
        fp = lambda x:  - pdf_g_x(x,g)
    sol = root_scalar(f,bracket=[x12[0],x12[1]],fprime=fp)
    x = sol.root
    if normed:
        x = x / mu
    return x

def quantile_g(g,N,normed=False):
    # for rank plot of network size N, decending order
    mu = mu_g(g)
    x12 = support_g(g)
    x_ls = np.zeros(N)
    for i in range(N):
        p = (N-i-1/2)/N
        if p<1/2:
            x_ls[i] = quantile_g_p(p,g,lower_tail=True)
        else:
            x_ls[i] = quantile_g_p(1-p,g,lower_tail=False)
    if normed:
        x_ls = x_ls / mu
    return x_ls





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







# time-sampled theory for iid Gaussian random connectivity
def mu_g_a(g,a):
    # mean of cov eigs
    if a>1:
        return a/(1-g**2)
    else:
        return 1/(1-g**2)

def support_g_a(g,a, normed=False):
    # exact support
    b = a
    n = 1/g
    A3 = 4*(-1. + n**2)**3
    A2 = +(1 - 20*n**2 - 8*n**4 + b**2*(-1 + n**2)**4 + 2*b*(-1 + n**2)**2*(-5 + 2*n**2))
    A1 = -2*(-2*n**2 + b**3*(-1 + n**2)**3 + b*(-1 + 15*n**2 + 4*n**4) + b**2*(4 - 10*n**2 + 5*n**4 + n**6))
    A0 = (4*b - 8*b**2 + 4*b**3 + 4*b*(-1 + n**2) - 8*b**2*(-1 + n**2) + 4*b**3*(-1 + n**2) + b**2*(-1 + n**2)**2
          -2*b**3*(-1 + n**2)**2 + b**4*(-1 + n**2)**2)
    z_ls = np.sort(np.roots([A3, A2, A1, A0])) / g**2
    x12 = np.zeros(2)
    x12[0] = z_ls[1]
    x12[1] = z_ls[2]
    mu = mu_g_a(g,a)
    if normed:
        return x12/mu
    else:
        return x12

def pdf_g_a_x(x,g,a, normed=False):
    # return the prob density for a list of x
    x = np.array(x)
    mu = mu_g_a(g,a)
    if normed:
        x = x * mu
    n = np.size(x)
    p = np.zeros(n)
    x12 = support_g_a(g,a)
    # inside the support
    tf_in = np.logical_and(x>x12[0], x<x12[1])
    x_in = x[tf_in]
    p_in = np.zeros(len(x_in))

    eta = 1/g
    for i, xi in enumerate(x_in):
        z = xi * g**2
        p_eq = [(z+a)**2, 2*z + 2*a + z*a*(1-abs(eta)**2) + a**2,
               1+z*(1-abs(eta)**2) + 2*a, 1]
        x_p =  -(np.roots(p_eq) + 1) / z
        p_in[i] = np.max(x_p.imag) / np.pi * g**2
    p[tf_in] = p_in
    if a>1:
        p *= a
    if normed:
        return p*mu
    else:
        return p

def pdf_g_a(g, a, nx=100, normed=False):
    # return a list of x and probability density
    # x endpoints are exact supports (pdf=0)
    x12 = support_g_a(g,a)
    x = np.linspace(x12[0], x12[1], nx)
    p = pdf_g_a_x(x,g,a)
    mu = mu_g_a(g,a)
    if normed:
        return x/mu, p*mu
    else:
        return x, p

def cdf_g_a_x(x,g,a, normed=False):
    # cdf value at given point(s) x
    return pdf2cdf(x, (g,a), mu_g_a, support_g_a, pdf_g_a_x, normed)

def cdf_g_a(g,a, nx=40, normed=False):
    # x, cdf pairs spanning the support
    # the discritization (nx points) is based on evenly dividing 1/eig
    x12 = support_g_a(g,a)
    x = np.linspace(x12[0], x12[1], nx)
    P = cdf_g_a_x(x,g,a)
    mu = mu_g_a(g,a)
    if normed:
        return x/mu, P
    else:
        return x, P








# space-sampled theory for iid Gaussian random connectivity
def mu_g_f(g,f):
    # mean of cov eigs
    return 1/(1-g**2)

def support_g_f(g,f, normed=False):
    # Bisection from the non-sample case support
    if f==1:
        return support_g(g,normed)
    else:
        max_iter = 100
        xtol = 1e-7
        ytol = 1e-10
        mu = mu_g(g)*g**2
        x12_g = support_g(g)*g**2
        r = (1-f)/f
        eta = 1/g
        pw = np.zeros(6)
        a = x12_g[0]
        b = mu
        for t in range(max_iter):
            z = (b+a)/2
            if (b-a) < xtol:
                break
            else:
                pw[5] =  -(f**3*r**2)
                pw[0] = -f**3*z**5
                pw[1] = f**2*z**3*(2 + f*(-3 + 2*r)*z)
                pw[4] = -f*r*(-1 + 2*f + eta**2 + f**2*(-2 + 3*r)*z)
                pw[2] = -f*z*(1 + z - eta**2*z + 2*f*(-2 + r)*z + f**2*(3 - 6*r + r**2)*z**2)
                pw[3] = (1 + f**2*(2 - 4*r)*z - f**3*(1 - 6*r + 3*r**2)*z**2 - f*(1 + (-1 + eta**2)*(-1 + r)*z))
                if np.max(np.roots(pw).imag) >  ytol:
                    b = z
                else:
                    a = z
        x1 = z
        a = mu
        b = x12_g[1]
        for t in range(max_iter):
            z = (b+a)/2
            if (b-a) < xtol:
                break
            else:
                pw[5] =  -(f**3*r**2)
                pw[0] = -f**3*z**5
                pw[1] = f**2*z**3*(2 + f*(-3 + 2*r)*z)
                pw[4] = -f*r*(-1 + 2*f + eta**2 + f**2*(-2 + 3*r)*z)
                pw[2] = -f*z*(1 + z - eta**2*z + 2*f*(-2 + r)*z + f**2*(3 - 6*r + r**2)*z**2)
                pw[3] = (1 + f**2*(2 - 4*r)*z - f**3*(1 - 6*r + 3*r**2)*z**2 - f*(1 + (-1 + eta**2)*(-1 + r)*z))
                if np.max(np.roots(pw).imag) >  ytol:
                    a = z
                else:
                    b = z
        x2 = z
        x12 = np.array([x1,x2]) / g**2
        if normed:
            return x12/mu
        else:
            return x12

def pdf_g_f_x(x,g,f, normed=False):
    # return the prob density for a list of x
    if f==1:
        return pdf_g_x(x,g, normed)
    else:
        x = np.array(x)
        mu = mu_g_f(g,f)
        if normed:
            x = x * mu
        n = np.size(x)
        p = np.zeros(n)
        x12 = support_g_f(g,f)
        # inside the support
        tf_in = np.logical_and(x>x12[0], x<x12[1])
        x_in = x[tf_in]
        p_in = np.zeros(len(x_in))
        eta = 1/g
        r = (1-f)/f
        pw = np.zeros(6)
        for i, xi in enumerate(x_in):
            z = xi * g**2
            pw[5] =  -(f**3*r**2)
            pw[0] = -f**3*z**5
            pw[1] = f**2*z**3*(2 + f*(-3 + 2*r)*z)
            pw[4] = -f*r*(-1 + 2*f + eta**2 + f**2*(-2 + 3*r)*z)
            pw[2] = -f*z*(1 + z - eta**2*z + 2*f*(-2 + r)*z + f**2*(3 - 6*r + r**2)*z**2)
            pw[3] = (1 + f**2*(2 - 4*r)*z - f**3*(1 - 6*r + 3*r**2)*z**2 - f*(1 + (-1 + eta**2)*(-1 + r)*z))
            p_in[i] = np.max(np.roots(pw).imag)/np.pi * g**2
        p[tf_in] = p_in
        if normed:
            return p*mu
        else:
            return p

def pdf_g_f(g, f, nx=100, normed=False):
    # return a list of x and probability density
    # x endpoints are supports (pdf=0)
    if f == 1:
        return pdf_g(g,nx,normed)
    else:
        x12 = support_g_f(g,f)
        x = np.linspace(x12[0], x12[1], nx)
        p = pdf_g_f_x(x,g,f)
        mu = mu_g_f(g,f)
        if normed:
            return x/mu, p*mu
        else:
            return x, p

def cdf_g_f_x(x,g,f, normed=False):
    # cdf value at given point(s) x
    if f == 1:
        return cdf_g_x(x,g,normed)
    else:
        return pdf2cdf(x, (g,f), mu_g_f, support_g_f, pdf_g_f_x, normed)

def cdf_g_f(g,f, nx=40, normed=False):
    # x, cdf pairs spanning the support
    if f == 1:
        return cdf_g(g,nx,normed)
    else:
        x12 = support_g_f(g,f)
        x = np.linspace(x12[0], x12[1], nx)
        P = cdf_g_f_x(x,g,f)
        mu = mu_g_f(g,f)
        if normed:
            return x/mu, P
        else:
            return x, P






# MP laws, one parameter a
def support_MP(a):
    x12 = np.array([(1-np.sqrt(a))**2, (1+np.sqrt(a))**2])
    return x12

def pdf_MP_x(x,a):
    # return the prob density for a list of x
    x = np.array(x)
    n = np.size(x)
    p = np.zeros(n)
    x12 = support_MP(a)
    # inside the support
    tf_in = np.logical_and(x>x12[0], x<x12[1])
    xz = x[tf_in]
    pz = np.sqrt((x12[1]-xz)*(xz-x12[0])) / (2*np.pi*a*xz)
    p[tf_in] = pz
    return p

def pdf_MP(a, nx=1000):
    # return a list of x and probability density
    x12 = support_MP(a)
    x = np.linspace(x12[0], x12[1], nx)
    p = pdf_MP_x(x,a)
    return x, p

def cdf_MP_x(x,a):
    # cdf value at given point(s) x
    id_sort = np.argsort(x)
    x_sort = np.sort(x)
    x12 = support_MP(a)
    m = np.size(x)
    P = np.zeros(m)
    for i,xi in enumerate(x_sort):
        if xi <= x12[0]:
            P[i] = 0
        elif xi >= x12[1]:
            P[i] = 1
        else:
            if i>0:
                P[i] = P[i-1] + integrate.quad(lambda x: pdf_MP_x(x,a),
                max(x12[0], x_sort[i-1]), xi)[0]
            else:
                P[i] = integrate.quad(lambda x: pdf_MP_x(x,a), x12[0], xi)[0]
    P_o = np.zeros(m) # original order
    P_o[id_sort] = P
    return P_o

def cdf_MP(a, nx=40):
    # x, cdf pairs spanning the support
    x12 = support_a(a)
    x = np.linspace(x12[0], x12[1], nx)
    P = cdf_MP_x(x,a)
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

def pdf_ring_NN_ac_d_x(x, d):
    # critical case
    # symmetic case, a is the total connection i.e. a_i*d
    # ai = 1/d
    if d==1:
        return x**(-5/4)/(2*np.pi * np.sqrt(2-1/np.sqrt(x)))
    elif d==2:
        from scipy.special import ellipk
        a_i = 1/d
        y = 1 - (1 - 1/np.sqrt(x))**2
        y = ellipk(y) * x**(-3/2) /(np.pi**2)
        return y
    else:
        y = np.array([F_J0_d((1-1/np.sqrt(x1))*d, d) for x1 in x])
        y *= x**(-3/2)*d/2
        return y




# outlier location: perturb C
def xmin_C_u_outlier(g):
    x12 = support_g(g)
    xmin = x12[1]**2/(x12[1]+ (3+np.sqrt(1+8/g**2))/(4*(1-g**2)))
    return xmin

def C_u_outlier(g, a):
    A4 = 1/a**3
    A3 = -3/a**2
    A2 = 3/a+2/(a**2 * g**2)
    A1 = -(1 + 1/(g**4 * a) + 3/(g**2 * a))
    A0 = 1/g**4 + 1/(a*g**4) + 1/g**2
    z4 = np.roots([A4,A3,A2,A1,A0])
    id_sort = np.argsort(np.abs(z4.imag))
    z4 = z4[id_sort]
    if z4[2].imag > z4[3].imag:
        z4 = z4[[0,1,3,2]]
    if z4[0].real > z4[1].real or z4[0].imag > z4[1].imag:
        z4 = z4[[1,0,2,3]]
    x1p = z4[0].real
    return x1p, z4

# orthogonal u,v
def xmin_J_uv_outlier(g):
    x12 = support_g(g)
    D12 = np.zeros(2)
    D12[0] = (3-np.sqrt(1+8/g**2))/(4*(1-g**2)) #C_inv support (correspond to x12)
    D12[1] = (3+np.sqrt(1+8/g**2))/(4*(1-g**2))
    xmin12 = [np.sqrt(x12[i])/abs(D12[i]) for i in range(2)] # left and right outlier
    xmin12 = np.array(xmin12)
    return xmin12

def J_uv_outlier(g, a):
    A = -a**2-2*g**2
    B = np.abs(a)*np.sqrt(a**2+4)
    x12p = np.zeros(2)
    x12p[0] = 4*a**2/(A-B)**2
    x12p[1] = 4*a**2/(A+B)**2
    return x12p


# identical u,v
def xmin_J_uu_outlier(g):
    x12 = support_g(g)
    D12 = np.zeros(2)
    D12[0] = (3-np.sqrt(1+8/g**2))/(4*(1-g**2)) #C_inv support (correspond to x12)
    D12[1] = (3+np.sqrt(1+8/g**2))/(4*(1-g**2))
    f_Dz = lambda D,z: g**2*z*D**2 + (z+np.sqrt(z))*D + 1
    xmin12 = [1/f_Dz(D12[i],1/x12[i]) for i in range(2)] # for x<0 and x>0
    xmin12 = np.array(xmin12)
    return xmin12

def J_uu_outlier(g,a):
    x12 = support_g(g)
    D12 = np.zeros(2)
    D12[0] = (3-np.sqrt(1+8/g**2))/(4*(1-g**2)) #C_inv support (correspond to x12)
    D12[1] = (3+np.sqrt(1+8/g**2))/(4*(1-g**2))
    if a<0:
        db = 0
        da = D12[0]
    else:
        db = D12[1]
        da = 1/(1-g**2)
    f_zD = lambda D: -1/(D*(g**2*D+1)) + 1/(g**2*D+1)**2
    L_D = lambda D: g**2*D**2*f_zD(D) + (f_zD(D)+np.sqrt(f_zD(D)))*D+1-1/a
    x12min = xmin_J_uu_outlier(g)
    if a<1 and (a<x12min[0] or a>x12min[1]):
        niter = 100
        tol = 1e-9
        for t in range(niter):
            dx = (da+db)/2
            if L_D(dx) > 0: #sign of L(db) depends on a
                db = dx
            else:
                da = dx
            if db-da < tol:
                dx = (da+db)/2
                break
        zx = f_zD(dx)
        return 1/zx
    else:
        return -1 # no outlier











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
    cost = result.fun
    return gh, sigma2, cost

def fit_cdf_g_a(x, cost='CvM'):
    # fitting to time-sampled iid Gaussian J theory
    # x is a list of cov eigs
    mu_x = np.mean(x)
    x = x / mu_x
    L = {
    'CvM': lambda parameters: D_CvM(x, lambda x: cdf_g_a_x(x,*(parameters),normed=True)),
    'KS': lambda parameters: D_KS(x, lambda x: cdf_g_a_x(x,*(parameters),normed=True))
    }
    result = opt.minimize(L[cost], x0=[0.5,0.5], bounds=([0.01,0.99], [0,10]))
    if not result.success:
        print('fitting unsuccessful')
    gh,ah = result.x
    mu = mu_g_a(gh,ah)
    sigma2 = mu_x/mu
    cost = result.fun
    return gh, ah, sigma2, cost

def fit_cdf_g_a0(x, a0, cost='CvM'):
    # fitting to space-sampled iid Gaussian J theory, given f
    # x is a list of cov eigs
    mu_x = np.mean(x)
    x = x / mu_x

    L = {
    'CvM': lambda g: D_CvM(x, lambda x: cdf_g_a_x(x,g,a0,normed=True)),
    'KS': lambda g: D_KS(x, lambda x: cdf_g_a_x(x,g,a0,normed=True))
    }
    result = opt.minimize_scalar(L[cost], bounds=(0.01,0.99), method='Bounded')
    if not result.success:
        print('fitting unsuccessful')
    gh = result.x
    mu = mu_g_a(gh,a0)
    sigma2 = mu_x/mu
    cost = result.fun
    return gh, sigma2, cost

def fit_cdf_g_f0(x, f0, cost='CvM'):
    # fitting to space-sampled iid Gaussian J theory, given f
    # x is a list of cov eigs
    mu_x = np.mean(x)
    x = x / mu_x

    L = {
    'CvM': lambda g: D_CvM(x, lambda x: cdf_g_f_x(x,g,f0,normed=True)),
    'KS': lambda g: D_KS(x, lambda x: cdf_g_f_x(x,g,f0,normed=True))
    }
    result = opt.minimize_scalar(L[cost], bounds=(0.01,0.99), method='Bounded')
    if not result.success:
        print('fitting unsuccessful')
    gh = result.x
    mu = mu_g_f(gh,f0)
    sigma2 = mu_x/mu
    return gh, sigma2, cost

def fit_cdf_g_f(x, cost='CvM'):
    # fitting to space-sampled iid Gaussian J theory
    # x is a list of cov eigs
    mu_x = np.mean(x)
    x = x / mu_x
    L = {
    'CvM': lambda parameters: D_CvM(x, lambda x: cdf_g_f_x(x,*(parameters),normed=True)),
    'KS': lambda parameters: D_KS(x, lambda x: cdf_g_f_x(x,*(parameters),normed=True))
    }
    result = opt.minimize(L[cost], x0=[0.5,0.5], bounds=([0.01,0.99], [0.001,1]))
    if not result.success:
        print('fitting unsuccessful')
    gh,fh = result.x
    mu = mu_g_f(gh,fh)
    sigma2 = mu_x/mu
    cost = result.fun
    return gh, fh, sigma2, cost

def fit_cdf_g_kre(x, cost='CvM'):
    # fitting to space-sampled iid Gaussian J theory
    # x is a list of cov eigs
    mu_x = np.mean(x)
    x = x / mu_x
    L = {
    'CvM': lambda parameters: D_CvM(x, lambda x: cdf_g_kre_x(x,*(parameters),normed=True)),
    'KS': lambda parameters: D_KS(x, lambda x: cdf_g_kre_x(x,*(parameters),normed=True))
    }
    result = opt.minimize(L[cost], x0=[0.5,-0.1], bounds=([0.01,0.99], [-0.99,1]))
    if not result.success:
        print('fitting unsuccessful')
    gh,kreh = result.x
    mu = mu_g_kre(gh,kreh)
    sigma2 = mu_x/mu
    cost = result.fun
    return gh, kreh, sigma2, cost


def fit_cdf_MP(x, cost='CvM'):
    # fitting to MP law
    # x is a list of cov eigs
    mu_x = np.mean(x)
    x = x / mu_x
    L = {
    'CvM': lambda a: D_CvM(x, lambda x: cdf_MP_x(x,a)),
    'KS': lambda a: D_KS(x, lambda x: cdf_MP_x(x,a))
    }
    result = opt.minimize_scalar(L[cost], bounds=(0.01,0.999), method='Bounded')
    if not result.success:
        print('fitting unsuccessful')
    ah = result.x
    mu = 1
    sigma2 = mu_x/mu
    cost = result.fun
    return ah, sigma2, cost



# for simulation
def relu(x):
    r = x * (x>=0)
    return r

def J2C(J):
    n,_ = J.shape
    C = np.eye(n) - J
    C = np.dot(C.T, C)
    C = np.linalg.inv(C)
    return C

# def JY2C(J):
#     n,_ = J.shape
#     y = solve(eye(n) - J, ones(n))
#     y[y<0] = 0
# #     y -= min(0,min(y))
#     C = solve(eye(n)-J, diag(sqrt(y)))
#     C = dot(C, C.T)
#     return C

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
