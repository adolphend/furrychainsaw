import numpy as np
from cosamp import cosamp

def CoPRAM(y_abs, A, s, max_iter, tol1, tol2, z):
    m, n = A.shape
    z = np.zeros((n,1))
    p = np.zeros((m,1))
    error_hist = [[1], [2]]
    Marg = np.zeros((1,n))
    MShat = np.zeros(s)
    AShat = np.zeros((m,s))
    supp = np.zeros((1,n))
    y_abs2 = y_abs ** 2
    phi_sq = np.sum(y_abs2) / m
    phi = np.sqrt(phi_sq)

    Marg = (np.dot(y_abs2.T, A ** 2)).T / m
    Mg = np.sort(Marg)
    MgS = np.argsort(Marg)
    S0 = MgS[:s]
    Shat = np.sort(S0)
    AShat = A[:, Shat]
    print(AShat.shape)
    TAF = 'on'
    TAF = 'off'
    if TAF == 'on':
        card_Marg = np.ceil(m/6)
        M_eval = []
        for i in range(m):
            M_eval.append(y_abs[i]/np.linalg.normal(AShat[i,:]))
        M_eval = np.asarray(M_eval)
        MmS = np.argsort(M_eval)
        Io = Mms[1:card_Marg]
    else:
        card_Marg = m
        Io = np.arange(card_Marg)
    for i in range(card_Marg):
        ii = Io[i]
        MShat = MShat + np.dot(np.dot(y_abs[ii], (AShat[ii, :])).reshape((s, 1)), (AShat[ii, :]).reshape(1,s))
    svd_opt = 'power'
    svd_opt = 'svd'
    if svd_opt == 'svd':
        u, sigma, v = np.linalg.svd(MShat)
        v1 = u[:, 0]
    else:
        v1 = np.linalg.svdvals(MShat)
    v = np.zeros((n, 1), dtype=np.complex128)
    for i, idx in enumerate(Shat):
        v[idx] = v1[i]
    x_init = np.dot(phi, v)
    x = x_init
    for i in range(max_iter):
        p = np.sign(np.dot(A, x))
        something = p.reshape(-1) * y_abs.reshape(-1)
        x = cosamp.cosamp(A / np.sqrt(m), something / np.sqrt(m), s, 1e-8, 10)
    return x
