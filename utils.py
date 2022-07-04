import numpy as np
import ffttools as fftt


# Miscellaneous

def mad(X=0, M=None):
    """Compute median absolute estimator.

    Parameters
    ----------
    X: np.ndarray
        data
    M: np.ndarray
        mask with the same size of x, optional

    Returns
    -------
    float
        mad estimate
        """
    if M is None:
        return np.median(abs(X - np.median(X))) / 0.6735
    xm = X[M == 1]
    return np.median(abs(xm - np.median(xm))) / 0.6735


def fit_power_law(y, thrd=None):
    """Fit a 2D power law of the form y = np.exp(a*r+b), with r the distance to the center and a & b two parameters.

    Parameters
    ----------
    y: np.ndarray
        (,n) float array, input 2D-signal to fit, flattened into a vector
    thrd: float
        Threshold below which to ignore entries of y

    Returns
    -------
    (,n) float array
        Fitted signal
    """

    if thrd is None:
        thrd = np.max(y)*1e-10

    size = int(np.sqrt(len(y)))

    xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2).flatten()

    locs = y > thrd
    logy = np.log(y[locs])

    mean_r = np.mean(r)
    mean_logy = np.mean(logy)

    a = np.sum((r[locs]-mean_r)*(logy-mean_logy))/np.sum((r[locs]-mean_r)**2)
    b = mean_logy - a*mean_r

    return np.exp(a*r+b)


def generate_sources(n=1, size=256, cutmin=None, cutmax=None, sparseLvl=2., nscales=3, nbIt=25, verb=0):
    """Generate synthetic sources.

    Parameters
    ----------
    n: int
        number of sources
    size: int
        size of the sources
    cutmin: int
        frequency at which the band-limiting filter starts to cut (default: int(size/2))
    cutmax: int
        frequency above which the band-limiting filter is 0 (default: max(minResol, size/2))
    nscales: int
        number of detail scales
    sparseLvl: float
        desired sparsity level on the wavelet domain (k*mad per scale)
    nbIt: int
        number of iterations
    verb: int
        verbosity level

    Returns
    -------
    np.ndarray
        (m,n) float array, mixing matrix
    """

    if cutmin is None:
        cutmin = int(size/6)
    if cutmax is None:
        cutmax = int(size/2)

    if verb > 0:
        print(" cutmin ", cutmin, " - cutmax ", cutmax)

    bl = fftt.get_ideal_beam(size, cutmin, cutmax)

    S = np.random.randn(n, size**2)
    Sfft = fftt.fft(S)
    Sfft = fftt.fftprod(Sfft, bl)  # band-limited

    for it in range(nbIt):
        if verb > 0:
            print("iteration: ", it+1)
        Swt = fftt.wt_trans(Sfft, nscales=nscales, fft_in=True)
        for i in range(n):
            for j in range(nscales+1):
                threshold = sparseLvl * mad(Swt[i, :, j])
                Swt[i, :, j] = (Swt[i, :, j]-threshold*np.sign(Swt[i, :, j]))*(abs(Swt[i, :, j]) > threshold)
        S = fftt.wt_rec(Swt)
        S = S * (S > 0)  # positivity but not guaranteed after band-limiting constraint
        Sfft = fftt.fft(S)
        Sfft = fftt.fftprod(Sfft, bl)  # band-limited

    if n == 1:
        Sfft = Sfft[0, :]

    S = fftt.ifft(Sfft).real
    S /= np.linalg.norm(S, axis=1)[:, np.newaxis]
    S /= np.max(S)

    return S  # multiply by 20 to have sources of approx. unit value


def generate_mixmat(n=2, m=4, condn=2., dcondn=1e-2, max0s=None, forceMax0s=False):
    """Generate a synthetic mixing matrix.

    Parameters
    ----------
    n: int
        number of sources
    m: int
        number of observations
    condn: float
        desired condition number
    dcondn: float
        condition number precision
    max0s: int
        maximum number of zeros (default: no condition)
    forceMax0s: bool
        if False, max0s is incremented every minute, to ensure the convergence of the function

    Returns
    -------
    np.ndarray
        (m,n) float array, mixing matrix
    """

    A = np.random.rand(m, n)
    A = A / np.maximum(1e-24, np.linalg.norm(A, axis=0))
    if max0s is None:
        max0s = n * m
    it = 0
    while True:
        it += 1
        if not forceMax0s and it >= 5e4:  # relax condition on max number of zeros
            it = 0
            max0s += 1
        try:
            U, d, V = np.linalg.svd(A)
        except np.linalg.LinAlgError:  # divergence, new A drawn
            A = np.random.rand(m, n)
            A = A / np.maximum(1e-24, np.linalg.norm(A, axis=0))
        else:
            d = d[-1] * ((d - d[-1]) * (condn - 1) / (d[0] - d[-1]) + 1)
            D = A * 0
            D[:n, :n] = np.diag(d)
            A = U @ D @ V.T
            for i in range(n):
                if sum(A[:, i] > 0) <= np.int(m / 2):  # if there are more negative numbers
                    A[:, i] = np.maximum(-A[:, i], 0)
                else:
                    A[:, i] = np.maximum(A[:, i], 0)
            A = A / np.maximum(1e-24, np.linalg.norm(A, axis=0))
            err = np.abs(np.linalg.cond(A) - condn)
            if err < dcondn or err > 1e10:
                if np.count_nonzero(A == 0) <= max0s:
                    return A
                A = np.random.rand(m, n)
                A = A / np.maximum(1e-24, np.linalg.norm(A, axis=0))


def generate_filt(m=8, size=256, minResol=None, maxResol=None, infResol=False):
    """Generate Gaussian-shaped filters in Fourier domain.

    Parameters
    ----------
    m: int
        number of observations
    size: int
        size of the sources
    minResol: float
        filters: fwhm in Fourier space of the worse-resolved observation (default: int(size/4))
    maxResol: float
        filters: fwhm in Fourier space of the best-resolved observation (default: max(minResol, size/2))
    infResol: bool
        filters: the observations are not convolved, overrides minResol and maxResol

    Returns
    -------
    np.ndarray
         (m,p) float array, convolution kernels in Fourier domain
    """

    if infResol:
        return np.ones((m, size**2))

    if minResol is None:
        minResol = int(size/4)
    if maxResol is None:
        maxResol = int(max(minResol, size/2))

    resol = np.linspace(minResol, maxResol, m)
    sigma2 = resol**2/(2*np.log(2))
    Hfft = np.zeros((m, size**2))
    xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
    for observation in range(m):
        Hfft[observation, :] = np.exp(-((xx-size/2)**2+(yy-size/2)**2)/(2*sigma2[observation])).flatten()
    return Hfft


# Metrics

def evaluate(A0, S0, A, S, each_component=False, which='both', do_corr_perm=True):
    """Computes performance metrics for A and S.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    S0: np.ndarray
        (n,p) float array, ground truth sources
    A: np.ndarray
        (m,n) float array, estimated mixing matrix
    S: np.ndarray
        (n,p) float array, estimated sources
    each_component: bool
        return the metrics for each component
    which: str
        performance metrics which are evaluated ('both': of both A and S, 'S': of S only, 'A': of A only)
    do_corr_perm: bool
        do permutation correction

    Returns
    -------
    dict
    """

    which = which.lower()
    res = {}

    if do_corr_perm:
        A, S = corr_perm(A0, A, S, norm_data=False)

    if which == 'both' or which == 's':
        SNMSE = nmse(S0, S, each_component=each_component)
        res = {**res, **{'SNMSE': SNMSE}}

    if which == 'both' or which == 'a':
        A0 = A0 / np.maximum(1e-9, np.linalg.norm(A0, axis=0))
        A = A / np.maximum(1e-9, np.linalg.norm(A, axis=0))
        SAD = sad(A0, A, each_component=each_component)
        res = {**res, **{'SAD': SAD}}

    return res


def corr_perm(A0, A, S, inplace=False, optInd=False, norm_data=True):
    """Correct the permutation of the solution.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    A: np.ndarray
        (m,n) float array, estimated mixing matrix
    S: np.ndarray
        (n,p) float array, estimated sources
    inplace: bool
        in-place update of A and S
    optInd: bool
        return permutation
    norm_data: bool
        normalize the lines of S and the columns of A according to the columns of A0

    Returns
    -------
    None or np.ndarray or (np.ndarray,np.ndarray) or (np.ndarray,np.ndarray,np.ndarray)
        A (if not inplace),
        S (if not inplace),
        ind (if optInd)
    """

    n = np.shape(A0)[1]

    if not inplace:
        A = A.copy()
        S = S.copy()

    # Normalize data for comparison
    A_norm = A/np.maximum(1e-9, np.linalg.norm(A, axis=0))
    A0_norm = A0/np.maximum(1e-9, np.linalg.norm(A0, axis=0))

    try:
        diff = abs(np.dot(np.linalg.inv(np.dot(A0_norm.T, A0_norm)), np.dot(A0_norm.T, A_norm)))
    except np.linalg.LinAlgError:
        diff = abs(np.dot(np.linalg.pinv(A0_norm), A_norm))
        print('Warning! Pseudo-inverse used.')

    ind = np.argmax(diff, axis=1)

    if len(np.unique(ind)) != n:  # if there are duplicates in ind, we proceed differently
        ind = np.ones(n)*-1
        args = np.flip(np.unravel_index(np.argsort(diff, axis=None), (n, n)), axis=1)
        for i in range(n**2):
            if ind[args[0, i]] == -1 and args[1, i] not in ind:
                ind[args[0, i]] = args[1, i]

    A[:] = A[:, ind.astype(int)]
    S[:] = S[ind.astype(int), :]

    for i in range(0, n):
        p = np.sum(A[:, i] * A0[:, i])
        if p < 0:
            S[i, :] = -S[i, :]
            A[:, i] = -A[:, i]

    # Norm data
    if norm_data:
        factor = np.maximum(1e-9, np.linalg.norm(A0, axis=0)/np.linalg.norm(A, axis=0))
        S /= factor[:, np.newaxis]
        A *= factor

    if inplace and not optInd:
        return None
    elif inplace and optInd:
        return ind.astype(int)
    elif not optInd:
        return A, S
    else:
        return A, S, ind.astype(int)


def nmse(x0, x, each_component=False):
    """Compute the normalized mean-square error (NMSE) in dB

    Parameters
    ----------
    x0: np.ndarray
        (,p) or (n,p) float array, ground truth signal or set of n ground truth signal
    x: np.ndarray
        (,p) or (n,p) float array, estimated signal or set of n estimated signals
    each_component: bool
        return the NMSE for each component

    Returns
    -------
    float or (,n) float array
    """
    if len(np.shape(x)) == 1:
        return -10 * np.log10(np.sum((x - x0) ** 2) / np.maximum(np.sum(x0 ** 2), 1e-10))
    if each_component:
        return -10 * np.log10(np.sum((x - x0) ** 2, axis=-1) / np.maximum(np.sum(x0 ** 2, axis=-1), 1e-10))
    return np.mean(-10 * np.log10(np.sum((x - x0) ** 2, axis=-1) / np.maximum(np.sum(x0 ** 2, axis=-1), 1e-10)))


def sad(A0, A, each_component=False):
    """Compute the spectral angle distance (SAD) in dB.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    A: np.ndarray
        (m,n) float array, estimated mixing matrix
    each_component: bool
        return the SAD for each component

    Returns
    -------
    float or (,n) float array
    """

    if not each_component:
        return np.mean(-10*np.log10(np.arccos(
            np.clip(np.sum(A0*A, axis=0)/np.sqrt(np.sum(A0**2, axis=0)*np.sum(A**2, axis=0)), -1, 1))))
    else:
        return -10 * np.log10(np.arccos(
            np.clip(np.sum(A0 * A, axis=0) / np.sqrt(np.sum(A0 ** 2, axis=0) * np.sum(A ** 2, axis=0)), -1, 1)))
