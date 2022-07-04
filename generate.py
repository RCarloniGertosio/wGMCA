import numpy as np
import ffttools as fftt
from skimage.transform import resize
from utils import mad


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
        (m,size**2) float array
    """

    if cutmin is None:
        cutmin = int(size / 6)
    if cutmax is None:
        cutmax = int(size / 2)

    if verb > 0:
        print(" cutmin ", cutmin, " - cutmax ", cutmax)

    bl = fftt.get_ideal_beam(size, cutmin, cutmax)

    S = np.random.randn(n, size ** 2)
    Sfft = fftt.fft(S)
    Sfft = fftt.fftprod(Sfft, bl)  # band-limited

    for it in range(nbIt):
        if verb > 0:
            print("iteration: ", it + 1)
        Swt = fftt.wt_trans(Sfft, nscales=nscales, fft_in=True)
        for i in range(n):
            for j in range(nscales + 1):
                threshold = sparseLvl * mad(Swt[i, :, j])
                Swt[i, :, j] = (Swt[i, :, j] - threshold * np.sign(Swt[i, :, j])) * (abs(Swt[i, :, j]) > threshold)
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
        (m,n) float array
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


def generate_gaussian_filters(m=8, size=256, minResol=None, maxResol=None, infResol=False):
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
         (m,size**2) float array
    """

    if infResol:
        return np.ones((m, size ** 2))

    if minResol is None:
        minResol = int(size / 4)
    if maxResol is None:
        maxResol = int(max(minResol, size / 2))

    resol = np.linspace(minResol, maxResol, m)
    sigma2 = resol ** 2 / (2 * np.log(2))
    Hfft = np.zeros((m, size ** 2))
    xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
    for observation in range(m):
        Hfft[observation, :] = np.exp(
            -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2 * sigma2[observation])).flatten()
    return Hfft


def generate_masks(m=8, size=256, n_stacks=11, alpha_vis=.1, sigma_uv=1, sigma_w=1, resolution_ratio=.1):
    """Generate non-coplanar interferometric masks.

    Parameters
    ----------
    m: int
        number of observations
    size: int
        size of the sources
    n_stacks: int
        number of samples along the w axis (must be odd)
    alpha_vis: float
        mask support, as a ratio (0 < alpha_vis <= 1)
    sigma_uv: float
        dispersion along the u & v axes
    sigma_w: float
        dispersion along the w axis
    resolution_ratio: float
        ratio of the lowest frequency to the highest frequency

    Returns
    -------
    np.ndarray
         (m,n_stacks,size**2) float array
    """

    # Define the gridded visibility set 'vis_set'
    def uv_idx2idx(u, v):
        u = np.copy(u)
        v = np.copy(v)
        u[u == size / 2] = -size / 2  # for the mask generation
        v[v == size / 2] = -size / 2  # for the mask generation
        return (int((size ** 2 + size) / 2) + size * v + u).astype('int')

    def w_idx2idx(w):
        if hasattr(w, '__len__'):
            return (int(n_stacks / 2) + w).astype('int')
        return int(n_stacks / 2 + w)

    uv_idx = np.arange(-int(size / 2), int(size / 2))
    w_idx = np.arange(-int((n_stacks - 1) / 2), int((n_stacks + 1) / 2))
    vis_set = np.array(np.meshgrid(uv_idx, np.delete(uv_idx, np.arange(1, int(size / 2))), w_idx))
    vis_set = np.reshape(vis_set, (3, np.prod(np.shape(vis_set)[1:]))).T

    # The visibility space verifies some conjugate-symmetry propriety (eg., V(u,v,w)=V*(-u,-v,-w)). Therefore,
    # for each conjugate-symmetric pair in `vis_set`, one of the two elements is removed.
    vis_set = np.delete(vis_set, np.argwhere(np.logical_and(1 <= vis_set[:, 0], vis_set[:, 1] <= 0)), axis=0)
    vis_set = np.delete(vis_set, np.argwhere(np.logical_and(
        np.logical_and(vis_set[:, 2] < 0, np.logical_or(vis_set[:, 0] == -size // 2, vis_set[:, 0] == 0)),
        np.logical_or(vis_set[:, 1] == -size // 2, vis_set[:, 1] == 0))), axis=0)

    # First, generate the mask at F_max
    if alpha_vis <= 1:
        law = np.exp(
            -(vis_set[:, 0] ** 2 + vis_set[:, 1] ** 2) / (2 * sigma_uv ** 2) - vis_set[:, 2] ** 2 / (2 * sigma_w ** 2))
        law /= np.sum(law)
        vis_meas_idx = np.random.choice(np.arange(len(vis_set)), int(alpha_vis * len(vis_set)), replace=False, p=law)

        masks = np.zeros((n_stacks, size ** 2))
        masks[w_idx2idx(vis_set[vis_meas_idx, 2]), uv_idx2idx(vis_set[vis_meas_idx, 0], vis_set[vis_meas_idx, 1])] = 1
    else:
        masks = np.ones((n_stacks, size ** 2))
    # Ensure that the mask respects the conjugate-symmetry propriety
    masks[w_idx2idx(-vis_set[:, 2]), uv_idx2idx(-vis_set[:, 0], -vis_set[:, 1])] = np.conj(
        masks[w_idx2idx(vis_set[:, 2]), uv_idx2idx(vis_set[:, 0], vis_set[:, 1])])
    masks[w_idx2idx(0), uv_idx2idx([-size // 2, -size // 2, 0, 0], [-size // 2, 0, -size // 2, 0])] = np.abs(
        masks[w_idx2idx(0), uv_idx2idx([-size // 2, -size // 2, 0, 0], [-size // 2, 0, -size // 2, 0])])

    # Then, deduce the masks at other frequencies by interpolation
    resolution_ratios = np.linspace(resolution_ratio, 1, m)
    multiresol_masks = np.zeros((m, n_stacks, size ** 2))
    for mi in range(m):
        if resolution_ratios[mi] != 1:
            w_idx_temp = np.round(w_idx * resolution_ratios[mi]).astype(int)
            holes_temp = np.zeros_like(masks)
            for j in range(n_stacks):  # resize along w
                holes_temp[w_idx2idx(w_idx_temp[j])] += masks[j]
            for j in range(n_stacks):  # resize along (u,v)
                size_temp = int(np.round(size * resolution_ratios[mi]))
                temp = resize(np.reshape(holes_temp[j].astype('float64'), (size, size)), (size_temp, size_temp),
                              order=1,
                              mode='constant', cval=0)
                if np.linalg.norm(temp) != 0:
                    temp *= np.linalg.norm(holes_temp[j]) / np.linalg.norm(temp)
                temp_resized = np.zeros((size, size))
                if size_temp % 2 == 0:
                    temp_resized[size // 2 - size_temp // 2:size // 2 + size_temp // 2,
                    size // 2 - size_temp // 2:size // 2 + size_temp // 2] = temp
                else:
                    temp_resized[size // 2 - size_temp // 2:size // 2 + size_temp // 2 + 1,
                    size // 2 - size_temp // 2:size // 2 + size_temp // 2 + 1] = temp
                multiresol_masks[mi, j, :] = temp_resized.flatten()
        else:
            multiresol_masks[mi, :, :] = masks
    # Again, ensure that the masks respect the conjugate-symmetry propriety
    multiresol_masks[:, w_idx2idx(-vis_set[:, 2]), uv_idx2idx(-vis_set[:, 0], -vis_set[:, 1])] = np.conj(
        multiresol_masks[:, w_idx2idx(vis_set[:, 2]), uv_idx2idx(vis_set[:, 0], vis_set[:, 1])])
    multiresol_masks[:, w_idx2idx(0),
    uv_idx2idx([-size // 2, -size // 2, 0, 0], [-size // 2, 0, -size // 2, 0])] = np.abs(
        multiresol_masks[:, w_idx2idx(0), uv_idx2idx([-size // 2, -size // 2, 0, 0], [-size // 2, 0, -size // 2, 0])])

    return multiresol_masks


def generate_w_matrices(size=256, l_max=1e-4, n_stacks=11, w_max=1e4):
    """Generate w-term matrices.

    Parameters
    ----------
    size: int
        size of the sources
    l_max: float
        maximum value of l and m (direction cosine)
    n_stacks: int
        number of samples along the w axis (must be odd)
    w_max: float
        maximum value of w-term

    Returns
    -------
    np.ndarray
         (n_stacks,size**2) float array
    """

    # Define the (u,v) grid
    ll, mm = np.meshgrid(np.linspace(-l_max, l_max, size + 1)[:-1], np.linspace(-l_max, l_max, size + 1)[:-1])
    ll = ll.flatten()
    mm = mm.flatten()

    # Define the w grid
    w = np.linspace(-w_max, w_max, 2 * n_stacks + 1)[1:-1:2]
    min_n_stacks = 2 * np.pi * 2 * w_max * (1 - np.sqrt(1 - 2 * l_max ** 2))
    if 10 * min_n_stacks > n_stacks:
        print("Beware, number of stacks considered is not realistic.")

    G = np.exp(-2 * np.pi * 1j * w[:, np.newaxis] * (np.sqrt(1 - ll ** 2 - mm ** 2) - 1)[np.newaxis, :])

    return G


def generate_noise(Var, m=None, size=None, n_stacks=None):
    """Generate complex noise with desired conjugate-symmetry properties.

    Parameters
    ----------
    Var: np.ndarray or float
        noise variance(s) (if not identically distributed, (m,n_stacks,size**2) float array)
    m: int
        number of observations (must be provided if Var is float, otherwise deduced from Var)
    size: int
        size of the sources (must be provided if Var is float, otherwise deduced from Var)
    n_stacks: int
        number of samples along the w axis (must be provided if Var is float, otherwise deduced from Var)

    Returns
    -------
    np.ndarray
         (m,n_stacks,size**2) complexe array
    """

    if not np.isscalar(Var):
        m = np.shape(Var)[0]
        size = int(np.sqrt(np.shape(Var)[2]))
        n_stacks = np.shape(Var)[1]

    # Define the gridded visibility set 'vis_set'
    def uv_idx2idx(u, v):
        u = np.copy(u)
        v = np.copy(v)
        u[u == size / 2] = -size / 2  # for the mask generation
        v[v == size / 2] = -size / 2  # for the mask generation
        return (int((size ** 2 + size) / 2) + size * v + u).astype('int')

    def w_idx2idx(w):
        if hasattr(w, '__len__'):
            return (int(n_stacks / 2) + w).astype('int')
        return int(n_stacks / 2 + w)

    uv_idx = np.arange(-int(size / 2), int(size / 2))
    w_idx = np.arange(-int((n_stacks - 1) / 2), int((n_stacks + 1) / 2))
    vis_set = np.array(np.meshgrid(uv_idx, np.delete(uv_idx, np.arange(1, int(size / 2))), w_idx))
    vis_set = np.reshape(vis_set, (3, np.prod(np.shape(vis_set)[1:]))).T

    # The visibility space verifies some conjugate-symmetry propriety (eg., V(u,v,w)=V*(-u,-v,-w)). Therefore,
    # for each conjugate-symmetric pair in `vis_set`, one of the two elements is removed.
    vis_set = np.delete(vis_set, np.argwhere(np.logical_and(1 <= vis_set[:, 0], vis_set[:, 1] <= 0)), axis=0)
    vis_set = np.delete(vis_set, np.argwhere(np.logical_and(
        np.logical_and(vis_set[:, 2] < 0, np.logical_or(vis_set[:, 0] == -size // 2, vis_set[:, 0] == 0)),
        np.logical_or(vis_set[:, 1] == -size // 2, vis_set[:, 1] == 0))), axis=0)

    N = (np.random.randn(m, n_stacks, size**2) + 1j * np.random.randn(m, n_stacks, size**2))*np.sqrt(Var/2)
    N[:, w_idx2idx(vis_set[:, 2]), uv_idx2idx(vis_set[:, 0], vis_set[:, 1])] = np.conj(
        N[:, w_idx2idx(-vis_set[:, 2]), uv_idx2idx(-vis_set[:, 0], -vis_set[:, 1])])
    N[:, w_idx2idx(0), uv_idx2idx([-size // 2, -size // 2, 0, 0], [-size // 2, 0, -size // 2, 0])] = np.abs(
        N[:, w_idx2idx(0), uv_idx2idx([-size // 2, -size // 2, 0, 0], [-size // 2, 0, -size // 2, 0])])

    return N
