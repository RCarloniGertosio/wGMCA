import numpy as np
import ffttools as fftt
import utils
from scipy.sparse.linalg import cg, LinearOperator


class wGMCA:
    """wGMCA joint deconvolution and blind source separation algorithm for non-coplanar interferometric data.

    The multichannel input data are assumed to be gridded on a regular (u,v,w) grid and flattened along the (u,v)-axes.
    The filters are assumed to have the zero frequency shifted to the center and to be flattened along the (u,v)-axes.

    Example
    --------
    wgmca = wGMCA(X, H, 4, Var, k=3, K_max=0.5, nscales=3, c_wu=0.5, c_ref=0.5)
    wgmca.run()
    S = wgmca.S.copy()
    A = wgmca.A.copy()
    """

    def __init__(self, X, H, n, Var, G, **kwargs):
        """Initialize object.

        Parameters
        ----------
        X : np.ndarray
            (m,w,p) complex array, input data (gridded visibilities); 1st axis: channel, 2nd axis: w axis, 3rd axis: flattened
            (u,v) axes
        H : np.ndarray
            (m,w,p) float array, interferometric response in visibility domain for several w-values, with zero-frequency
             shifted to the center and flattened
        n : int
            number of sources to be estimated
        Var : np.ndarray or float
            (m,w,p) float array or float, noise variance (wGMCA does not account for potential noise covariances)
        G : np.ndarray
            (w,p) float array, w-term matrices
        AInit : np.ndarray
            (m,n) float array, initial value for the mixing matrix. Default: None (PCA-based initialization)
        nnegA : bool
            non-negativity constraint on A. Default: True
        nnegS : bool
            non-negativity constraint on S. Default: True
        nneg : bool
            non-negativity constraint on A and S, overrides nnegA and nnegS if not None. Default: None
        keepWuRegStr : bool
            keep warm-up regularization strategy during refinement. Default: False (power-spectrum-based coefficients)
        cstWuRegStr : bool
            use constant regularization coefficients during warm-up. Default: False (mixing-matrix-based coefficients)
        minWuIt : int
            minimum number of iterations at warm-up. Default: 50
        c_wu : float
            Tikhonov regularization hyperparameter at warm-up. Default: 0.5
        c_ref : float
            Tikhonov regularization hyperparameter at refinement. Default: 0.5
        c_end : float
            Tikhonov regularization hyperparameter at finale refinement of S. Default: 0.5
        itCG : int
            maximum number of iterations of the conjugate gradient algorithm during refinement. Default: 100
        nscales : int
            number of detail scales. Default: 3
        useMad : bool
            estimate noise std in source domain with MAD (else: analytical estimation, use with caution). Default: True
        useMad_end : bool
            estimate noise std in source domain with MAD during finale refinement of S (else: analytical estimation,
            use with caution). Default: True
        k : float
            parameter of the k-std thresholding. Default: 3
        L1 : bool
            L1 penalization (else: L0 penalization). Default: True
        thr_end : bool
            perform thresholding during finale refinement of S. Default: True
        K_max : float
            maximal L0 norm of the sources. Being a percentage, it should be between 0 & 1
        K_end : float
            maximal L0 norm of the sources during finale refinement of S. Being a percentage, it should be between 0 & 1
        doRw : bool
            do L1 reweighing during refinement (only if L1 penalization). Default: True
        doRw_end : bool
            do L1 reweighing during finale refinement of S (only if L1 penalization). Default: doRw
        H_reconv : np.ndarray
            (p,) float array, kernel in Fourier domain, with zero-frequency shifted to the center and flattened, by
            which the sources are reconvolved for the estimation of A. Default: None
        removeCoarseScaleData : bool
            remove coarse scale from data for the estimation of A. Default: False
        eps : np.ndarray
            (3,) float array, stopping criteria of (1) the warm-up, (2) the refinement and (3) the finale refinement of
            S. Default: np.array([1e-2, 1e-4, 1e-4])
        verb : int
            verbosity level, from 0 (mute) to 5 (most talkative). Default: 0
        S0 : np.ndarray
            (n,p) float array, ground truth sources (for testing purposes). Default: None
        A0 : np.ndarray
            (m,n) float array, ground truth mixing matrix (for testing purposes). Default: None

        Returns
        -------
        wGMCA
        """

        # Initialize given attributes
        self.X = X
        self.H = H
        self.n = n
        self.Var = Var
        if np.isscalar(self.Var):
            self.Var = self.Var * np.ones(np.shape(X))
        self.G = G
        self.AInit = kwargs.get('AInit', None)
        nneg = kwargs.get('nneg', None)
        if nneg is not None:
            self.nnegA = nneg
            self.nnegS = nneg
        else:
            self.nnegA = kwargs.get('nnegA', True)
            self.nnegS = kwargs.get('nnegS', True)
        self.keepWuRegStr = kwargs.get('keepWuRegStr', False)
        self.cstWuRegStr = kwargs.get('cstWuRegStr', False)
        self.minWuIt = kwargs.get('minWuIt', 50)
        self.c_wu = kwargs.get('c_wu', .5)
        self.c_ref = kwargs.get('c_ref', .5)
        self.c_end = kwargs.get('c_end', .5)
        self.itCG = kwargs.get('itCG', 0)
        self.nscales = kwargs.get('nscales', 3)
        self.useMad = kwargs.get('useMad', True)
        self.useMad_end = kwargs.get('useMad_end', self.useMad)
        self.k = kwargs.get('k', 3)
        self.L1 = kwargs.get('L1', True)
        self.thr_end = kwargs.get('thr_end', True)
        self.K_max = kwargs.get('K_max', .5)
        self.K_end = kwargs.get('K_end', 1)
        self.doRw = kwargs.get('doRw', True)
        self.doRw_end = kwargs.get('doRw_end', True)
        self.H_reconv = kwargs.get('H_reconv', None)
        self.removeCoarseScaleData = kwargs.get('removeCoarseScaleData', False)
        self.eps = kwargs.get('eps', np.array([1e-2, 1e-4, 1e-4]))
        self.verb = kwargs.get('verb', 0)
        self.S0 = kwargs.get('S0', None)
        self.A0 = kwargs.get('A0', None)

        # Initialize deduced attributes
        self.m = np.shape(X)[0]  # number of channels
        self.w = np.shape(X)[1]  # number of w-stacks
        self.p = np.shape(X)[2]  # number of pixels per channel
        self.size = np.int(np.sqrt(self.p))
        self.wt_filters = fftt.get_wt_filters(size=self.size, nscales=self.nscales)  # wavelet filters
        self.X_det = self.H * fftt.fft(self.G * (
            fftt.ifft((1 - self.wt_filters[:, -1]) * fftt.fft(self.G.conj() * fftt.ifft(self.X / (self.H + 1e-5))))))
        # For testing purposes:
        if self.S0 is not None:
            self.Swfft0 = fftt.fft(self.G[np.newaxis, :, :] * self.S0[:, np.newaxis, :])
            self.S0wt = fftt.wt_trans(self.S0, nscales=self.nscales)  # true sources in the wavelet domain
            self.Swfft0_det = fftt.fft(
                self.G[np.newaxis, :, :] * fftt.wt_rec(self.S0wt[:, :, :self.nscales])[:, np.newaxis, :])
            if self.H_reconv is not None:
                self.Swfft0_det_reconv = fftt.fft(self.G[np.newaxis, :, :] * np.real(
                    fftt.convolve(fftt.wt_rec(self.S0wt[:, :, :self.nscales])[:, np.newaxis, :], self.H_reconv)))

        # Initialize other attributes
        self.S = np.zeros((self.n, self.p))  # current estimation of the sources
        self.Swfft = np.zeros((self.n, self.w, self.p), dtype=complex)  # current est. of the w-mod. sources in Fourier
        self.Swfft_det = np.zeros((self.n, self.w, self.p), dtype=complex)  # current est. of the detail scales of S
        self.Swfft_reconv = None  # reconvolved version of Swftt using H_reconv (if provided)
        self.Swfft_det_reconv = None  # reconvolved version of Swftt using H_reconv (if provided)
        self.stds = np.zeros((self.n, self.nscales))  # std of the noise in the source space, per detail scale
        self.Swtrw = np.zeros((self.n, self.p, self.nscales))  # weights for the l1 reweighing
        self.A = np.zeros((self.m, self.n))  # current estimation of the mixing matrix
        self.lastWuIt = None  # last warm-up iteration
        self.lastRefIt = None  # last refinement iteration
        self.metrics = {}
        self.aborted = False  # True if last resolution has been aborted

    def __str__(self):
        res = '\n'
        res += '  - Number of sources: %i\n' % self.n
        if self.nnegA or self.nnegS:
            res += '  - Non-negativity constraint on '
            if self.nnegA and self.nnegS:
                res += 'A and S\n'
            elif not self.nnegS:
                res += 'A\n'
            else:
                res += 'S\n'
        if self.cstWuRegStr:
            res += '  - Constant regularization coefficients during warm-up\n'
        res += '  - Minimum number of iterations at warm-up: %i\n' % self.minWuIt
        res += '  - Tikhonov regularization hyperparameter at warm-up: %.2e\n' % self.c_wu
        if not self.keepWuRegStr:
            res += '  - Tikhonov regularization hyperparameter at refinement: %.2e\n' % self.c_ref
        else:
            res += '  - Keep warm-up regularization strategy during refinement\n'
        if self.useMad:
            res += '  - Noise std estimated with MAD\n'
        else:
            res += '  - Noise std estimated analytically with the input noise std of the observations ' \
                   '(implemented approximately, not recommended)\n'
        res += '  - nscales = %i  |  k = %.2f  |  K_max = %i%%\n' % (self.nscales, self.k, self.K_max * 100)
        if not self.L1:
            res += '  - L0 penalization\n'
        elif not self.doRw:
            res += '  - L1 penalization\n'
        else:
            res += '  - L1 penalization with L1-reweighting\n'
        if not self.thr_end:
            res += '  - No source thresholding after the separation process\n'
        return res

    def run(self):
        """Run wGMCA with the data and the parameters stored in the attributes.

        Returns
        -------
        int
            error code
        """

        if self.verb:
            print(self)
        self.initialize()
        core = self.core()
        if core == 1:
            self.aborted = True
            return 1
        refine_s_end = self.refine_s_end()
        if refine_s_end == 1:
            self.aborted = True
            return 1
        self.terminate()
        return 0

    def initialize(self):
        """Initialize the attributes for a new separation.

        Returns
        -------
        int
            error code
        """

        X = fftt.fftprod(self.X, self.H[0, :] * np.conj(self.H) / (np.abs(self.H) ** 2 + 1e-10))
        X = np.sum(np.conj(self.G)[np.newaxis, :, :] * X, axis=1) / self.w
        # Initialize A
        if self.AInit is not None:
            self.A = self.AInit.copy()
        else:  # PCA with the deteriorated data
            R = np.real(X @ np.conj(X.T))
            D, V = np.linalg.eig(R)
            self.A = np.real(V[:, :self.n])
        self.A /= np.maximum(np.linalg.norm(self.A, axis=0), 1e-24)

        # Initialize other parameters
        self.S = np.zeros((self.n, self.p))
        self.Swfft = np.zeros((self.n, self.w, self.p), dtype=complex)
        self.Swtrw = np.zeros((self.n, self.p, self.nscales))
        self.lastWuIt = None
        self.lastRefIt = None
        self.metrics = {}
        self.aborted = False

        return 0

    def core(self):
        """Manage the separation.

        This function handles the alternate updates of S and A, as well as the two stages (warm-up and refinement).

        Returns
        -------
        int
            error code
        """

        stage = "wu"

        S_old = np.zeros((self.n, self.p))
        A_old = np.zeros((self.m, self.n))
        it = 0

        while True:
            it += 1
            # Get parameters of wGMCA for the current iteration
            strat, c, K, do_cg, doRw, nnegS = self.get_parameters(stage, it)

            if self.verb >= 2:
                print("Iteration #%i" % it)

            # Update S
            update_s = self.update_s(strat, c, K, do_cg=do_cg, doRw=doRw, nnegS=nnegS)
            if update_s:  # error caught
                return 1

            # Update A
            update_a = self.update_a()
            if update_a:  # error caught
                return 1

            # Post processing

            delta_S = np.linalg.norm(S_old - self.S) / np.linalg.norm(self.S)
            delta_A = np.max(abs(1 - abs(np.sum(self.A * A_old, axis=0))))
            cond_A = np.linalg.cond(self.A)
            S_old = self.S.copy()
            A_old = self.A.copy()

            if self.A0 is not None and self.S0 is not None and self.verb >= 2:
                metrics = utils.evaluate(self.A0, self.S0, self.A, self.S)
                print("SNMSE = %.2f  -  SAD = %.2f" % (metrics['SNMSE'], metrics['SAD']))

            if self.verb >= 2:
                print("delta_S = %.2e - delta_A = %.2e - cond(A) = %.2f" % (delta_S, delta_A, cond_A))
            if self.verb >= 5:
                print("A:\n", self.A)

            # Stage update

            if stage == 'wu' and it >= self.minWuIt and (delta_S <= self.eps[0] or it >= self.minWuIt + 50):
                if self.verb >= 2:
                    print("> End of the warm-up (iteration %i)" % it)
                self.lastWuIt = it
                stage = 'ref'

            if stage == 'ref' and (delta_S <= self.eps[1] or it >= self.lastWuIt + 50) and (it >= self.lastWuIt + 25):
                if self.verb >= 2:
                    print("> End of the refinement (iteration %i)" % it)
                self.lastRefIt = it
                return 0

    def get_parameters(self, stage, it):
        """Get the parameters of wGMCA.

        Return the parameters of wGMCA according to the stage and the iteration.

        Parameters
        ----------
        stage : str
            stage ('wu': warm-up, 'ref': refinement)
        it : int
            iteration

        Returns
        -------
        (int, float, float, bool, bool, bool)
            regularization strategy,
            regularization hyperparameter,
            L0 support of the sources,
            do conjugate gradient refinement,
            do L1 reweighting,
            apply non-negativity constraint on the sources
        """

        if self.cstWuRegStr:
            strat = 0  # constant regularization coefficients
        else:
            strat = 1  # mixing-matrix-based regularization coefficients
        if stage == 'wu':
            c = self.c_wu
            K = np.minimum(self.K_max / self.minWuIt * it, self.K_max)
            do_cg = it >= self.minWuIt  # conjugate gradient when K reaches K_max (and if self.itCG > 0)
            doRw = False  # no l1 reweighing during warm-up
            nnegS = False  # no non-negativity constraint on S during warm-up
        else:
            if self.keepWuRegStr:
                c = self.c_wu
            else:
                strat = 2  # spectrum-based regularization coefficients
                c = self.c_ref
            do_cg = True  # conjugate gradient performed if self.itCG > 0
            K = self.K_max
            doRw = self.doRw
            nnegS = self.nnegS
        return strat, c, K, do_cg, doRw, nnegS

    def update_s(self, strat, c, K, do_cg=False, doThr=True, doRw=None, nnegS=None, useMad=None, stds=None,
                 oracle=False):
        """Perform the update of the sources.

        Perform the update of the sources, including a first Tikhonov-regularized least squares to get the w-modulated
        sources in Fourier, a second least squares to retrieve the sources, a thresholding in the starlet domain
        and possibly a projection on the positive orthant.

        Parameters
        ----------
        strat: int
            regularization strategy (0: constant, 1: mixing-matrix-based, 2: spectrum-based)
        c: float
            regularization hyperparameter
        K: float
            L0 support of the sources
        do_cg: bool
            refine estimation of the sources with conjugate gradient
        doThr: bool
            perform thresholding
        doRw: bool
            do reweighting (default: self.doRw)
        nnegS: bool
            apply non-negativity constraint on the sources (default: self.nnegS)
        useMad : bool
            estimate noise std in wavelet domain with MAD (else: analytical estimation). Default: self.useMad
        stds: np.ndarray
            noise std in wavelet domain (overrides useMad). Default: None
        oracle: bool
            perform an oracle update (using the ground-truth A and S)

        Returns
        -------
        int
            error code
        """

        if nnegS is None:
            nnegS = self.nnegS
        if doRw is None:
            doRw = self.doRw

        ls_s = self.ls_s(strat, c, do_cg=do_cg, oracle=oracle)
        if ls_s:  # error caught
            return 1

        self.constraints_s(doThr, K, doRw, nnegS, useMad=useMad, stds=stds, oracle=oracle)

        return 0

    def ls_s(self, strat, c, do_cg=False, oracle=False):
        """Perform the (approximate) least-square update of the sources. First the w-modulated sources expressed in
        Fourier domain are estimated, then the sources are retrieved.

        Parameters
        ----------
        strat: int
            regularization strategy (0: constant, 1: mixing-matrix-based, 2: spectrum-based)
        c: float
            regularization hyperparameter
        do_cg: bool
            refine estimation of the sources with conjugate gradient
        oracle: bool
            perform an oracle update (using the ground-truth A and S)

        Returns
        -------
        int
            error code
        """

        if self.verb >= 3:
            if strat == 0:
                regstrat = 'constant'
            elif strat == 2:
                regstrat = 'spectrum-based'
            else:
                regstrat = 'mixing-matrix-based'
            print("Regularization strategy: " + regstrat + " - hyperparameter: c = %e  " % c)

        if not oracle:
            A = self.A
            SpS = np.abs(self.Swfft) ** 2
        else:
            A = self.A0
            SpS = np.abs(self.Swfft0) ** 2

        SpS = np.maximum(SpS, np.max(SpS * 1e-16))

        normAA = np.linalg.norm(A.T @ A, ord=-2)

        AHNvarHA = np.einsum('ijk,il,im->jklm', np.divide(np.abs(self.H) ** 2, self.Var,
                                                          out=np.zeros_like(np.abs(self.H)), where=self.Var != 0), A, A)

        # 1: estimate w-modulated sources in Fourier domain (i.e., Swfft)

        if strat == 0:  # constant regularization coefficients
            iAHNvarHA = AHNvarHA + c * np.eye(self.n)[np.newaxis, np.newaxis, :, :]
        elif strat == 2:  # spectrum-based regularization coefficients
            eps = np.zeros((self.w, self.p, self.n, self.n))
            diag = np.arange(self.n)
            eps[:, :, diag, diag] = c * np.moveaxis(1 / SpS, 0, -1)
            iAHNvarHA = AHNvarHA + eps
        else:  # mixing-matrix-based regularization coefficients
            iAHNvarHA = AHNvarHA + np.maximum(0,
                                              c - np.linalg.norm(AHNvarHA, -2, axis=(-2, -1)) /
                                              (normAA + 1e-3))[:, :, np.newaxis, np.newaxis] * \
                        np.eye(self.n)[np.newaxis, np.newaxis, :, :]
        try:
            iAHNvarHA = np.linalg.inv(iAHNvarHA)
        except np.linalg.LinAlgError as e:
            print('Error! ' + str(e) + ' (during least-squares of S)')
            return 1
        Swfft = np.einsum('ijkl,ml,mij->kij', iAHNvarHA, A,
                          np.divide(self.H.conj(), self.Var, out=np.zeros_like(self.H), where=self.Var != 0) * self.X)

        # 2: retrieve sources from Swfft

        B = np.einsum('ijkl,ijlm->ijkm', iAHNvarHA, AHNvarHA) - np.eye(self.n)[np.newaxis, np.newaxis, :, :]

        if strat != 2 and not oracle:  # estimate the spectra of S with a power law fit
            SpS = np.abs(Swfft[:, self.w // 2, :]) ** 2
            for i in range(self.n):
                SpS[i, :] = utils.fit_power_law(SpS[i, :])

            weights_mse = np.einsum('ijkl,lj,ijlk->kij', B, SpS, B) + \
                          np.abs(np.einsum('ijkl,ijlm,ijmk->kij', iAHNvarHA, AHNvarHA, iAHNvarHA))

        else:
            former_Swfft = self.Swfft.copy() if not oracle else self.Swfft0.copy()
            weights_mse = np.abs(np.einsum('ijkl,lij,mij,ijkm->kij', B, former_Swfft,
                                           former_Swfft.conj(), B)) + \
                          np.abs(np.einsum('ijkl,ijlm,ijmk->kij', iAHNvarHA, AHNvarHA, iAHNvarHA))

        weights_mse = np.maximum(weights_mse, np.max(weights_mse * 1e-16))
        weights_mse = 1 / weights_mse

        normed_weights_mse = weights_mse / np.sum(weights_mse, axis=1)[:, np.newaxis, :]

        S = np.real(np.sum(np.conj(self.G)[np.newaxis, :, :] * fftt.ifft(normed_weights_mse * Swfft), axis=1))

        # Renormalize sources
        if oracle:
            norm_data_model = np.linalg.norm(
                self.H * np.einsum('ij,jkl', self.A0, fftt.fft(self.G[np.newaxis, ...] * S[:, np.newaxis, :])))
        else:
            norm_data_model = np.linalg.norm(
                self.H * np.einsum('ij,jkl', self.A, fftt.fft(self.G[np.newaxis, ...] * S[:, np.newaxis, :])))
        corr_gain = np.linalg.norm(self.X) / norm_data_model
        if self.verb >= 4:
            print('Correcting gain: ', corr_gain)
        S *= corr_gain
        Swfft *= corr_gain

        if not self.useMad or not self.useMad_end:
            Swfft_var = np.abs(np.einsum('ijkl,ijlm,ijmk->kij', iAHNvarHA, AHNvarHA, iAHNvarHA))
            self.stds = np.sqrt(np.sum(
                self.wt_filters[np.newaxis, np.newaxis, ...] ** 2 * (normed_weights_mse ** 2 * Swfft_var)[
                    ..., np.newaxis], axis=(1, 2)) / self.size ** 4)  # caution, this formula is approximate

        # Affine source estimate with conjugate gradient
        if do_cg and self.itCG > 0:
            sqrt_weights_mse = np.sqrt(weights_mse)

            def forward_operator(vec_s):
                return (sqrt_weights_mse * fftt.fft(
                    self.G[np.newaxis, :, :] * np.reshape(vec_s, (self.n, 1, -1)))).flatten()

            def forward_conj_operator(vec_swfft):
                return np.real(np.sum(self.G.conj()[np.newaxis, :, :] *
                                      fftt.ifft(sqrt_weights_mse * np.reshape(vec_swfft, (self.n, self.w, -1))),
                                      axis=1)).flatten()

            def bi_forward_operator(vec_s):
                return forward_conj_operator(forward_operator(vec_s))

            linearoperator = LinearOperator(shape=(self.n * self.size ** 2, self.n * self.size ** 2),
                                            matvec=bi_forward_operator, dtype='complex')

            itCG_real_current = 0

            def callback(xk):
                nonlocal itCG_real_current
                itCG_real_current += 1

            res, info = cg(linearoperator, forward_conj_operator((sqrt_weights_mse * Swfft).flatten()),
                           x0=S.flatten(), maxiter=self.itCG, callback=callback)

            if info < 0:
                raise ValueError('CG did not converge')
            if self.verb >= 4:
                print('CG: convergence reached in %i iterations' % itCG_real_current)

            S = np.real(np.reshape(res, (self.n, self.size ** 2)))

        self.S = S

        return 0

    def constraints_s(self, doThr, K, doRw, nnegS, useMad=None, stds=None, oracle=False):
        """Apply the constraints on the sources (thresholding in the wavelet domain and possibly a projection on the
        positive orthant).

        Parameters
        ----------
        doThr : bool
            perform thresholding
        K: float
            L0 support of the sources
        doRw: bool
            do reweighting
        nnegS: bool
            apply non-negativity constraint on the sources
        useMad : bool
            estimate noise std in source domain with MAD (else: analytical estimation). Default: self.useMad
        stds: np.ndarray
            noise std in wavelet domain (overrides useMad). Default: False
        oracle: bool
            perform an oracle update (using the ground-truth A and S)

        Returns
        -------
        int
            error code
        """

        if not doThr and not nnegS:  # nothing to do
            return 0

        if useMad is None:
            useMad = self.useMad

        if not oracle:
            Swtrw = self.Swtrw
        else:
            Swtrw = self.S0wt

        if doThr:

            if self.verb >= 3:
                print("Maximal L0 norm of the sources: %.1f %%" % (K * 100))

            Swt = fftt.wt_trans(self.S, nscales=self.nscales)

            # Thresholding
            for i in range(self.n):
                for j in range(self.nscales):
                    Swtij = Swt[i, :, j]
                    Swtrwij = Swtrw[i, :, j]
                    if stds is None:
                        if useMad:
                            std = utils.mad(Swtij)
                        else:
                            std = self.stds[i, j]
                    else:
                        std = stds[i, j]
                    thrd = self.k * std

                    # If oracle, threshold Swtrw
                    if oracle and self.L1 and doRw:
                        Swtrwij = (Swtrwij - np.sign(Swtrwij) * (thrd - np.sqrt(np.abs(
                            (Swtrwij - thrd * np.sign(Swtrwij)) * (3 * thrd * np.sign(Swtrwij) + Swtrwij))))) / 2 * (
                                          np.abs(Swtrwij) >= thrd)

                    # Support based threshold
                    if K != 1:
                        npix = np.sum(abs(Swtij) - thrd > 0)
                        if int(K * npix) < np.maximum(5, int(self.p * 0.001)) \
                                and ((self.verb == 4 and i == 0) or self.verb > 4):
                            print('Too few pixels (minimum: %i, current: %i)'
                                  % (np.maximum(5, int(self.p * 0.001)), int(K * npix)))
                        Kval = np.maximum(int(K * npix), np.maximum(5, int(self.p * 0.001)))
                        thrd = np.partition(abs(Swtij), self.p - Kval)[self.p - Kval]

                    if self.verb == 4 and i == 0:
                        print("Threshold of source %i at scale %i: %.5e" % (i + 1, j + 1, thrd))
                    elif self.verb == 5:
                        print("Threshold of source %i at scale %i: %.5e" % (i + 1, j + 1, thrd))

                    # Adapt the threshold if reweighing demanded
                    if doRw and self.L1:
                        thrd = thrd / (np.abs(Swtrwij) / np.maximum(1e-20, self.k * std) + 1)
                    else:
                        thrd = thrd * np.ones(self.p)

                    # Apply the threshold
                    Swtij[(abs(Swtij) < thrd)] = 0
                    if self.L1:
                        indNZ = np.where(abs(Swtij) > thrd)[0]
                        Swtij[indNZ] = Swtij[indNZ] - thrd[indNZ] * np.sign(Swtij[indNZ])

                    Swt[i, :, j] = Swtij

        # Reconstruct S
        self.S = fftt.wt_rec(Swt)

        # Non-negativity constraint
        if nnegS:
            nneg = self.S > 0
            self.S *= nneg

        # Update Swfft and SpS accordingly
        self.Swfft = fftt.fft(self.G[np.newaxis, :, :] * self.S[:, np.newaxis, :])
        self.Swfft_det = fftt.fft(self.G[np.newaxis, :, :] * fftt.wt_rec(Swt[:, :, :self.nscales])[:, np.newaxis, :])

        if self.H_reconv is not None:  # If H_reconv provided, reconvolve sources for upcoming update of A
            self.Swfft_reconv = fftt.fft(
                self.G[np.newaxis, :, :] * np.real(fftt.convolve(self.S, self.H_reconv))[:, np.newaxis, :])
            self.Swfft_det_reconv = fftt.fft(self.G[np.newaxis, :, :] * np.real(
                fftt.convolve(fftt.wt_rec(Swt[:, :, :self.nscales])[:, np.newaxis, :], self.H_reconv)))

        if oracle:
            return 0

        # Save the wavelet coefficients of S for next iteration
        if doThr and doRw and self.L1 and not oracle:
            if nnegS:
                Swt *= nneg[:, :, np.newaxis]
            self.Swtrw = Swt[:, :, :-1]

        return 0

    def update_a(self, oracle=False):
        """Perform the least-square update of the mixing matrix.

        Parameters
        ----------
        oracle: bool
            perform an oracle update (using the ground-truth S)

        Returns
        -------
        int
            error code
        """

        if oracle:
            Swfft = self.Swfft0_det if self.H_reconv is None else self.Swfft0_det_reconv
        else:
            Swfft = self.Swfft_det if self.H_reconv is None else self.Swfft_det_reconv

        weights = np.divide(np.sum(self.wt_filters[:, :self.nscales], axis=1)[np.newaxis, np.newaxis, :] ** 2,
                            self.Var,
                            out=np.zeros_like(self.Var), where=self.Var != 0)
        if self.H_reconv is not None:
            weights = weights * self.H_reconv[np.newaxis, np.newaxis, :] ** 2

        if self.removeCoarseScaleData:
            X = self.X * np.sum(self.wt_filters[:, :self.nscales], axis=1)[np.newaxis, np.newaxis, :]
        else:
            X = self.X

        HSS = np.einsum('ijk,ljk,mjk', weights * np.abs(self.H) ** 2, Swfft, Swfft.conj())
        YHS = np.einsum('ijk,ljk', weights * X * np.conj(self.H), Swfft.conj())
        try:
            self.A = np.real(np.linalg.solve(HSS, YHS))
        except np.linalg.LinAlgError as e:
            print('Error! ' + str(e) + ' (during least-squares of A)')
            return 1

        # Non-negativity constraint
        if self.nnegA:
            sign = np.sign(np.sum(self.A, axis=0))
            sign[sign == 0] = 1
            self.A *= sign
            self.A = np.maximum(self.A, 0)

        # Oblique constraint
        self.A /= np.maximum(np.linalg.norm(self.A, axis=0), 1e-24)
        return 0

    def refine_s_end(self):
        """Perform the finale refinement of the sources, with K = self.K_end.

        Returns
        -------
        int
            error code"""

        if self.verb >= 2:
            print("Finale refinement of the sources with the finale estimation of A...")

        if self.cstWuRegStr:
            strat = 0
        else:
            strat = 1
        c = np.min(self.c_wu)

        update_s = self.update_s(strat, c, self.K_end, doThr=self.thr_end, doRw=False)
        if update_s:  # error caught
            return 1

        # Initialize attributes
        self.Swtrw = np.zeros((self.n, self.p, self.nscales))
        S_old = np.zeros((self.n, self.p))
        delta_S = np.inf
        it = 0

        if not self.keepWuRegStr:
            strat = 2
            c = self.c_end

        while delta_S >= self.eps[2] and it < 25:
            it += 1

            update_s = self.update_s(strat, c, self.K_end, doThr=self.thr_end, useMad=self.useMad_end,
                                     doRw=self.doRw_end)
            if update_s:  # error caught
                return 1

            delta_S = np.linalg.norm(S_old - self.S) / np.linalg.norm(self.S)
            S_old = self.S.copy()

            if self.A0 is not None and self.S0 is not None:
                Acp, Scp, optInd = utils.corr_perm(self.A0, self.A, self.S, optInd=True)
                if self.verb >= 2:
                    print("SNMSE = %.2f" % utils.nmse(self.S0, Scp))

            if self.verb >= 2:
                print("delta_S = %.2e" % delta_S)

        return 0

    def terminate(self):
        """If ground-truth data provided, correct permutations and evaluate solution.

        Returns
        -------
        int
            error code
        """

        if self.A0 is not None and self.S0 is not None:
            self.metrics = utils.evaluate(self.A0, self.S0, self.A, self.S)
            if self.verb:
                print('SAD : %.2f dB | SNMSE: %.2f dB' % (self.metrics['SAD'], self.metrics['SNMSE']))

        return 0

    def oracle(self, strat=2, c=.5, do_cg=False, withThr=False, stds=None, which='both'):
        """Solve the oracle problems (estimate A/S using the ground-truth S/A).

        Parameters
        ----------
        strat: int
            regularization strategy (0: constant, 1: mixing-matrix-based, 2: power-spectrum-based)
        c: float
            regularization hyperparameter
        do_cg: bool
            refine estimation of the sources with conjugate gradient
        withThr: bool
            estimate A with the thresholded sources
        stds: np.ndarray
            noise std in wavelet domain (otherwise, determined based on data). Default: None
        which: str
            parameters which are evaluated ('both': A and S, 'S': S, 'A': A)

        Returns
        -------
        int
            error code
        """

        which = which.lower()
        self.metrics = {}

        if which == 'both' or which == 's':
            update_s = self.update_s(strat, c, 1, do_cg=do_cg, doRw=self.doRw_end, useMad=self.useMad_end, oracle=True)
            if not update_s:  # no error caught
                self.metrics = {**self.metrics,
                                **utils.evaluate(self.A0, self.S0, self.A, self.S, which='S', do_corr_perm=False)}
            elif which == 's':
                return 1

        if which == 'both' or which == 'a':
            if not withThr:
                update_a = self.update_a(oracle=True)
            elif stds is not None:
                S_copy = self.S.copy()
                self.S = self.S0.copy()
                self.constraints_s(True, self.K_max, self.doRw, self.nnegS, oracle=True, stds=stds)
                update_a = self.update_a()
                self.S = S_copy
            else:
                S_copy = self.S.copy()
                useMad_end_copy = self.useMad_end
                self.useMad_end = False
                ls_s = self.ls_s(strat, c, do_cg=do_cg, oracle=True)  # determine stds with ls_s (stored in self.stds)
                self.useMad_end = useMad_end_copy
                if ls_s:  # error caught
                    update_a = 1
                else:
                    self.S = self.S0.copy()
                    self.constraints_s(True, self.K_max, self.doRw, self.nnegS, useMad=self.useMad, oracle=True)
                    update_a = self.update_a()
                self.S = S_copy
            if not update_a:  # no error caught
                self.metrics = {**self.metrics,
                                **utils.evaluate(self.A0, self.S0, self.A, self.S, which='A', do_corr_perm=False)}
            elif which == 'a':
                return 1

        return 0

    def find_optc(self, strat=2, c_lim=None, do_cg=False, tol=1e-3):
        """Grid search of the optimal regularization hyperparameter.

        Parameters
        ----------
        strat: int
            regularization strategy (0: constant, 1: mixing-matrix-based, 2: spectrum-based)
        c_lim: np.ndarray
            zone of the grid search, in log10 scale (default: np.array([-5, 1]))
        do_cg: bool
            refine estimation of the sources with conjugate gradient
        tol: float
            tolerance of the optimal hyperparameter

        Returns
        -------
        (float, float)
            optimal regularization parameter,
            associated NMSE
        """

        if c_lim is None:
            c_lim = np.array([-5., 1.])

        c_min = 10. ** c_lim[0]
        c_max = 10. ** c_lim[1]
        c_mid = 10. ** ((c_lim[0] + c_lim[1]) / 2)

        while True:
            oracle_res = self.oracle(strat, c_mid, do_cg=do_cg, which='S')
            if oracle_res == 0:
                nmse_mid = self.metrics['SNMSE']
                break
            print('Warning, hyperparameters too low! Consequently, lower limit set to (c_lim[0]+c_lim[1])/2')
            c_min, c_mid = c_mid, 10. ** ((np.log10(c_min) + 3 * np.log10(c_max)) / 4)

        it = 0
        while it < 50 and c_max / c_min > 1 + tol:
            it += 1
            c_a = 10. ** ((3 * np.log10(c_min) + np.log10(c_max)) / 4)
            c_b = 10. ** ((np.log10(c_min) + 3 * np.log10(c_max)) / 4)
            oracle_res = self.oracle(strat, c_a, do_cg=do_cg, which='S')
            nmse_a = self.metrics['SNMSE'] if not oracle_res else -np.inf
            self.oracle(strat, c_b, do_cg=do_cg, which='S')  # since c_b > c_a, should converge
            nmse_b = self.metrics['SNMSE']
            if self.verb >= 2:
                print("NMSE : %.2f | %.2f | %.2f " % (nmse_a, nmse_mid, nmse_b))
            if nmse_mid <= nmse_b:
                c_min, c_mid = c_mid, c_b
                nmse_mid = nmse_b
            elif nmse_mid < nmse_a:
                c_mid, c_max = c_a, c_mid
                nmse_mid = nmse_a
            else:
                c_min, c_max = c_a, c_b
            if self.verb:
                print("Min bound:", c_min, "& max bound:", c_max)

        if c_mid / 10. ** c_lim[0] < 1.01 or 10. ** c_lim[1] / c_mid < 1.01:
            print("Warning! Opt c is near the boundary of the search zone (strat=%i)" % strat)

        return c_mid, nmse_mid
