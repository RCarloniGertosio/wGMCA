# wGMCA

The wGMCA algorithm aims at solving joint deconvolution and blind source separation (DBSS) problems from non-coplanar interferometeric data.

## Contents
1. [Introduction](#intro)
1. [Procedure](#procedure)
1. [Getting started](#getst)
1. [Parameters](#param)
1. [Example](#example)
1. [Authors](#authors)
1. [Reference](#ref)
1. [License](#license)

<a name="intro"></a>
## Introduction

The wGMCA builds upon the *w*-stacking framework [[Offringa et all, 2014]](https://doi.org/10.1093/mnras/stu1368). The *w*-axis is discretized uniformly into ![equation](https://latex.codecogs.com/svg.latex?W) values, and for each channel, the interferometric samples are assigned to their nearest *w*-plane. Next, for each channel and for each *w*-plane, the interferometric samples are gridded along the (*u*,*v*) axes on a uniform grid of size ![equation](https://latex.codecogs.com/svg.image?\sqrt{P}\times\sqrt{P}) and then flattened in a vector of size ![equation](https://latex.codecogs.com/svg.image?P). This leads to the obtaining of a three-dimensional tensor ![equation](https://latex.codecogs.com/svg.latex?\boldsymbol{\mathcal{Y}}%20\in%20\mathbb{R}^{J%20\times%20W\times%20P}), with ![equation](https://latex.codecogs.com/svg.image?J) the number of channels.

The mixing model is described by (see [Reference](#ref) for a more precise formulation):
> ![equation](https://latex.codecogs.com/svg.image?\boldsymbol{\mathcal{Y}}&space;=&space;\boldsymbol{\mathcal{H}}&space;\odot&space;\left[&space;\mathcal{F}\circ\mathcal{G}&space;\right]\left(\mathbf{AS}\right)&space;&plus;&space;\boldsymbol{\mathcal{N}}),

where:
- ![equation](https://latex.codecogs.com/svg.latex?\boldsymbol{\mathcal{H}}%20\in%20\mathbb{R}^{J%20\times%20W\times%20P}) is the interferometer response (in the visibility space),
- ![equation](https://latex.codecogs.com/svg.latex?%5Codot) denotes the element-wise product,
- ![equation](https://latex.codecogs.com/svg.latex?\mathcal{F}) is a tensor operator based on the two-dimensional (fast) Fourier transform,
- ![equation](https://latex.codecogs.com/svg.latex?\mathcal{G}) is a tensor operator which accounts for the non-coplanar effect, 
- ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A}%20\in%20\mathbb{R}^{J%20\times%20I}) is the mixing matrix,
- ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S}%20\in%20\mathbb{R}^{I%20\times%20P}) are the ![equation](https://latex.codecogs.com/svg.latex?I) sources, flattened and stacked in a matrix,
- ![equation](https://latex.codecogs.com/svg.latex?\boldsymbol{\mathcal{N}}%20\in%20\mathbb{R}^{J%20\times%20W\times%20P}) is a complex Gaussian noise with known variances.

The sources are assumed to be sparse in the starlet domain ![equation](https://latex.codecogs.com/svg.latex?\mathbf{W}). Moreover, both the sources and the mixing matrix are supposed to be constituted of nonnegative elements.

The wGMCA algorithm aims at minimizing the following objective function with respect to ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A}) and ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S}):
> ![equation](https://latex.codecogs.com/svg.image?\min_{\mathbf{A},&space;\mathbf{S}}&space;\frac{q}{2}&space;\left(\boldsymbol{\mathcal{Y}}&space;-&space;\boldsymbol{\mathcal{H}}&space;\odot&space;\left[&space;\mathcal{F}\circ\mathcal{G}&space;\right]\left(\mathbf{AS}\right)\right)&space;&plus;&space;\left\lVert\boldsymbol{\Lambda}&space;\odot&space;\left(\mathbf{SW}^\top\right)\right\rVert_1&space;&plus;&space;\iota_{\mathcal{O_A}\cap\mathcal{B_A}}(\mathbf{A})&space;&plus;&space;\iota_\mathcal{O_S}(\mathbf{S}),)

where ![equation](https://latex.codecogs.com/svg.latex?q) is a quadratic form which depends on the noise variance, ![equation](https://latex.codecogs.com/svg.latex?\mathbf{\Lambda}) contains the sparsity regularization parameters, ![equation](https://latex.codecogs.com/svg.latex?\mathcal{O_S}) and
![equation](https://latex.codecogs.com/svg.latex?\mathcal{O_A}) are the nonnegative orthants for sources and mixing matrices, and ![equation](https://latex.codecogs.com/svg.latex?\mathcal{B_A}) ensures that the columns of the mixing matrix have a norm less than or equal to unity.


<a name="procedure"></a>
## Procedure

The wGMCA algorithm is based on GMCA, which is a BSS procedure built upon a projected alternating least-square (pALS) minimization scheme. In brief, the updates of ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A}) and ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S}) comprise a least-square estimate, to minimize the data-fidelity term, followed by the application of the proximal operator of the corresponding regularization term.

In contrast to standard BSS problems, the least-square update of the sources is *(i)* generally intractable and *(ii)* not necessarily stable with respect to noise. Thus, *(i)* the least-square update is decomposed into two simpler updates and *(ii)* an extra Tikhonov regularization is added.

The separation is comprised of two stages. The first stage estimates a first guess of the mixing matrix and the sources **(warm-up)**; it provides robustness with respect to the initial point. The second stage refines the separation by employing more precise strategies **(refinement)**. Lastly, the sources `S` are improved during a finale step with the output mixing matrix.

<a name="getst"></a>
## Getting started

### Requirements

- Python (last tested with v3.7.10)
- NumPy (last tested with v1.20.2)
- SciPy (last tested with v1.7.0)

### wGMCA class

The wGMCA algorithm is implemented in a class `wGMCA`. The data and the algorithm parameters are provided at the object initialization. The DBSS is performed by running the method `run`. The results are stored in the attributes `A` and `S`.


<a name="param"></a>
## Parameters

Below are the five parameters of the `wGMCA` class that must always be provided at initialization.

| Parameter | Type                            | Information                                                                                | Default value            |
|-----------|---------------------------------|--------------------------------------------------------------------------------------------|--------------------------|
| `X`       | (m,w,p) complex numpy.ndarray   | input data (gridded visibilities); 1st axis: channel, 2nd axis: w axis, 3rd axis: flattened (u,v) axes                                         | N/A                      |
| `H`       | (m,w,p) float numpy.ndarray     | interferometer response in visibility domain for several w-values, with zero-frequency shifted to the center and flattened                                         | N/A                      |
| `n`       | int                             | number of sources to be estimated                                                          | N/A                      |
| `Var`     | (m,w,p) float array or float    | noise variance (wGMCA does not account for potential noise covariances)                    | N/A|
| `G`       | (w,p) float array               | w-term matrices                                                                            | N/A|


Below are the essential parameters of the `wGMCA` class. They may be assigned their default value.

| Parameter | Type                            | Information                                                                                | Default value            |
|-----------|---------------------------------|--------------------------------------------------------------------------------------------|--------------------------|
| `nnegA`   | bool                            | non-negativity constraint on ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A})  | True                    |
| `nnegS`   | bool                            | non-negativity constraint on ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S})  | True                    |
| `nneg`    | bool                            | non-negativity constraint on ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A}) and ![equation](https://latex.codecogs.com/svg.latex?\mathbf{S}), overrides nnegA and nnegS if not None   | None                     |
| `c_wu`    | float                           | Tikhonov regularization hyperparameter at warm-up                                          | 0.5   |
| `c_ref`    | float                           | Tikhonov regularization hyperparameter at refinement                                       | 0.5   |
| `c_end`    | float                           | Tikhonov regularization hyperparameter at finale refinement of `S`                        | 0.5   |
| `itCG`    | int                           | maximum number of iterations of the conjugate gradient algorithm during refinement (useful for strong non-coplanar effects)         | 100   |
| `nscales` | int                             | number of starlet detail scales                                                            | 2                      |
| `k`       | float                           | parameter of the k-std thresholding (~1 for approximately sparse sources, 3 for very sparse sources)                                                      | 3                        |
| `K_max`   | float                           | maximal L0 norm of the sources. Being a percentage, it should be between 0 and 1 (small for very sparse sources, 1 for approximately sparse sources)         | 0.5                      |
| `K_end`   | float                     | maximal L0 norm of the sources during finale refinement of `S`. Being a percentage, it should be between 0 and 1         | 1                     |
| `thr_end`  | bool                            | perform thresholding during the finale refinement of `S` (consider False if significant information lies in the coarse scales)                          | True                     |
| `doRw`   | bool                                | do L1 reweighing during refinement (consider False if algorithm unstable)     | True                      |
| `doRw_end`   | bool                                | do L1 reweighing during finale refinement of `S`    | `doRw`       |
| `H_reconv`  | (,p) float numpy.ndarray      | kernel in Fourier domain, with zero-frequency shifted to the center and flattened, by which the sources are reconvolved for the estimation of ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A}) (typically to get the resolution of the first channel)  | None                    |
| `removeCoarseScaleData`  | bool             | remove coarse scale from data for the estimation of ![equation](https://latex.codecogs.com/svg.latex?\mathbf{A})  (implemented very approximately, consider True only if resolution ratio is ≳ 0.5)      | False                     |
| `eps`     | (3,) float numpy.ndarray        | stopping criteria of *(1)* the warm-up, *(2)* the refinement and *(3)* the finale refinement of `S` | [1e-2, 1e-4, 1e-4]       |
| `verb`    | int                             | verbosity level, from 0 (mute) to 5 (most talkative)                                       | 0                        |

Below are other parameters of the `wGMCA` class, which can reasonably be assigned their default value.

| Parameter | Type                            | Information                                                                                | Default value            |
|-----------|---------------------------------|--------------------------------------------------------------------------------------------|--------------------------|
| `AInit`   | (m,n) float numpy.ndarray       | initial value for the mixing matrix. If None, PCA-based initialization                    | None                     |
| `keepWuRegStr`   | bool  | keep warm-up regularization strategy during refinement                                                        | False                     |
| `cstWuRegStr`      | bool                            | use constant regularization coefficients during warm-up                              | False                    |
| `minWuIt`| int                             | minimum number of iterations at warm-up                                                      | 50                      |
| `useMad`| bool             | estimate noise std in source domain with MAD (else: analytical estimation, use with caution)      | False                      |
| `useMad_end` | bool        | estimate noise std in source domain with MAD during finale refinement of S (else: analytical estimation, use with caution)      | False                   |
| `L1`| int            | L1 penalization (else: L0 penalization)                                                   | True                    |
| `S0`| (n,p) float array      | ground truth sources (for testing purposes)         | None              |
| `A0`| (m,n) float array      | ground truth mixing matrix (for testing purposes)         | None              |


<a name="example"></a>
## Example

Perform a DBSS on the gridded data `X` with four sources. The interferometer response is stored in `H` and the data variance in`Var`. 

```python
    wgmca = wGMCA(X, H, 4, Var, K_max=0.5, nscales=3)
    wgmca.run()
    S = wgmca.S.copy()
    A = wgmca.A.copy()
```

<a name="authors"></a>
## Authors

- Rémi Carloni Gertosio
- Jérôme Bobin

<a name="ref"></a>
## Reference

*TODO: add ref*

<a name="license"></a>
## License

This project is licensed under the LGPL-3.0 License.
