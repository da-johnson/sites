# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
rg = np.random.default_rng()

def ecdf(x, data):
    """Give the value of an ECDF at arbitrary points x."""
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))

def log_like_iid_gamma(params, n):
    """Log likelihood for i.i.d. Gamma measurements."""
    alpha, beta = params

    if alpha <= 0 or beta <= 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(n, alpha, 1/beta))

def log_like_iid_mix(params, n):
    """Log likelihood for i.i.d. Mixture model."""
    b1, b2 = params
    
    return np.sum(st.expon.logpdf(n, b1) + st.expon.logpdf(n, b2))

def mle_iid_gamma(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    Gamma measurements, parametrized by alpha, b=1/beta"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_gamma(params, n),
            x0=np.array([1, 0.0001]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def draw_parametric_bs_reps_mle(
    mle_fun, gen_fun, data, args=(), size=1, progress_bar=False
):
    """Draw parametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(*params, size)`.
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    params = mle_fun(data, *args)

    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array(
        [mle_fun(gen_fun(*params, size=len(data), *args)) for _ in iterator]
    )

def draw_parametric_bs_reps_mle_two(
    mle_fun, gen_fun, data, args=(), size=1, progress_bar=False
):
    """Draw parametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(*params, size)`.
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    params = mle_fun(data)
    
    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array(
        [mle_fun(gen_fun(*params, size = len(data))) for _ in iterator]
    )

def gen_gamma(alpha, beta, size):
    return rg.gamma(alpha, 1/beta, size = size)

def gen_b1_b2(beta_1, beta_2, size):
    return rg.exponential(1/beta_1, size = size) + rg.exponential(1/beta_2, size = size)

def b1_b2_mle(data):
    b1, b2 = np.array([2 / data.mean()]*2)
    return b1, b2


# +
bs_reps_parametric_two = draw_parametric_bs_reps_mle_two(
    b1_b2_mle,
    gen_b1_b2,
    vals_12,
    args=(),
    size=10000,
    progress_bar=True,
)
conf_int_2 = np.percentile(bs_reps_parametric_two, [2.5, 97.5], axis=0)

print('MLEs for the Mixture model')
print('95% confidence interval for α: ' + str(conf_int_2[0][0]) + ' — ' + str(conf_int_2[1][0]))
print('95% confidence interval for β: ' + str(conf_int_2[0][1]) + ' — ' + str(conf_int_2[1][1]))
