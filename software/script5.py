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

# +
size = len(vals_12)
alpha, beta = mle_iid_gamma(vals_12)
b_1, b_2 = b1_b2_mle(vals_12)

mix_samples = np.array(
    [gen_b1_b2(b_1, b_2, size=size) for _ in tqdm.tqdm(range(100000))]
)

gamma_samples = np.array(
    [gen_gamma(alpha, beta, size=size) for _ in tqdm.tqdm(range(100000))]
)

gamma_pecdf = bebi103.viz.predictive_ecdf(
    samples=gamma_samples, data=vals_12, discrete=True, x_axis_label="n", title = 'Gamma'
)

mix_pecdf = bebi103.viz.predictive_ecdf(
    samples=mix_samples, data=vals_12, discrete=True, x_axis_label="n", title = 'Mixture'
)

gamma_diff = bebi103.viz.predictive_ecdf(
    samples=gamma_samples, data=vals_12, diff='ecdf', discrete=True, x_axis_label="n", title = 'Gamma'
)

mix_diff = bebi103.viz.predictive_ecdf(
    samples=mix_samples, data=vals_12, diff='ecdf', discrete=True, x_axis_label="n", title = 'Mixture'
)

bokeh.io.show(bokeh.layouts.gridplot([gamma_pecdf, mix_pecdf, gamma_diff, mix_diff], ncols = 2))

# +
df_mle = pd.DataFrame(index=['beta_1_mix', 'beta_2_mix'])
for x in df['variable'].unique():
    n = df.loc[df['variable'] == x, 'value'].values

    # Mixture model MLE
    b_1, b_2 = b1_b2_mle(n)

    # Store results in data frame
    df_mle[x] = [b_1, b_2]

df_mle
