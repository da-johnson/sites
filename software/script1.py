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
# comment = '#' lets read_csv() know to skip over the lines that start with a pound sign
df = pd.read_csv('https://s3.amazonaws.com/bebi103.caltech.edu/data/gardner_mt_catastrophe_only_tubulin.csv', comment = '#')

df = df.melt()
df = df.dropna()

vals_12 = df.loc[df['variable'] == '12 uM', 'value'].values`
