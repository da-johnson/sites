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
ecdf = iqplot.ecdf(data=df, q="value", cats="variable", style="staircase", conf_int=True)

stripbox = iqplot.stripbox(
    data=df,
    q='value',
    cats=['variable'],
    jitter=True,
    show_legend=True,
    frame_height=350,
    frame_width=300,
    marker_kwargs = {'alpha' : 0.2},
)

bokeh.io.show(bokeh.layouts.gridplot([ecdf, stripbox], ncols = 2))
