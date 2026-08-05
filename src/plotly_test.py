import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import numpy as np
    import pandas as pd

    return np, pd, px


@app.cell
def _(np, px):
    size = 20
    x = np.arange(0, size)
    y1 = np.sqrt(x)

    _fig = px.line(x=x, y=y1, title="Square Root", markers=True)
    _fig
    return size, x, y1


@app.cell
def _(np, pd, px, size, x, y1):
    y2 = 0.5 * x * np.sin(4 * np.pi * x / size)
    df = pd.DataFrame({"X1": y1, "X2": y2})
    _fig = px.line(df, title=r"\text{Functions} $\sqrt{x} \text{and} x\sin(x)$", markers=True)
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
