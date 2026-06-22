import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    k11 = mo.ui.number().style({"width": "85px"})
    k22 = mo.ui.number().style({"width": "85px"})
    k21 = mo.ui.number().style({"width": "85px"})
    return k11, k21, k22


@app.cell
def _(k11, k21, k22, mo):
    k = mo.vstack(
        [
            mo.hstack([k11], justify="start"),
            mo.hstack([k21, k22], justify="start"),
        ]
    )
    k
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
