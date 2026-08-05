import marimo

__generated_with = "0.23.16"
app = marimo.App()

with app.setup:
    import marimo as mo


@app.cell
def _():
    from turtle import width

    sty = {"width": "5em", "display": "inline-block"}
    # .style(**sty)
    n1 = mo.ui.number()
    n2 = mo.ui.number()
    n3 = mo.ui.number()

    s = f"""
    <style>
    .fick {{color: blue;}}
    .buttock {{width:5em; display:inline-block; text-align:left}}
    </style>

    I am trying my best.

    <span class="fick">
    But, **sometimes** the best is not good enough.
    </span>

    <span class=buttock>
    {n1}
    </span>
    <span class=buttock>
    {n2} {n3}
    </span>
    """
    f = mo.md(s)
    f
    return n1, n2, n3


@app.cell
def _(n1, n2, n3):
    (n1.value, n2.value, n3.value)
    return


@app.cell
def _():
    n =  mo.ui.number()
    n
    return (n,)


@app.cell
def _(n):
    n.value
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
