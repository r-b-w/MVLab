import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", css_file="my.css")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    # For matrix input
    num_style = {"width": "85px"}
    return mo, np, num_style


@app.cell
def _(mo, np, num_style, uctl):
    def make_mat_ui(ndof: int):
        """Creates an interactive ui for matrices.
        I don't like marimo's one because it only uses sliders.
        """
        rows = []
        controls = []
        for irow in range(ndof):
            cols = []
            for col in range(irow + 1):
                v = mo.ui.number(value=0.0)  #
                cols.append(v)
            rows.append(
                mo.hstack([c.style(num_style) for c in cols], justify="start")
            )
            controls.append(cols)
        ui = mo.vstack(rows)
        return ui, controls

    def get_matrix(uctrl, ndof):
        """Tranforms the user interface values into a numpu array."""
        mat = np.array((ndof, ndof), dtype=np.floa32)
        for row in range(ndof):
            for col in range(row):
                mat[row, col] = uctl

    return (make_mat_ui,)


@app.cell
def _(mo):
    prob = (
        mo.md(
            """
        **Problem Parameters**

        Please note that changing these will erase the matrices' entries!

        {ndof} &emsp; {damp}
        """
        )
        .batch(
            ndof=mo.ui.number(
                value=2,
                step=1,
                start=2,
                stop=8,
                label="Number of degrees of freedom:",
            ),
            damp=mo.ui.switch(value=False, label="Damped system?"),
        )
        .form()
    )
    prob
    return (prob,)


@app.cell
def _(make_mat_ui, mo, prob):
    ndof = prob.value["ndof"]
    damped = prob.value["damp"]
    k, k_ctl = make_mat_ui(ndof)
    m, m_ctl = make_mat_ui(ndof)
    matrices = {"Mass Matrix": m, "Stiffness Matrix": k}
    if damped:
        c, c_ctl = make_mat_ui(ndof)
        matrices["Damping Matrix"] = c
    matrices_input = mo.accordion(matrices, multiple=True)
    matrices = (
        mo.md(
            f"""Matrices will be *always* be symmetrized! Enter only the lower triagular part!

        When ready press "Submit" to start computation.

        {matrices_input}
        """
        )
        .batch()
        .form()
    )
    matrices
    return


@app.cell
def _(mo):
    ix = (
        mo.md("""
    Please enter matrix:

    {i11}

    {i21} {i22}

    {i31} {i32} {i33}
    """)
        .batch(
            i11=mo.ui.number(value=0.0),
            i21=mo.ui.number(value=0.0),
            i22=mo.ui.number(value=0.0),
            i31=mo.ui.number(value=0.0),
            i32=mo.ui.number(value=0.0),
            i33=mo.ui.number(value=0.0),
        )
        .form()
    )
    ix
    return (ix,)


@app.cell
def _(ix):
    ix.value
    return


@app.function
def make_template(ndof: int) -> str:
    def wrap_col(idx: str) -> str:
        templ = f'<div class="inline-box"> {{{idx}}} </div>'
        return templ

    def wrap_row(row_idx: int, row: [str]) -> str:
        entries = [wrap_col(item) for item in row]
        row = (
            f'<div class="container">\n'
            + f"{row_idx}: "
            + "\n".join(entries)
            + "\n</div>"
        )
        return row

    tmpl = []
    for row in range(1, ndof + 1):
        row_l = [f"i{row}{col}" for col in range(1, row + 1)]
        row_s = wrap_row(row, row_l)
        tmpl.append(row_s)
    return "\n".join(tmpl)


@app.cell
def _():
    print(make_template(3))
    return


@app.cell
def _(mo):
    def make_ctrls(ndof: int) -> dict[str, mo.ui.number]:
        return {
            f"i{row}{col}": mo.ui.number(value=0.0, full_width=True)
            for row in range(1, ndof + 1)
            for col in range(1, row + 1)
        }

    return (make_ctrls,)


@app.cell
def _(make_ctrls):
    make_ctrls(3)
    return


@app.cell
def _(mo):
    d = {
        "i11": mo.ui.number(value=0.0, full_width=True),
        "i21": mo.ui.number(value=0.0, full_width=True),
        "i22": mo.ui.number(value=0.0, full_width=True),
        "i31": mo.ui.number(value=0.0, full_width=True),
        "i32": mo.ui.number(value=0.0, full_width=True),
        "i33": mo.ui.number(value=0.0, full_width=True),
    }

    iy = (
        mo.md("""Please enter matrix:

    <div class="container">
    <div class="inline-box">
    {i11}
    </div>
    </div>
    <div class="container">
    <div class="inline-box">
    {i21}
    </div>
    <div class="inline-box">
    {i22}
    </div>
    </div>
    <div class="container">
    <div class="inline-box">
    {i31}
    </div>
    <div class="inline-box">
    {i32}
    </div>
    <div class="inline-box">
    {i33}
    </div>
    </div>
    """)
        .batch(**d)
        .form()
    )
    iy
    return


@app.cell
def _(make_ctrls, mo):
    _d = make_ctrls(6)
    md = (
        "Please enter matrix, use commas or dot according to your locale!\n"
        + make_template(6)
    )
    yx = mo.md(md).batch(**_d).form()
    yx
    return (yx,)


@app.cell
def _(yx):
    yx.value
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
