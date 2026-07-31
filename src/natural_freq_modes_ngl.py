import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", css_file="my.css")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import scipy as sp

    return mo, np, sp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Natural Frequencies and normal modes
    """)
    return


@app.cell
def _(mo):
    prob = (
        mo.md(
            """

    This notebook computes the natural frequencies and modal shapes of an $N$ degree of freedom system. 

        Please note that changing these will erase the matrices' entries!

        {ndof}
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
        )
        .form()
    )
    prob
    return (prob,)


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
def _(mo):
    def make_ctrls(ndof: int) -> dict[str, mo.ui.number]:
        return {
            f"i{row}{col}": mo.ui.number(value=0.0, full_width=True)
            for row in range(1, ndof + 1)
            for col in range(1, row + 1)
        }

    return (make_ctrls,)


@app.cell
def _(np):
    def make_mat(ndof: int, mat: dict[str, float]) -> np.ndarray:
        m = np.zeros((ndof, ndof), dtype=np.float32)
        for row in range(1, ndof + 1):
            for col in range(1, row + 1):
                r = row - 1
                c = col - 1
                idx = f"i{row}{col}"
                m[r, c] = m[c, r] = mat[idx]
        return m

    return (make_mat,)


@app.cell
def _(np):
    def vec2ltx(vec: np.ndarray) -> str:
        """Create a latex representation of a numpy vector"""
        l = ["$\\begin{bmatrix}"]
        for v in vec:
            l.append(rf"{v:.3f} \\")
        l.append("\\end{bmatrix}$")
        return " ".join(l)

    return (vec2ltx,)


@app.cell
def _(make_ctrls, mo, prob):
    _nd = prob.value["ndof"]
    _mc = make_ctrls(_nd)
    _kc = make_ctrls(_nd)

    _mmd = mo.md(("***Mass matrix***:\n\n" + make_template(_nd)))
    _kmd = mo.md(("***Stiffness matrix***:\n\n" + make_template(_nd)))

    _um = _mmd.batch(**_mc)
    _uk = _kmd.batch(**_kc)

    _form_ui = """
    Please enter mass and stiffness matrices. 

    Enter only the lower triangular factor, the upper factor will be filled in using symmetry.

    Use dots or commas for decimal numbers according to your operating system configuration!

    {mass}

    {stif}
    """

    _form_ctls = {"mass": _um, "stif": _uk}

    mats = mo.md(_form_ui).batch(**_form_ctls).form()
    mats
    return (mats,)


@app.cell
def _(make_mat, mats, prob):
    if mats.value:
        mass = make_mat(prob.value["ndof"], mats.value["mass"])
        stif = make_mat(prob.value["ndof"], mats.value["stif"])
    return mass, stif


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Natural frequencies and mode shapes

    We compute the natural frequencies solving the generalized eigenvalue problem

    $$\boldsymbol{k} \boldsymbol{X} = \omega^2 \boldsymbol{m}\boldsymbol{X},$$

    using the [SciPy](https://scipy.org) function [scipy.linalg.eigh](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.linalg.eigh.html).
    """)
    return


@app.cell
def _(mass, np, sp, stif):
    o2, X = sp.linalg.eigh(stif, mass)
    w = np.sqrt(o2)
    return X, o2, w


@app.cell(hide_code=True)
def _(mo, o2, prob, w):
    _nd = prob.value["ndof"]
    _l = [f"$\\omega_{i} = {w:.3f}$" for i, w in enumerate(w, start=1)]
    _m = ", &nbsp; &nbsp;".join(_l)
    _l2 = [f"$\\omega_{i}^2 = {o2:.3f}$" for i, o2 in enumerate(o2, start=1)]
    _m2 = ", &nbsp; &nbsp;".join(_l2)
    _t = f"""
    ### Squares of the natural frequencies (rad/s)²
    {_m2}

    ### Natural frequencies (rad/s)
    {_m}
    """
    _d = mo.md(_t)
    _d
    return


@app.cell
def _(X, mo, prob, vec2ltx):
    _l = ["### Modal shapes\n Modal shapes are normalized to norm 1.\n\n"]
    for _w in range(1, prob.value["ndof"] + 1):
        _l.append(f"$X^{{({_w})}} =$")
        _l.append(vec2ltx(X[:, _w - 1]))
        _l.append("&nbsp; &nbsp;")
    _s = " ".join(_l)
    mo.md(_s)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graphical representation

    Please understand that these shapes are drawn only to for a quick visual assessment of the relative magnitudes. These values may represent displacements, rotations or other degrees of freedom, and most likely aren't even in the same direction.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
