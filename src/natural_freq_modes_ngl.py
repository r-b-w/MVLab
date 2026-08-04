import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", css_file="my.css")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    from matplotloom import Loom

    return Loom, mo, np, plt, sp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Natural Frequencies and normal modes
    """)
    return


@app.cell
def _(mo):
    ndof = mo.ui.number(
        value=2,
        start=2,
        stop=8,
        step=1,
        label="Number of degrees of freedom: ",
    )
    prob = mo.md(
        f"""
         This notebook computes the natural frequencies and modal shapes (normal modes)
         of an $N$ degree of freedom system. 

         Please note that changing the number of degrees of freedom will reset the whole computation and
         erase the entries in the matrices below!

         {ndof}
         """
    )
    prob
    return (ndof,)


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
def _(make_ctrls, mo, ndof):
    _nd = ndof.value
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
def _(make_mat, mats, mo, ndof):
    mo.stop(mats.value is None, mo.md("**Submit the form to continue.**"))
    mass = make_mat(ndof.value, mats.value["mass"])
    stif = make_mat(ndof.value, mats.value["stif"])
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
def _(mo, ndof, o2, w):
    _nd = ndof.value
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
def _(X, mo, ndof, vec2ltx):
    _l = ["### Modal shapes\n Modal shapes are normalized to norm 1.\n\n"]
    for _w in range(1, ndof.value + 1):
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

    Remember also that the absolute magnitudes don't mean anything, any multiple of a mode shape is also a mode shape.

    This can take a while to generate the animation. Be a little bit patient.
    """)
    return


@app.cell
def _(Loom, X, ndof, np, plt, w):
    _tau1 = 2.0 * np.pi / w[0]
    _modes = range(1, ndof.value + 1)

    with Loom("ndfree_animation.mp4", fps=12, overwrite=True) as loom:
        for _t in np.linspace(0, 2 * _tau1, 100):
            _fig, _ax = plt.subplots()
            for _i in _modes:
                _ax.plot(
                    _modes,
                    X[:, _i - 1] * np.cos(w[_i - 1] * _t),
                    label=f"$X^{{({_i})}}$",
                )
            _fig.set_size_inches((9, 4))
            _ax.legend(loc="upper right")
            _ax.set_ylim(bottom=-1.0, top=1.0)
            _ax.set_xticks(_modes)
            _ax.set_xlabel("Dof")
            _ax.set_ylabel("Generalized Displacement")
            _ax.set_title("Modal Shapes")
            loom.save_frame(_fig)
    return


@app.cell
def _(mo, w):
    _ = w[0]  # Just to force update
    mo.video(src="ndfree_animation.mp4", autoplay=True, loop=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
