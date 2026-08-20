import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", css_file="", auto_download=["html"])

with app.setup:
    import marimo as mo
    import numpy as np
    import scipy as sp
    import plotly.graph_objects as go
    import plotly.colors as pc


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Free vibration response
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This notebook computes the response of an undamped free vibration system with $N$ degrees of freedom, governed by the system of differential equations

    \[ \boldsymbol m  \ddot{\boldsymbol{x}}(t) + \boldsymbol k  \boldsymbol{x}(t) = \boldsymbol{0},\]

    with $\boldsymbol{x}(0) = \boldsymbol{x}_0$ and  $\dot{\boldsymbol{x}}(0) = \dot{\boldsymbol{x}}_0$.
    """)
    return


@app.cell
def _():
    _theory = r"""
    /// details | Theory summary
        type: danger

    The solution can be written as a linear combination of the modal shapes multiplied by the corresponding harmonic term,
    \[ \boldsymbol{x}(t) = \sum_i^N
    \boldsymbol{X}^{(i)}  A_i  \cos(\omega_i t + \phi_i), \]
    where  $\omega_i$ and $\boldsymbol{X}^{(i)}$ are a natural frequency and its corresponding mode shape, and $A_i$ and $\phi_i$ are the amplitude and phase of the $i$ harmonic, to be determined.

    Taking the derivative with regard to time,
    \[ \dot{\boldsymbol{x}}(t) = -\sum_i^N
    \boldsymbol{X}^{(i)}  A_i \omega_i  \sin(\omega_i t + \phi_i), \]
    which is a system of linear equations for the velocities.

    There are $2N$ unknowns, $A_i$ and $\phi_i$, for $i=1,\ldots, N$.
     Using $t=0$ in the above two vector equations, we have

    \[ \boldsymbol{x}(0) = \boldsymbol{x}_0 = \sum_i^N
    \boldsymbol{X}^{(i)}  A_i  \cos(\phi_i); \qquad
    \dot{\boldsymbol{x}}(0) = \dot{\boldsymbol{x}}_0 = -\sum_i^N \boldsymbol{X}^{(i)}  A_i \omega_i  \sin(\phi_i), \]

    With known values for $\boldsymbol{x}_0$ and $\dot{\boldsymbol{x}}_0$, we have
    linear systems with $N$ equations each, so the problem is well posed.
    ///
    """
    mo.md(_theory)
    return


@app.cell(hide_code=True)
def _():
    _example = r"""
    /// details | Example
        type: danger
    For a system with three degrees of freedom, we have

    \[ \boldsymbol{x}_0 =
    \boldsymbol{X}^{(1)}  A_1  \cos( \phi_1) +
    \boldsymbol{X}^{(2)}  A_2  \cos( \phi_2) +
    \boldsymbol{X}^{(3)}  A_1  \cos( \phi_3),\]

    expanding the vectors,

    \[
    \begin{bmatrix}
    {x_0}_1 \\
    {x_0}_2 \\
    {x_0}_3
    \end{bmatrix} =
    \begin{bmatrix}
    {X}^{(1)}_1 \\
    {X}^{(1)}_2 \\
    {X}^{(1)}_3
    \end{bmatrix}
     A_1  \cos( \phi_1) +
    \begin{bmatrix}
    {X}^{(2)}_1 \\
    {X}^{(2)}_2 \\
    {X}^{(2)}_3
    \end{bmatrix}
     A_2  \cos( \phi_2) +
    \begin{bmatrix}
    {X}^{(3)}_1 \\
    {X}^{(3)}_2 \\
    {X}^{(3)}_3
    \end{bmatrix}
     A_3  \cos( \phi_3),
     \]

    \[
    \begin{bmatrix}
    {x_0}_1 \\
    {x_0}_2 \\
    {x_0}_3
    \end{bmatrix} =
    \begin{bmatrix}
    {X}^{(1)}_1 A_1  \cos( \phi_1) + {X}^{(2)}_1 A_2  \cos( \phi_2) +{X}^{(3)}_1 A_3  \cos( \phi_3)\\
    {X}^{(1)}_2  A_1 \cos( \phi_1) + {X}^{(2)}_2 A_2  \cos( \phi_2) +{X}^{(3)}_2 A_3  \cos( \phi_3)\\
    {X}^{(1)}_3 A_1  \cos( \phi_1) + {X}^{(2)}_3 A_2  \cos( \phi_2) +{X}^{(3)}_3 A_3  \cos( \phi_3)\\
    \end{bmatrix} ,
    \]

    \[
    \begin{bmatrix}
    {x_0}_1 \\
    {x_0}_2 \\
    {x_0}_3
    \end{bmatrix} =
    \begin{bmatrix}
    {X}^{(1)}_1 & {X}^{(2)}_1 & {X}^{(3)}_1 \\
    {X}^{(1)}_2 & {X}^{(2)}_2 & {X}^{(3)}_2 \\
    {X}^{(1)}_3 & {X}^{(2)}_3 & {X}^{(3)}_3 \\
    \end{bmatrix}
    \begin{bmatrix}
    A_1  \cos( \phi_1) \\
    A_2  \cos( \phi_2) \\
    A_3  \cos( \phi_3)
    \end{bmatrix},
    \]
    which is a system of equations that can be solved for $A_i\cos\phi_i$. Note that the coeficient matrix is the modal matrix.

    With a similar procedure for the velocity system, we get
    \[
    \begin{bmatrix}
    {\dot x_0}_1 \\
    {\dot x_0}_2 \\
    {\dot x_0}_3
    \end{bmatrix} =
    \begin{bmatrix}
    {X}^{(1)}_1 & {X}^{(2)}_1 & {X}^{(3)}_1 \\
    {X}^{(1)}_2 & {X}^{(2)}_2 & {X}^{(3)}_2 \\
    {X}^{(1)}_3 & {X}^{(2)}_3 & {X}^{(3)}_3 \\
    \end{bmatrix}
    \begin{bmatrix}
    -A_1 \omega_1 \sin( \phi_1) \\
    -A_2 \omega_2 \sin( \phi_2) \\
    -A_3 \omega_3 \sin( \phi_3)
    \end{bmatrix},
    \]
    which can be solved for $-A_i\omega_i\sin\phi_i$.

    With this vector we change the sign and divide by $\omega_i$, getting $A_i \sin\phi_i$, and then compute the desired amplitudes and phases with
    \[ A_i = \sqrt{(A_i \cos\phi_i)^2+ (A_i \sin\phi_i)^2}; \qquad \tan\phi_i = \frac{A_i \sin\phi_i}{A_i \cos\phi_i}.\]
    ///
    """
    mo.md(_example)
    return


@app.cell
def _():
    ndof = mo.ui.number(
        value=2,
        start=2,
        stop=8,
        step=1,
        label="Number of degrees of freedom: ",
    )
    prob = mo.md(f"""
         Enter the number of degrees of freedom.
         Please note that changing the number of degrees of freedom will reset the whole computation and
         erase the entries in the matrices below!

         {ndof}
         """)
    prob
    return (ndof,)


@app.cell
def _make_template():
    # OK. Inlining all css styles is an abomination.
    # I'm not proud of this, but I couldn't get it working any other way
    def wrap_col(idx: str) -> str:
        templ = (
            '<span class="inline-box "'
            + 'style="display:inline-block; width:6em; vertical-align:top;  box-sizing:border-box;"> '
            + f"{{{idx}}} </span>"
        )
        return templ

    def wrap_row(row: list[str], row_lbl: str) -> str:
        entries = [wrap_col(item) for item in row]
        div = '<div class="container" style="margin: 0 auto;">\n'
        return div + f"{row_lbl}: " + "\n".join(entries) + "\n</div>"

    def make_template_mat(ndof: int) -> str:
        tmpl = []
        for row in range(1, ndof + 1):
            row_l = [f"i{row}{col}" for col in range(1, row + 1)]
            row_s = wrap_row(row_l, f"{row:>2}")
            tmpl.append(row_s)
        return "\n".join(tmpl)

    def make_template_vec(ndof: int) -> str:
        tmpl = []
        row_l = [f"v{col}" for col in range(1, ndof + 1)]
        row_s = wrap_row(row_l, " 1")
        tmpl.append(row_s)
        return "\n".join(tmpl)

    return make_template_mat, make_template_vec


@app.cell
def _():
    def make_ctrls_mat(ndof: int) -> dict[str, mo.ui.number]:
        return {
            f"i{row}{col}": mo.ui.number(value=0.0, full_width=True)
            for row in range(1, ndof + 1)
            for col in range(1, row + 1)
        }

    def make_ctrls_vec(ndof: int) -> dict[str, mo.ui.number]:
        return {
            f"v{col}": mo.ui.number(value=0.0, full_width=True)
            for col in range(1, ndof + 1)
        }

    return make_ctrls_mat, make_ctrls_vec


@app.cell
def _():
    def make_mat(ndof: int, mat: dict[str, float]) -> np.ndarray:
        m = np.zeros((ndof, ndof), dtype=np.float32)
        for row in range(1, ndof + 1):
            for col in range(1, row + 1):
                r = row - 1
                c = col - 1
                idx = f"i{row}{col}"
                m[r, c] = m[c, r] = mat[idx]
        return m

    def make_vec(ndof: int, vec: dict[str, float]) -> np.ndarray:
        v = np.zeros((ndof,), dtype=np.float32)
        for row in range(1, ndof + 1):
            r = row - 1
            v[r] = vec[f"v{row}"]
        return v

    return make_mat, make_vec


@app.cell
def _():
    def vec2ltx(vec: np.ndarray) -> str:
        """Create a latex representation of a numpy vector"""
        l: list[str] = ["\\begin{bmatrix}"]
        for v in vec:
            l.append(rf"{v:.3f} \\")
        l.append("\\end{bmatrix}")
        return " ".join(l)

    def mat2ltx(mat: np.ndarray) -> str:
        """Create a latex representation of a numpy 2D matrix"""
        l: list[str] = ["\\begin{bmatrix}"]
        nr, nc = mat.shape
        for row in np.arange(nr):
            c = []
            for col in np.arange(nc):
                c.append(f"{mat[row, col]:.3f}")
            l.append(" & ".join(c) + r"\\")
        l.append("\\end{bmatrix}")
        return "\n".join(l)

    def texvec(tmpl: str, ndof: int) -> str:
        """Create a latex representation a latex string
        String must be parameterized by the index "i".
        """
        l: list[str] = [r"\begin{bmatrix}" + "\n"]
        for i in range(1, ndof + 1):
            l.append(tmpl.format(i=i) + r" \\")
        l.append("\n" + r"\end{bmatrix}")
        return " ".join(l)

    return mat2ltx, texvec, vec2ltx


@app.cell
def _(
    make_ctrls_mat,
    make_ctrls_vec,
    make_template_mat,
    make_template_vec,
    ndof,
):
    _nd = ndof.value
    _mc = make_ctrls_mat(_nd)
    _kc = make_ctrls_mat(_nd)

    _mmd = mo.md(("***Mass matrix***:\n\n" + make_template_mat(_nd)))
    _kmd = mo.md(("***Stiffness matrix***:\n\n" + make_template_mat(_nd)))

    _um = _mmd.batch(**_mc)
    _uk = _kmd.batch(**_kc)

    _u0c = make_ctrls_vec(_nd)
    _v0c = make_ctrls_vec(_nd)

    _u0md = mo.md("***Initial Displacements***:\n\n" + make_template_vec(_nd))
    _v0md = mo.md("***Initial Velocities***:\n\n" + make_template_vec(_nd))

    _u0b = _u0md.batch(**_u0c)
    _v0b = _v0md.batch(**_v0c)

    _form_ui = """
    Please enter mass and stiffness matrices.

    Enter only the lower triangular factor, the upper factor will be filled in using symmetry.

    Use dots or commas for decimal numbers according to your operating system configuration!

    If you don't see any results after submitting matrices, check your input, you probably have negative eigenvalues.

    {mass}

    {stif}

    Please enter the initial conditions as transposed vectors.

    {u0}

    {v0}
    """

    _form_ctls = {"mass": _um, "stif": _uk, "u0": _u0b, "v0": _v0b}

    mats = mo.md(_form_ui).batch(**_form_ctls).form()
    mats
    return (mats,)


@app.cell
def _(make_mat, make_vec, mats, ndof):
    mo.stop(mats.value is None, mo.md("**Submit the form to continue.**"))
    mass = make_mat(ndof.value, mats.value["mass"])
    stif = make_mat(ndof.value, mats.value["stif"])
    u0 = make_vec(ndof.value, mats.value["u0"])
    v0 = make_vec(ndof.value, mats.value["v0"])
    return mass, stif, u0, v0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Natural frequencies and mode shapes

    We compute the natural frequencies solving the generalized eigenvalue problem

    $$\boldsymbol{k} \boldsymbol{X} = \omega^2 \boldsymbol{m}\boldsymbol{X},$$

    using the [SciPy](https://scipy.org) function [scipy.linalg.eigh](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.linalg.eigh.html).
    """)
    return


@app.cell
def _(mass, stif):
    o2, X = sp.linalg.eigh(stif, mass)
    w = np.sqrt(o2)
    return X, o2, w


@app.cell(hide_code=True)
def _(o2, w):
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
def _(X, ndof, vec2ltx):
    _l = ["### Modal shapes\n Modal shapes are normalized to norm 1.\n\n"]
    for _w in range(1, ndof.value + 1):
        _l.append(f"$X^{{({_w})}} =$")
        _l.append(r"$" + vec2ltx(X[:, _w - 1]) + r"$")
        _l.append("&nbsp; &nbsp;")
    _s = " ".join(_l)
    mo.md(_s)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Displacements
    """)
    return


@app.cell
def _(X, mat2ltx, ndof, texvec, u0, vec2ltx):
    Acl = texvec(r"A_{i}\cos\phi_{i}", ndof.value)
    _m1 = mat2ltx(X)
    _b1 = vec2ltx(u0)
    _text = (
        r"""
    ### Linear equation system for $A_i \cos\phi_i$
    """
        + rf"""
    \[ {_m1} {Acl} = {_b1} \]
    """
    )

    mo.md(_text)
    return (Acl,)


@app.cell
def _(X, mat2ltx, ndof, texvec, v0, vec2ltx):
    Asl = texvec(r"-A_{i}\omega_{i}\sin\phi_{i}", ndof.value)
    _m1 = mat2ltx(X)
    _b1 = vec2ltx(v0)
    _text = (
        r"""
    ### Linear equation system for $-A_i \omega_i \sin\phi_i$
    """
        + rf"""
    \[ {_m1} {Asl} = {_b1} \]
    """
    )

    mo.md(_text)
    return (Asl,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Both systems can be solved simultaneoulsy because they have the same coeficient matrix. Using [np.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html), we get
    """)
    return


@app.cell
def _(Acl, Asl, X, u0, v0, vec2ltx):
    _b = np.hstack((u0[:, np.newaxis], v0[:, np.newaxis]))
    _xt = np.linalg.solve(X, _b)
    Acos = _xt[:, 0]
    Asin = _xt[:, 1]

    _text = rf"""
    \[ {Acl} = {vec2ltx(Acos)}; \qquad {Asl} = {vec2ltx(Asin)}. \]
    """
    mo.md(_text)
    return Acos, Asin


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Computing the amplitudes and phases, we have
    """)
    return


@app.cell
def _(Acos, Asin, vec2ltx, w):
    A = np.sqrt(Acos**2 + (Asin / w) ** 2)
    phi = np.atan2(-Asin / w, Acos)

    _text = rf"""
    \[ \boldsymbol{{A}} = {vec2ltx(A)}; \qquad \boldsymbol{{\phi}} = {vec2ltx(phi)}. \]
    """
    mo.md(_text)
    return A, phi


@app.cell
def _(A, X, ndof, phi, vec2ltx, w):
    Xa = X * A[np.newaxis, :]
    _suml = [
        vec2ltx(Xa[:, _i]) + rf"&\cos({w[_i]:.3f} t {phi[_i]:+.3f})"
        for _i in np.arange(ndof.value)
    ]
    _sums = r" + {{}} \\ ".join(_suml)
    _text = rf"""
    ## Displacements and velocities

    The displacement are given by

    \[ \boldsymbol{{x}}(t) = \sum_i^N
    \boldsymbol{{X}}^{{(i)}}  A_i  \cos(\omega_i t + \phi_i), \]

    so, in this case, we have


    \begin{{split}}
    \boldsymbol{{x}}(t)   = {_sums}.
    \end{{split}}

    """
    mo.md(_text)
    return (Xa,)


@app.cell
def _(A, X, ndof, phi, vec2ltx, w):
    Xva = X * A[np.newaxis, :] * w[np.newaxis, :]
    _suml = [
        vec2ltx(Xva[:, _i]) + rf"&\sin({w[_i]:.3f} t {phi[_i]:+.3f})"
        for _i in np.arange(ndof.value)
    ]
    _sums = r" + {{}} \\ ".join(_suml)
    _text = rf"""
    The velocities are  given by

    \[ \dot{{\boldsymbol{{x}}}}(t) = -\sum_i^N
    \boldsymbol{{X}}^{{(i)}}  A_i \omega_i  \sin(\omega_i t + \phi_i), \]

    so, in this case, we have


    \begin{{split}}
    \dot{{\boldsymbol{{x}}}}(t) = {_sums}.
    \end{{split}}

    """
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Graphical representation

    ### Displacements
    """)
    return


@app.cell
def _():
    nper = mo.ui.slider(value=3, start=1, step=1, stop=15, label="Cycles: ")
    return (nper,)


@app.cell
def _():
    cmap = pc.qualitative.Vivid
    return (cmap,)


@app.cell
def _(Xa, cmap, ndof, nper, phi, w):
    _dofs = np.arange(0, ndof.value, dtype=np.int16)

    def calc_x(
        nper: int,
        w: np.ndarray[tuple[int]],
        ndof: int,
        Xa: np.ndarray[tuple[int]],
        phi: np.ndarray[tuple[int]],
    ):
        nt = 100
        tau1 = 2.0 * np.pi / w[0]
        times = np.linspace(0, nper * tau1, nt * nper, dtype=np.float32)
        xt = np.zeros((ndof, nt * nper))
        for _m in range(ndof):
            xt += Xa[:, _m].reshape(ndof, 1) * np.cos(
                w[_m, np.newaxis] * times + phi[_m]
            )
        return xt, times

    xmax, _ = calc_x(nper.stop, w, ndof.value, Xa, phi)
    maxval = np.max(np.abs(xmax)) * 1.1

    xt, _times = calc_x(nper.value, w, ndof.value, Xa, phi)
    _data = [
        go.Scatter(
            x=_times,
            y=xt[_d, :],
            mode="lines",
            name=f"{_d + 1}",
            line=dict(color=cmap[_d]),
        )
        for _d in _dofs
    ]
    _fig = go.Figure(data=_data)

    _fig.update_layout(
        title=dict(text="Displacements"),
        yaxis=dict(
            title=dict(text="Generalized Displacement"),
            range=[-maxval, maxval],
        ),
        xaxis=dict(
            title=dict(text="Time (s)"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            title_text="DoF",
        ),
    )
    mo.vstack([nper, _fig])
    return maxval, xt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Animation

    Please understand this is just a conceptual visualization of the variation of the amplitude of the generalized coordinates in time.

    **In no way** this is to be interpreted as a representation of the movement of the components of the sistem in the physical space! It is impossible to do that knowing only the mass and stiffness matrices!
    """)
    return


@app.cell
def _(cmap, maxval, ndof, xt):
    _dof = np.arange(1, ndof.value + 1)
    _data = [
        go.Scatter(
            x=_dof,
            y=xt[:, 0],
            mode="markers",
            marker=dict(
                size=25, color=_dof - 1, colorscale=cmap[: ndof.value]
            ),
        )
    ]

    _fig = go.Figure(
        data=_data,
    )

    _fig.update_layout(
        title=dict(text="Displacements"),
        yaxis=dict(
            title=dict(text="Generalized Displacement"),
            range=[-maxval, maxval],
        ),
        xaxis=dict(
            title=dict(text="Degree of freedom"),
            tickmode="array",
            tickvals=_dof,
            range=[0.0, ndof.value + 1],
        ),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        args=[
                            None,
                            {
                                "frame": {"duration": 70, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 10},
                            },
                        ],
                        label="Play",
                        method="animate",
                    ),
                    dict(
                        label="Stop",
                        method="animate",
                        args=[
                            [None],  # Clears the current frame queue
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
            )
        ],
    )

    _fig.update(
        frames=[
            go.Frame(
                data=[
                    go.Scatter(
                        x=_dof,
                        y=xt[:, _t],
                    )
                ],
                traces=list(range(ndof.value)),
            )
            for _t in range(xt.shape[1])
        ]
    )

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
