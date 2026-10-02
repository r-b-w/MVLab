import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Resposta Forçada Amortecida
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// details | Resumo da teoria
        type: danger
    Resposta total de um sistema em vibração forçada amortecida com 1GL, para sistemas com amortecimento subcrítico.

    A resposta total é:
    $$x(t) = x_h(t) + x_p(t),$$
    com
    $$x_h(t) = X_0 e^{-ζ \omega_n t} \cos(\omega_n t - \phi_0)$$
    e
    $$x_p(t) = X \cos(ω t - \phi).$$
    Temos que
    $$ X = \frac{\delta_\textrm{st}}{\sqrt{(1-r^2)^2 + (2\zeta r)^2}}, \quad \tan \phi = \frac{2\zeta r }{1-r^2}.$$
    Com a introdução das condições de contorno $x_0$ e $\dot x_0$, temos as belezuras abaixo.
    $$X_0 =\left[ (x_0 - X \cos\phi)^2 + \frac{1}{\omega_d^2}(\zeta\omega_n x_0 + \dot x_0  - \zeta\omega_n X \cos\phi -\omega X \sin\phi)^2\right]^\frac{1}{2}, $$
    e
    $$ \tan \phi_0 = \frac{\zeta\omega_n x_0 + \dot x_0 - \zeta\omega_n X \cos\phi - \omega X \sin\phi}{\omega_d(x_0 - X\cos\phi)}.$$
    Poderemos estudar aqui vários fenômenos interessantes: os regimes permanente e transiente, o batimento, a ressonância e outros.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Vamos torcer para que estas fórmulas estejam transcritas corretamente...

    def steady_state(dst, r, zeta):
        """
        Computes amplitude and fase of steady state response;
        """
        X = dst / np.sqrt((1 - r**2) ** 2 + (2 * zeta * r) ** 2)
        phi = np.arctan2(2 * zeta * r, 1 - r**2)
        return X, phi

    def transient_state(x0, v0, zeta, wn, wd, w, X, phi):
        """
        Calcula a "amplitude" e "fase" do regime transiente.
        """
        X02 = (x0 - X * np.cos(phi)) ** 2 + (
            (zeta * wn * x0 + v0 - zeta * wn * X * np.cos(phi) - w * X * np.sin(phi))
            ** 2
            / wd**2
        )
        X0 = np.sqrt(X02)
        dy = zeta * wn * x0 + v0 - zeta * wn * X * np.cos(phi) - w * X * np.sin(phi)
        dx = wd * (x0 - X * np.cos(phi))
        phi0 = np.arctan2(dy, dx)
        return X0, phi0

    return steady_state, transient_state


@app.cell(hide_code=True)
def _():
    im = mo.ui.number(start=0.0, value=1.0, label="Massa $m$ (kg)")
    ik = mo.ui.number(start=0.0, value=40.0, label="Rigidez $k$ (N/m)")
    ic = mo.ui.number(start=0.0, value=0.6, label="Coef. Amortecimento  $c$ (kg/s)")
    iF0 = mo.ui.number(start=0.0, value=10.0, label="Amplitude $F_0$ (kg)")
    iw = mo.ui.number(start=0.0, value=3.0, label="Frequência $\omega$ (rad/s)")
    ix0 = mo.ui.number(value=5.0, label="Deslocamento inicial $x_0$ (m)")
    iv0 = mo.ui.number(value=-1.33, label="Velocidade inicial $\dot x_0$ (m/s)")
    return iF0, ic, ik, im, iv0, iw, ix0


@app.cell
def _(iF0, ic, ik, im, iv0, iw, ix0):
    param_tab = mo.md(rf"""
    {im}
    {ic}
    {ik}
    """)
    cond_tab = mo.md(rf"""
    {ix0}
    {iv0}
    """)
    force_tab = mo.md(rf"""
    {iF0}
    {iw}
    """)
    mo.ui.tabs(
        {
            "Sistema": param_tab,
            "Condições iniciais": cond_tab,
            "Força externa": force_tab,
        }
    )
    return


@app.cell
def _(iF0, ic, ik, im, iv0, iw, ix0):
    m = im.value
    k = ik.value
    c = ic.value
    F0 = iF0.value
    w = iw.value
    x0 = ix0.value
    v0 = iv0.value
    return F0, c, k, m, v0, w, x0


@app.cell
def _(F0, c, k, m, steady_state, transient_state, v0, w, x0):
    wn = np.sqrt(k / m)
    cc = 2 * m * wn
    zeta = c / cc
    wd = np.sqrt(1 - zeta**2) * wn
    r = w / wn
    dst = F0 / k
    X, phi = steady_state(dst, r, zeta)
    X0, phi0 = transient_state(x0, v0, zeta, wn, wd, w, X, phi)
    w_min = min(wn, wd)
    tau = 2 * np.pi / w_min  # período de interesse para plotagem
    return X, X0, cc, dst, phi, phi0, r, tau, wd, wn, zeta


@app.cell
def _(cc, dst, r, wd, wn, zeta):
    dyn_var = mo.md(rf"""
    | $c_c\, (\text{{kg/s}})$ | $\zeta$ | $\omega_n \, (\text{{rad/s}})$|  $\omega_d \, (\text{{rad/s}})$ | $r$ | $\delta_{{st}} \, (\text{{m}})$
    | :---: | :---: | :---: |:---: |:---: |:---: |
    | {cc:.3g} | {zeta:.2g} | {wn:.3g} | {wd:.3g} | {r:.3g} | {dst:.3g} |
    """)

    leg = mo.md("""
    - $c_c = 2 m \omega_n$: coeficiente de amortecimento crítico
    - $\zeta = c/c_c$: razão de amortecimento
    - $\omega_n = \sqrt{k/m}$: frequência natural
    - $\omega_d = \sqrt{1-\zeta^2}\omega_n$ : frequência de vibração livre amortecida
    - $r = \omega/\omega_n$: razão de frequências
    - $\delta_{st} = F_0/k$: deformação estática
    """)
    return dyn_var, leg


@app.cell
def _(dyn_var, leg):
    mo.ui.tabs({"Variáveis Dinâmicas": dyn_var, "Legenda": leg})
    return


@app.cell
def _(X, X0, nper, phi, phi0, tau, w, wn, zeta):
    npc = 100
    times = np.linspace(0, nper.value * tau, nper.value * npc, endpoint=True)
    xh = X0 * np.exp(-zeta * wn * times) * np.cos(wn * times - phi0)
    xp = X * np.cos(w * times - phi)
    sol = xh + xp
    biggie = 1.1 * np.max(np.abs(sol))
    return biggie, sol, times, xh, xp


@app.cell
def _():
    is_x = mo.ui.switch(label=r"$x(t)$", value=True)
    is_xh = mo.ui.switch(label=r"$x_h(t)$", value=True)
    is_xp = mo.ui.switch(label=r"$x_p(t)$", value=True)
    nper = mo.ui.number(start=2, value=10, step=1, label="Número de períodos")
    choices = mo.hstack([is_x, is_xh, is_xp, nper], justify="start")
    return choices, is_x, is_xh, is_xp, nper


@app.cell
def _(biggie, choices, is_x, is_xh, is_xp, sol, times, xh, xp):
    data = [
        (is_x, sol, r"$x$", "green"),
        (is_xh, xh, r"$x_h$", "blue"),
        (is_xp, xp, r"$x_p$", "red"),
    ]
    lines = [
        go.Scatter(x=times, y=y_data, mode="lines", name=label, line=dict(color=color))
        for flag, y_data, label, color in data
        if flag.value
    ]
    _fig = go.Figure(data=lines)

    _fig.update_layout(
        yaxis=dict(title=dict(text=r"Deslocamento (m)"), range=[-biggie, biggie]),
        xaxis=dict(
            title=dict(text="Tempo (s)"),
        ),
        width=900,
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )
    mo.vstack([_fig, choices])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
