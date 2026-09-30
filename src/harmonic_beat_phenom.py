import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Batimento

    Vamos examinar o que acontece quando somamos duas funções harmônicas, com mesma amplitude e frequências levemente diferentes,

    \[ x_1(t) = X \cos(\omega t); \quad
    x_2(t) = X \cos\left((1+\delta)\omega t\right).
    \]

    A constante $\delta$ é a variação relativa da frequência, e, para que o batimento aconteça, deve ser pequena, da ordem de 0,1.

    *Atenção:* No livro, $\delta$ é uma variação absoluta! Usamos uma variação relativa aqui pois é mais de controlar e entender.
    """)
    return


@app.cell
def _():
    delta = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.1,
        step=0.01,
    )
    return (delta,)


@app.cell(hide_code=True)
def _(delta):
    mo.md(rf"""
    ## Visualização

    Para ilustração, vamos somar duas senoides de amplitude unitária, com uma frequência que realce o efeito de batimento na tela.

    Selecione aqui a variação de frequência, $\delta$ = {delta.value:.2f} {delta}
    """)
    return


@app.cell
def _(delta):
    w = 1.0  # base frequency
    δ = delta.value  # Frequency variation

    tau = 2 * np.pi / w
    nper = 20
    ntp = 51
    nsamp = nper * ntp
    t = np.linspace(0, nper * tau, nsamp)  # time variable

    x1 = np.cos(w * t)
    x2 = np.cos((1 + δ) * w * t)
    x = x1 + x2
    return t, x, x1, x2


@app.cell
def _(t, x, x1, x2):
    fig, ax = plt.subplots()
    fig.set_size_inches((10, 4))
    ax.set_title("Sum of cosines")
    ax.plot(t, x1, alpha=0.3)
    ax.plot(t, x2, alpha=0.3)
    ax.plot(t, x)
    return


@app.cell
def _():
    d2 = mo.ui.slider(
        start=0.0,
        stop=0.1,
        value=0.0,
        step=0.0001,
    )
    return (d2,)


@app.cell(hide_code=True)
def _(d2):
    mo.md(rf"""
    ## Visualização sonora :)

    Este efeito também pode ser verificado acusticamente. Um ouvido humano jovem e saudável percebe sons na faixa de frequência de 20Hz a 20KHz, aproximadamente, e com muita variabilidade individual.

    Os autofalantes de telefones celulares, monitores e caixinhas de som normalmente estão longe de cobrir esta faixa toda, no entanto. Vamos usar uma frequência base de 440Hz, que é a frequência usada como base na [afinação de instrumentos musicais.](https://en.wikipedia.org/wiki/A440_(pitch_standard))

    Vamos repetir o procedimento acima, somar uma senoide base, com frequência igua a 440Hz, com uma outra senoide, de frequência $(1+\delta)$Hz.

    Selecione aqui a variação de frequência, $\delta$ = {d2.value:.4f} {d2}
    """)
    return


@app.cell
def _(d2):
    _sr = 22050  # Sampling rate

    def _():
        ω = 440.0  # base frequency
        δ = 0.04  # Frequency variation

        T = 25  # Playing time, seconds
        t = np.linspace(0, T, int(T * _sr))  # time variable

        x1 = np.cos(2 * np.pi * ω * t)
        x2 = np.cos(2 * np.pi * (1.0 + d2.value) * ω * t)
        return x1 + x2

    tone = _()
    mo.audio(tone, rate=_sr)
    return


if __name__ == "__main__":
    app.run()
