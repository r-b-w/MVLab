import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import io


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Análise Modal, Múltiplos Graus de Liberdade
    """)
    return


@app.cell
def _():
    ngl = mo.ui.number(
        step=1, start=2, stop=10, label="Número de graus de liberdade:"
    )
    sw_c = mo.ui.switch(label="Amortecimento")
    sw_diag = mo.ui.switch(label="Massa diagonal")
    params = mo.hstack([ngl, sw_c, sw_diag])
    return ngl, params


@app.cell
def _(params):
    params
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Entre com a submatriz triangular inferior, conforme o padrão mostrado.
    """)
    return


@app.cell
def _(ngl):
    _buf = io.StringIO()
    for _row in range(ngl.value):
        for _col in range(0, _row + 1):
            print(0.0, end=" " if _col < _row else "\n", file=_buf)
    k_def = _buf.getvalue()
    get_k, set_k = mo.state(k_def)
    get_k()
    return get_k, set_k


@app.cell
def _(get_k):
    val = get_k()
    k_text = mo.ui.text_area(value=val)
    k_text
    return (k_text,)


@app.cell
def _(k_text, ngl, set_k):
    set_k(k_text.value)
    k_lines = k_text.value.splitlines()
    k = np.zeros((ngl.value, ngl.value))
    for _irow, _row in enumerate(k_lines):
        k[_irow, : _irow + 1] = np.array(_row.split())
        k[:_irow, _irow] = k[_irow, :_irow]
    k
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
