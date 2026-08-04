import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    from matplotloom import Loom

    return Loom, np, plt


@app.cell
def _(Loom, np, plt):
    with Loom("sine_wave_animation.mp4", fps=30, overwrite=True) as loom:
        for phase in np.linspace(0, 2 * np.pi, 100):
            fig, ax = plt.subplots()

            x = np.linspace(0, 2 * np.pi, 200)
            y = np.sin(x + phase)

            ax.plot(x, y)
            ax.set_xlim(0, 2 * np.pi)

            loom.save_frame(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
