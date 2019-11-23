import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from matplotlib import colors

MAX_ITER = 100
DPI = 72


@jit
def mandelbrot(z, horizon, log_horizon):
    c = z
    for i in range(MAX_ITER):
        az = abs(z)
        if az > horizon:
            return i - np.log(az) / np.log(2) + log_horizon
        z = z ** 2 + c
    return 0


@jit
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height):
    horizon = 2.0 ** 40
    log_horizon = np.log(horizon) / np.log(2)

    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))

    for i in range(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[j] + 1j * r2[i], horizon, log_horizon)
    return (r1, r2, n3)


def mandelbrot_image(xmin, xmax, ymin, ymax, width=10, height=10,
                     cmap='jet', gamma=0.3):
    DPI = 72
    img_width = DPI * width
    img_height = DPI * height
    x, y, z = mandelbrot_set(xmin, xmax, ymin, ymax, img_width, img_height)

    fig, ax = plt.subplots(figsize=(width, height), dpi=DPI)
    ticks = np.arange(0, img_width, 3*DPI)
    xticks = xmin + (xmax-xmin)*ticks/img_width
    plt.xticks(ticks, xticks)
    yticks = ymin + (ymax-ymin)*ticks/img_width
    plt.yticks(ticks, yticks)
    ax.set_title(cmap)

    norm = colors.PowerNorm(gamma)
    ax.imshow(z, cmap=cmap, origin='lower', norm=norm)

    plt.show()


mandelbrot_image(-2.0, 0.5, -1.25, 1.25, cmap='summer', width=20, height=20)
