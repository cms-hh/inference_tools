# coding: utf-8

import numpy as np

from dhi.config import poi_labels, campaign_labels
from dhi.util import import_plt, rgb
from scipy.interpolate import interp1d

plt = import_plt()


def plot_likelihood1d(path, poi, data, campaign):
    deltaNLL = 2 * data["delta_nll"]
    poi_values = data[poi]
    interpol = interp1d(poi_values, deltaNLL)
    precision = poi_values.size * 100
    fine_poi = np.linspace(poi_values.min(), poi_values.max(), precision)
    fine_deltaNLL = interpol(fine_poi)
    # 1 sigma interval:
    idx1 = np.argwhere(np.diff(np.sign(fine_deltaNLL - np.full(fine_deltaNLL.shape, 1)))).flatten()
    # 2 sigma interval:
    idx2 = np.argwhere(np.diff(np.sign(fine_deltaNLL - np.full(fine_deltaNLL.shape, 4)))).flatten()
    best_fit_value = fine_poi[np.argmin(fine_deltaNLL)]

    fig, ax = plt.subplots()
    for idx in idx1:
        ax.plot(
            (fine_poi[idx], fine_poi[idx]),
            (0.0, fine_deltaNLL[idx]),
            linestyle="--",
            color=rgb(161, 16, 53),
        )
    for idx in idx2:
        ax.plot(
            (fine_poi[idx], fine_poi[idx]),
            (0.0, fine_deltaNLL[idx]),
            linestyle="--",
            color=rgb(161, 16, 53),
        )
    ax.plot(
        poi_values,
        deltaNLL,
        linestyle="-",
        marker=".",
        color=rgb(0, 84, 159),
        label=r"expected (best %s=%.2f)" % (poi_labels[poi], best_fit_value),
    )

    # legend, labels, titles, etc
    ax.set_xlabel(poi_labels[poi])
    ax.set_ylabel(r"$-2 \Delta \text{ln}\mathcal{L}$")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    ax.set_title(r"\textbf{CMS} \textit{Preliminary}", loc="left")
    ax.set_title(campaign_labels[campaign], loc="right")
    ax.grid()

    print("best fit value for {}: {:.2f}".format(poi, best_fit_value))
    print("1 sigma uncertainties: {}".format(fine_poi[idx1]))
    print("2 sigma uncertainties: {}".format(fine_poi[idx2]))

    # save
    fig.tight_layout()
    fig.savefig(path)
