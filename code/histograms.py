from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
import math

from algo.histogram import Histogram, HistogramClient


def gen_uniform(n: int):
    rand_gen = np.random.default_rng()
    return [rand_gen.uniform(0.0, 24.0) for i in range(n)]


def gen_constant(n: int):
    return [12 for i in range(n)]


def gen_normal(n: int):
    rand_gen = np.random.default_rng()
    return [min(24, max(0, rand_gen.normal(12, 2)))
            for i in range(n)]


def err_histo_err(true: List[float], est: List[float]):
    """Calculates the error between a true histogram and an estimated histogram"""
    assert len(true) == len(est)
    return np.amax([abs(est[i] - true[i]) for i in range(len(est))])


def err_wasserstein(true: List[float], est: List[float]):
    def hist_to_values(hist: List[float]) -> List[float]:
        h = [max(0, n) for n in hist]
        total = sum(h)
        outp = []
        for i in range(len(h)):
            outp.extend([i]*math.floor(h[i]/total*1000))
        return outp

    true_vals = hist_to_values(true)
    est_vals = hist_to_values(est)
    return wasserstein_distance(true_vals, est_vals)


def histo_error(epsilons=[0.1, 1.0, 3.0, 10.0],
                populations=[1e3, 1e4, 5*1e4, 1e5, 3*1e5],
                gen=gen_normal,
                measure=err_histo_err,
                num_experiments=10,
                max_value=24, num_buckets=32, pick_buckets=[1, 2, 4],
                ylabel="Error", yscale="log"):
    """Draws a graph of estimation errors for histograms for the given epsilons and population sizes"""

    # styles for each pick_buckets
    styles = ["solid", "dashed", "dotted", "dashdot"]

    print("Drawing errors plot")
    populations = [int(n) for n in populations]
    values = [gen(n) for n in populations]
    true_histograms = []
    for n in range(len(populations)):
        true_nondensity, _ = np.histogram(values[n], bins=int(num_buckets))
        true_histograms.append([k/populations[n] for k in true_nondensity])

    def estimate(eps: float, values: List[float]) -> List[float]:
        old_eps = histogram.eps
        histogram.eps = eps
        bits = [
            clients[i].d_bit_flip(values[i])
            for i in range(len(values))
        ]
        histo = histogram.estimate(bits)
        histogram.eps = old_eps
        for c in clients:
            c.clear_cache()
        return histo

    for e in range(len(epsilons)):
        for D in range(len(pick_buckets)):
            d = pick_buckets[D]
            histogram = Histogram(eps=1, max_value=max_value,
                                  num_buckets=num_buckets, pick_buckets=d)
            clients = [HistogramClient(histogram)
                       for _ in range(max(populations))]
            errors = [0.0 for n in populations]
            for n in range(len(populations)):
                print("Estimating for n={} eps={} d={}".format(
                    populations[n], epsilons[e], d))
                for i in range(num_experiments):
                    est = estimate(epsilons[e], values[n])
                    errors[n] += measure(true_histograms[n], est)
                errors[n] /= num_experiments
            plt.plot(populations, errors,
                     color="C{}".format(e), linestyle=styles[D],
                     label=("eps={}".format(
                            epsilons[e])) if D == 0 else "_nolegend_",
                     alpha=1.0 if D == 0 else 0.5)
    plt.xscale("log")
    plt.xlabel("Population size")
    plt.yscale(yscale)
    plt.ylabel(ylabel)
    plt.legend()


def histo_matrix(epsilons=[0.1, 1.0, 3.0, 10.0],
                 populations=[1e3, 1e4, 5*1e4, 1e5, 3*1e5],
                 gen=gen_normal,
                 max_value=24, num_buckets=32, pick_buckets=1):
    """Draws a matrix of private histograms, one column for each eps, one row for each population"""

    print("Drawing histogram matrix")
    populations = [int(n) for n in populations]
    histogram = Histogram(eps=1, max_value=max_value,
                          num_buckets=num_buckets, pick_buckets=pick_buckets)
    values = [gen(n) for n in populations]
    clients = [HistogramClient(histogram) for _ in range(max(populations))]

    def draw_hist(ax, eps: float, values: List[float]):
        """Sends our values through our histogram estimation, then draws the histogram"""
        print("Drawing histogram n={}, eps={}".format(len(values), eps))
        old_eps = histogram.eps
        histogram.eps = eps
        bits = [
            clients[i].d_bit_flip(values[i])
            for i in range(len(values))
        ]
        histo = histogram.estimate(bits)
        histogram.eps = old_eps
        for c in clients:
            c.clear_cache()
        ax.bar(
            [n * 24/len(histo) for n in range(len(histo))],
            histo,
            width=1.0
        )

    # set up plot
    fig, axs = plt.subplots(nrows=len(populations),
                            ncols=len(epsilons)+1, figsize=(2.5*len(epsilons), 2.5*len(populations)))
    axs[0, 0].set_title("True values")
    for i in range(len(epsilons)):
        axs[0, i+1].set_title("eps={}".format(epsilons[i]))
    for i in range(len(populations)):
        axs[i, 0].set_ylabel("n={:,}".format(populations[i]).replace(",", " "))
    # draw real values
    for i in range(len(values)):
        axs[i, 0].hist(values[i], bins=int(num_buckets),
                       color="green", density=True)
    # draw private values
    for x in range(len(epsilons)):
        for y in range(len(values)):
            draw_hist(axs[y, x+1], eps=epsilons[x], values=values[y])

    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    print(
        """Run one of:
    histo_matrix()
    histo_error()


If not running in Jupyter, edit the file to do so instead
    """)
