import random
import math
from typing import List, Tuple


class Histogram:
    def __init__(self, eps: float, max_value: float, num_buckets: int, pick_buckets: int):
        self.eps = eps
        self.max_value = max_value
        self.num_buckets = num_buckets
        self.pick_buckets = pick_buckets

    def estimate(self, responses: List[List[Tuple[int, int]]]) -> List[float]:
        histogram: List[float] = [0.0] * self.num_buckets
        n = len(responses)
        e_eps_half = math.exp(self.eps / 2)
        scale = self.num_buckets / (n * self.pick_buckets)
        for bits in responses:
            for (bucket, bit) in bits:
                top = float(bit)*(e_eps_half + 1) - 1
                bot = e_eps_half - 1
                histogram[bucket] += scale * top / bot
        return histogram


class HistogramClient:
    def __init__(self, histogram: Histogram):
        self.histogram = histogram
        self.buckets = random.sample(
            population=range(histogram.num_buckets),
            k=histogram.pick_buckets
        )

    def d_bit_flip(self, value: float) -> List[Tuple[int, int]]:
        """Implementation of dBitFlip
        Returns d bits, one for each of the client's picked buckets, indicating whether the value is in said bucket
        Each bit is returned as the tuple (bucket, bit) where bucket is the bucket index"""
        histo = self.histogram
        # TODO: remove, do better bucket calculation that doesn't expect [0;max_value]
        value = min(max(value, 0.0), histo.max_value)
        bucket_size = histo.max_value / histo.num_buckets
        real_bucket = int(value / bucket_size)

        e_eps_half = math.exp(histo.eps / 2)
        bits: List[Tuple[int, int]] = []

        for i in self.buckets:
            top = e_eps_half if i == real_bucket else 1.0
            bot = e_eps_half + 1
            probability = top/bot
            val: Tuple[int, int] = (
                i,
                1 if random.random() < probability else 0,
            )
            bits.append(val)

        return bits


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from typing import Tuple
    import numpy as np

    eps = 1
    max_value = 10
    num_buckets = int(1e2)
    pick_buckets = int(1e1)

    histogram = Histogram(eps=eps, max_value=max_value,
                          num_buckets=num_buckets, pick_buckets=pick_buckets)

    populations = [int(n) for n in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]]
    epsilons = [0.1, 1, 5]
    values = [np.random.randn(n)+5 for n in populations]
    clients = [HistogramClient(histogram) for _ in range(max(populations))]

    def private_hist(ax, eps: float, values: List[float]):
        """Draws a histogram of values after being sent through our private histogram estimation"""
        print("Private hist for {} values at eps={}".format(len(values), eps))
        old_eps = histogram.eps
        histogram.eps = eps
        bits = [
            clients[i].d_bit_flip(values[i])
            for i in range(len(values))
        ]
        histo = histogram.estimate(bits)
        histogram.eps = old_eps
        ax.bar(
            range(len(histo)),
            histo,
            width=1.0
        )

    # set up plot
    fig, axs = plt.subplots(nrows=len(populations),
                            ncols=len(epsilons)+1, figsize=(12, 12))
    axs[0, 0].set_title("Real values")
    for i in range(len(epsilons)):
        axs[0, i+1].set_title("eps={}".format(epsilons[i]))
    for i in range(len(populations)):
        axs[i, 0].set_ylabel("n={}".format(populations[i]))
    # draw real values
    for i in range(len(values)):
        axs[i, 0].hist(values[i], bins=int(num_buckets), color="green")
    # draw private values
    for x in range(len(epsilons)):
        for y in range(len(values)):
            private_hist(axs[y, x+1], eps=epsilons[x], values=values[y])

    fig.tight_layout()
    fig.show()
