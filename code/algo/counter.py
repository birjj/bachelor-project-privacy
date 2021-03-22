import math
import random
from typing import List


class Counter:
    def __init__(self, eps: float, max_value: float):
        self.eps = eps
        self.max_value = max_value

    def one_bit(self, value: float) -> int:
        """Implementation of 1BitMean
        Returns 0/1 representing the value in a private way"""
        e_eps = math.exp(self.eps)
        top = self.max_value + value*e_eps - value
        bot = self.max_value * e_eps + self.max_value
        prob = top / bot
        return 1 if random.random() <= prob else 0

    def mean(self, bits: List[int]) -> float:
        """Implementation of mean estimation
        Returns the estimated mean from a set of bits gathered from 1BitMean"""
        e_eps = math.exp(self.eps)
        n = len(bits)
        s = sum(
            [(val*(e_eps+1) - 1) / (e_eps - 1) for val in bits]
        )
        return (self.max_value / n) * s


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from typing import Tuple

    max_value = 1e6

    def calc(n: int, eps: float) -> Tuple[float, float]:
        """Generates n random values, and calculates both their actual and their estimated mean"""
        counter = Counter(eps, max_value)
        values = [random.uniform(0, max_value) for _ in range(n)]
        bits = [counter.one_bit(values[i]) for i in range(n)]
        return (
            sum(values) / n,
            counter.mean(bits)
        )

    populations = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    epsilons = [0.1, 0.5, 1]

    fig, axs = plt.subplots(nrows=len(epsilons), figsize=(12, 12), sharey=True)
    fig.suptitle("Mean estimation (max={})".format(max_value))
    for i in range(len(epsilons)):
        eps = epsilons[i]
        results = [calc(int(n), eps) for n in populations]
        y_1 = [y[0] for y in results]
        y_2 = [y[1] for y in results]

        p = axs[i]
        p.set_title("eps={}".format(eps))
        p.set_xlabel("Population size")
        p.set_xscale("log")
        p.set_ylabel("Mean")
        p.plot(populations, y_1, label="Real mean")
        p.plot(populations, y_2, label="Private mean")
        p.legend()

    fig.tight_layout()
    fig.show()
