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

    eps = 1
    max_value = 1e6

    def calc(n: int) -> Tuple[float, float]:
        counter = Counter(eps, max_value)
        values = [random.uniform(0, max_value) for _ in range(n)]
        bits = [counter.one_bit(values[i]) for i in range(n)]
        return (
            sum(values) / n,
            counter.mean(bits)
        )

    x = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    results = [calc(int(n)) for n in x]
    y_1 = [y[0] for y in results]
    y_2 = [y[1] for y in results]

    plt.title("Mean estimation (eps={}, max={})".format(eps, max_value))
    plt.xlabel("Population size")
    plt.xscale("log")
    plt.ylabel("Mean")
    plt.plot(x, y_1, label="Real mean")
    plt.plot(x, y_2, label="Private mean")
    plt.legend()
    plt.show()
