import math
import random
import functools
from typing import Final

import counter

MAX_VAL: Final = 1e6  # max value for counter (m in the paper)
DISC_GRANULARITY: Final = MAX_VAL / 20  # discretization granularity (s)
DISC_COUNT: Final = DISC_GRANULARITY / MAX_VAL


class User:
    def __init__(self, counter: counter.Counter):
        self.counter = counter
        self.alpha = random.randint(0, DISC_GRANULARITY - 1)

    @functools.lru_cache(maxsize=None)
    def __OneBitMean(self, value: int) -> int:
        return self.counter.one_bit(float(value))

    def respond(self, value: float) -> int:
        L = math.floor(value/DISC_GRANULARITY) * DISC_GRANULARITY
        R = L + DISC_GRANULARITY
        if value + self.alpha < R:
            return self.__OneBitMean(L)
        else:
            return self.__OneBitMean(R)
