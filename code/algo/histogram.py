import random
import math
from typing import List, Tuple

import functools


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

        return self.__d_bits(real_bucket)

    @functools.lru_cache(maxsize=None)
    def __d_bits(self, real_bucket: int) -> List[Tuple[int, int]]:
        outp: List[Tuple[int, int]] = []
        e_eps_half = math.exp(self.histogram.eps / 2)
        for i in self.buckets:
            top = e_eps_half if i == real_bucket else 1.0
            bot = e_eps_half + 1
            probability = top/bot
            val: Tuple[int, int] = (
                i,
                1 if random.random() < probability else 0,
            )
            outp.append(val)
        return outp

    def clear_cache(self):
        self.__d_bits.cache_clear()
