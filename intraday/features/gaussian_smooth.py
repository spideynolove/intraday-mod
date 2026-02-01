from typing import Sequence, Union
from collections import OrderedDict
import math
from intraday.frame import Frame
from intraday.feature import Feature


class GaussianSmooth(Feature):
    def __init__(
        self, radius: int, source: Union[str, Sequence[str]] = "vwap"
    ):
        super().__init__(write_to="frame", period=2 * radius + 1)
        assert isinstance(radius, int) and radius > 0
        self.radius = radius
        if isinstance(source, str):
            self.source = [source]
        elif isinstance(source, Sequence):
            self.source = source
        else:
            raise ValueError
        for name in self.source:
            assert isinstance(name, str)
            self.names.append(f"gauss_{radius}_{name}")
        self.spaces = OrderedDict()

    def process(self, frames: Sequence[Frame], state: OrderedDict):
        kernel = self._get_gaussian_kernel(self.radius)
        N = len(frames)
        R = self.radius
        P = 2 * R + 1
        i = N - R - 1
        if i < 0:
            return
        for j, name in enumerate(self.source):
            value = 0.0
            for k in range(P):
                if i - R + k < 0:
                    v = getattr(frames[0], name)
                elif i - R + k >= N:
                    v = getattr(frames[-1], name)
                else:
                    v = getattr(frames[i - R + k], name)
                value += kernel[k] * v
            setattr(frames[i], self.names[j], value)

    _gaussian_kernels = {}

    @staticmethod
    def _get_gaussian_kernel(radius: int):
        assert isinstance(radius, int) and radius > 0
        if radius in GaussianSmooth._gaussian_kernels:
            return GaussianSmooth._gaussian_kernels[radius]
        sigma = radius / 2
        norm = 1.0 / (math.sqrt(2 * math.pi) * sigma)
        coeff = 2 * sigma * sigma
        total = 0.0
        kernel = []
        for x in range(-radius, radius + 1):
            g = norm * math.exp(-x * x / coeff)
            kernel.append(g)
            total += g
        for i in range(len(kernel)):
            kernel[i] = kernel[i] / total
        kernel = tuple(kernel)
        GaussianSmooth._gaussian_kernels[radius] = kernel
        return kernel

    def __repr__(self):
        return f"{self.__class__.__name__}(radius={self.radius}, source={self.source})"
