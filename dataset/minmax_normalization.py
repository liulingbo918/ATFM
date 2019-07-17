"""
    MinMaxNormalization
"""


class MinMaxNormalization(object):
    """
    MinMax Normalization --> [-1, 1]
    x = (x - min) / (max - min).
    x = x * 2 - 1
    """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()

    def transform(self, x):
        x = 1. * (x - self.min) / (self.max - self.min)
        x = x * 2. - 1.
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = (x + 1.) / 2.
        x = 1. * x * (self.max - self.min) + self.min
        return x
