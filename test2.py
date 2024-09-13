import sys
import numpy as np


class MyForm():
    def __init__(self):
        x = 2
        y = x
        x = 3
        print(x)
        print(y)

        x = [1, 2]
        y = x
        x = [3, 4]

        print(x)
        print(y)

        x = [1, 2]
        y = x
        x[0] = 3
        print(x)
        print(y)

        x = np.array([1, 2])
        y = x
        x[0] = 3
        print(x)
        print(y)

        x = np.array([[1], [2]])
        y = x
        x[0] = 3
        print(x)
        print(y)


if __name__ == "__main__":
    a = MyForm()