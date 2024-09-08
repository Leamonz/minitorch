"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List

#
# Implementation of a prelude of elementary functions.
# TODO: Implement for Task 0.1.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """Computes the multiplication of x and y

    Args:
        x (float): x
        y (float): y

    Returns:
        float: x * y
    """
    return x * y


# - id
def id(x: float) -> float:
    """returns x identically

    Args:
        x (float): x

    Returns:
        float: x
    """
    return x


# - add
def add(x: float, y: float) -> float:
    """Computes the summation of x and y

    Args:
        x (float): x
        y (float): y

    Returns:
        float: x + y
    """
    return x + y


# - neg
def neg(x: float) -> float:
    """Negates a number

    Args:
        x (float): x

    Returns:
        float: -x
    """
    return float(-x)


# - lt
def lt(x: float, y: float) -> float:
    """Checks if x is less than y

    Args:
        x (float): x
        y (float): y

    Returns:
        float: 1.0 if x < y else 0.0
    """
    return 1.0 if x < y else 0.0


# - eq
def eq(x: float, y: float) -> float:
    """Checks if x is equal to y

    Args:
        x (float): x
        y (float): y

    Returns:
        float: 1.0 if x == y else 0.0
    """
    return 1.0 if x == y else 0.0


# - max
def max(x: float, y: float) -> float:
    """Returns the maximum of x and y

    Args:
        x (float): x
        y (float): y

    Returns:
        float: x if lt(y, x) else y
    """
    return x if lt(y, x) else y


# - is_close
def is_close(x: float, y: float) -> bool:
    """Checks if x and y are close

    Args:
        x (float): x
        y (float): y

    Returns:
        float: |x - y| < 1e-6
    """
    return abs(x - y) < 1e-6


# - sigmoid
def sigmoid(x: float) -> float:
    """Calculates the sigmoid of x

    Args:
        x (float): x

    Returns:
        float: sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    """Applies ReLU on x

    Args:
        x (float): x

    Returns:
        float: max(0.0, x)
    """
    return max(0.0, x)


# - log
def log(x: float) -> float:
    """Calculates the natural logarithm of x

    Args:
        x (float): x

    Returns:
        float: ln(x)
    """
    assert x != 0
    return math.log(x)


# - exp
def exp(x: float) -> float:
    """Calculates the exponential function

    Args:
        x (float): x

    Returns:
        float: exp(x)
    """
    return math.exp(x)


# - log_back
def log_back(x: float, y: float) -> float:
    """Computes the derivative of log(x) and times y. The derivative of ln(x) is 1 / x

    Args:
        x (float): x
        y (float): y

    Returns:
        float: y / x
    """
    assert x != 0
    return y / x


# - inv
def inv(x: float) -> float:
    """Computes the reciprocal of x

    Args:
        x (float): x

    Returns:
        float: 1 / x
    """
    assert x != 0
    return 1.0 / x


# - inv_back
def inv_back(x: float, y: float) -> float:
    """Computes the derivative of 1 / x and times y. The derivative of 1 / x is -1 / x**2

    Args:
        x (float): x
        y (float): y

    Returns:
        float: -y / x**2
    """
    assert x != 0
    return -y / x**2


# - relu_back
def relu_back(x: float, y: float) -> float:
    """Computes the derivative of relu(x) and times y. This acts like a gate controlled by x.

    Args:
        x (float): x
        y (float): y

    Returns:
        float: y if x > 0 else 0
    """
    return y if x > 0 else 0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# TODO: Implement for Task 0.3.


# Implement the following core functions
# - map
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply fn to every element in a list

    Args:
        fn (Callable[[float], float]): A Callable object

    Returns:
        function[[Iterable[float]], Iterable[float]]: a function which applies fn to every element in a list
    """

    # Closure
    def process(ls: Iterable[float]) -> Iterable[float]:
        return [fn(e) for e in ls]

    return process


# - zipWith
def zipWith(
    a: Iterable[float], b: Iterable[float], fn: Callable[[float, float], float]
) -> Iterable[float]:
    """Combines elements from two iterables a and b using a given function

    Args:
        a (Iterable[float]): iterable a
        b (Iterable[float]): iterable b
        fn (Callable[[float, float], float]): a callable which combines elements from two iterables

    Returns:
        Iterable[float]: combination of two iterables after applying fn
    """
    return [fn(e1, e2) for (e1, e2) in zip(a, b)]


# - reduce
def reduce(
    reduce_fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Combine every element in a list using reduce_fn

    Args:
        reduce_fn (Callable[[float, float], float]): A Callable object
        start (float): Initial value for reduction

    Returns:
        function[[Iterable[float]], float]: a function which combines every element in a list using reduce_fn
    """

    # Closure
    def process(ls: Iterable[float]) -> float:
        ans = start
        for e in ls:
            ans = reduce_fn(ans, e)
        return ans

    return process


#
# Use these to implement
# - negList : negate a list
def negList(ls: List[float]) -> Iterable[float]:
    """Negate all elements in a list

    Args:
        ls (List[float]): a list of floats

    Returns:
        List[float]: negated list
    """
    return map(neg)(ls)


# - addLists : add two lists together
def addLists(ls1: List[float], ls2: List[float]) -> Iterable[float]:
    """Computes element-wise sum of two lists

    Args:
        ls1 (List[float]): list1
        ls2 (List[float]): list2

    Returns:
        List[float]: element-wise sum of two lists
    """
    return zipWith(ls1, ls2, add)


# - sum: sum lists
def sum(ls: List[float]) -> float:
    """Returns the sum of all elements in a list

    Args:
        ls (List[float]): a list

    Returns:
        float: sum of all elements in a list
    """
    return reduce(add, 0)(ls)


# - prod: take the product of lists
def prod(ls: List[float]) -> float:
    """Returns the product of all elements in a list

    Args:
        ls (List[float]): a list

    Returns:
        float: product of all elements in a list
    """
    return reduce(mul, 1)(ls)
