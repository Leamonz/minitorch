from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    val_1, val_2 = list(vals), list(vals)
    val_1[arg] += epsilon
    val_2[arg] -= epsilon
    return (f(*val_1) - f(*val_2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    res = []
    permanent_mark = []
    temporary_mark = []

    def visit(var: Variable):
        # process non-constant variables only
        if var.is_constant():
            return
        if var.unique_id in permanent_mark:
            return
        if var.unique_id in temporary_mark:
            raise RuntimeError("Not a DAG")
        temporary_mark.append(var.unique_id)
        for par in var.parents:
            visit(par)
        temporary_mark.remove(var.unique_id)
        permanent_mark.append(var.unique_id)
        res.insert(0, var)

    visit(variable)
    return res


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # process the right-most variable
    var_deriv = {variable.unique_id: deriv}
    # find the topological order sequence starting from the right-most variable
    top_seq = topological_sort(variable)
    while top_seq:
        var = top_seq.pop(0)
        if not var.is_leaf():
            back = var.chain_rule(var_deriv[var.unique_id])
            for v, d in back:
                var_deriv[v.unique_id] = (
                    d
                    if v.unique_id not in set(list(var_deriv.keys()))
                    else var_deriv[v.unique_id] + d
                )
        else:
            var.accumulate_derivative(var_deriv[var.unique_id])


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
