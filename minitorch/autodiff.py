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
    val_temp=list(vals)
    ans_list=[]
    for i in range(arg):
        val_plus=list(val_temp)
        val_minus=list(val_temp)
        val_plus[i]+=epsilon/2
        val_minus[i]-=epsilon/2
        ans_list.append((f(val_plus)-f(val_minus))/epsilon)
    return ans_list
    # TODO: Implement for Task 1.1.
    #raise NotImplementedError("Need to implement for Task 1.1")


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
    PermanentMarked=[]
    TemporaryMarked=[]
    result=[]
    def visit(x):
        if x.is_constant():
            return
        if x.unique_id in PermanentMarked:
            return 
        elif x.unique_id in TemporaryMarked:
            raise(RuntimeErroe("not dag"))
        TemporaryMarked.append(n.unique_id)
        if x.is_leaf():
            pass
        else:
            for input in x.history.inputs:
                visit (input)
        TemporaryMarked.remove(x.unique_id)
        PermanentMarked.remove(x.unique_id)
        result.insert(0,x)
        
    visit(variable)
    return result
    #raise NotImplementedError("Need to implement for Task 1.4")


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
    order=topological_sort(variable)
    derivs={variable.unique_id:deriv}
    for node in order:
        d_output=derivs[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(d_output)
        else:
            for input,d in node.history.backprop_step(d_output):
                if input.unique_id not in derivs:
                    derivs[input.unique_id]=0.0
                derivs[input.unique_id]+=d
    return

    raise NotImplementedError("Need to implement for Task 1.4")


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
