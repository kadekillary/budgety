from typing import List, Tuple

from pandas import DataFrame
from pyomo.environ import *

from .types import Curves, SpendConstr


class OptimizeMax:
    def __init__(
        self,
        data: Curves,
        spend: SpendConstr,
        included_dimensions: List[str],
        budget: int,
    ):
        self.data: Curves = data
        self.spend: SpendConstr = spend
        self.included_dimensions: List[str] = included_dimensions
        self.model: ConcreteModel = ConcreteModel()
        self.budget: int = budget

    def spend_bounds(self, dimension: str) -> Tuple[float, float]:
        median_spend = getattr(self.data[dimension], "median_spend")
        spend_lower_bound = 1 + (self.spend[dimension][0] / 100)
        spend_upper_bound = 1 + (self.spend[dimension][1] / 100)
        return (spend_lower_bound * median_spend, spend_upper_bound * median_spend)

    def keep_dimensions(self) -> Curves:
        # Only include dimensions that have been passed in final
        # optimization
        return {k: v for k, v in self.data.items() if k in self.included_dimensions}

    def run(self) -> DataFrame:
        # @TODO: re-write to be more modular, less gross
        # might not need since we can filter out in app
        data = self.keep_dimensions()
        D = data.keys()
        model = self.model
        model.x = Var(
            D,
            domain=NonNegativeReals,
            initialize=lambda model, x: getattr(data[x], "median_spend"),
            bounds=lambda model, x: self.spend_bounds(x),
        )
        model.obj = Objective(
            expr=sum(
                getattr(data[d], "coef") * log(model.x[d])
                + getattr(data[d], "intercept")
                for d in D
            ),
            sense=maximize,
        )
        model.budget = Constraint(expr=self.budget == sum(model.x[d] for d in D))
        solver = SolverManagerFactory("neos")
        # @TODO: potentially expose alternative solver options
        solution = solver.solve(model, opt="couenne")
        # Make sure optimal solution was found
        print(solution.solver.termination_condition)
        if solution.solver.termination_condition == TerminationCondition.infeasible:
            raise Exception("Current constraints are infeasible!")
        elif solution.solver.termination_condition == TerminationCondition.optimal:
            results = DataFrame.from_dict(
                model.x.extract_values(), orient="index", columns=[str(model.x)]
            )
            return results.reset_index()
        else:
            print(solution.solver)
