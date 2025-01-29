from basicAlgorithms import Point
from scipy.optimize import linprog



class LinearConstraint:
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c
    
    @property
    def variables(self) -> tuple:
        return self.a, self.b, self.c

    def __repr__(self):
        return f"{self.a} * x + {self.b} * y <=  {self.c}"

# Seidel's algorithm for 2D linear programming
# objective: (a, b) coefficients of the objective function ax + by
# constraints: List of (a, b, c) where ax + by <= c
def seidel(objective: tuple[int, int], constraints: list[LinearConstraint]) -> tuple:
    # Check if a point satisfies all constraints
    def satisfies_constraints(point: Point, constraints: list[LinearConstraint])-> bool:
        for constraint_item in constraints:
            a, b, c = constraint_item.variables
            if a * point.x + b * point.y  > c:
                return False
        return True

    # Find the intersection of two lines
    def line_intersection(cos1:LinearConstraint, cos2:LinearConstraint) -> Point:
        a1, b1, c1 = cos1.variables
        a2, b2, c2 = cos2.variables
        det = a1 * b2 - b1 * a2
        if abs(det) == 0 :  # Use a small threshold to account for floating point errors
            return None
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        return Point(x, y)
    
    def find_optimal_point(feasible_points: list[Point], objective: tuple[int, int]):
        best_point = None
        best_value = float("inf")
        # Find the optimal point among the feasible points
        for point in feasible_points:
            value = objective[0] * point.x + objective[1] * point.y
            if value < best_value:
                best_point = point
                best_value = value

        return (best_point, best_value)
    
    
    def custom_linprog(objective, constraints):
        feasible_points = []
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                intersection_point = line_intersection(constraints[i], constraints[j])
                if intersection_point and satisfies_constraints(intersection_point, constraints):
                    feasible_points.append(intersection_point)

        if not feasible_points:
            raise ValueError("No feasible points found")

        return find_optimal_point(feasible_points, objective)


    # Start with the first 3 constraints
    current_constraints = constraints[:3]
    best_point, best_value = custom_linprog(objective, current_constraints)

    # Add one constraint at a time
    for i in range(3, len(constraints)):
        new_constraint = constraints[i]

        # Check current feasible points against the new constraint
        if satisfies_constraints(best_point, [new_constraint]):
            continue

        # Add the new contraint and reapeat
        current_constraints.append(new_constraint)
        best_point, best_value = custom_linprog(objective, current_constraints)

    return best_point, best_value

# Seidel's algorithm for 2D Linear Programming, leveraging linprog for optimality checks.
# - objective: (c1, c2) coefficients of objective function (Minimize c1*x + c2*y)
# - constraints: List of (a, b, c) where ax + by <= c
def seidel_with_linprog(objective: tuple[float, float], constraints: list[tuple[float, float, float]]):

    def satisfies_constraints(point, constraints):
        x, y = point
        for constraints_item in constraints:
            a, b, c = constraints_item.variables
            if a * x + b * y > c:  # Ensures point is within feasible region
                return False
        return True

    # Start with 3 constraints
    current_constraints = constraints[:3]

    A = []
    B = []
    # Use linprog to get an initial feasible point
    for constraints_item in current_constraints:
        a, b, c = constraints_item.variables
        A.append([a, b])
        B.append(c)

    result = linprog(c=objective, A_ub=A, b_ub=B, method='highs')
    
    if not result.success:
        raise ValueError("No feasible solution found in initialization")

    best_point = (result.x[0], result.x[1])
    best_value = result.fun

    # Step 2: Add remaining constraints one by one
    for i in range(3, len(constraints)):
        new_constraint = constraints[i]

        # Check if the current best point still satisfies the new constraint
        if satisfies_constraints(best_point, [new_constraint]):
            continue  # No need to recompute

        # Compute new optimal point using linprog
        a, b, c = new_constraint.variables
        A.append([a, b])
        B.append(c)

        # Add new constraint to active set
        current_constraints.append(new_constraint)

        result = linprog(c=objective, A_ub=A, b_ub=B, method='highs')
        
        if not result.success:
            raise ValueError("No feasible solution found after adding constraints")

        best_point = (result.x[0], result.x[1])
        best_value = result.fun


    return best_point, best_value

# Solves a 2D Linear Programming problem using SciPy's linprog.
# :param objective: Coefficients (c1, c2) of the objective function to minimize c1*x + c2*y.
# :param constraints: List of constraints in the form (a, b, c) for ax + by <= c.
# :return: Optimal point (x, y) and the optimal value.
def solve_linear_programming(objective: tuple[float, float], constraints: list[tuple[float, float, float]]):

    # Convert constraints into SciPy format
    A = []
    B = []
    
    for constraint_item in constraints:
        a, b, c = constraint_item.variables
        A.append([a, b])
        B.append(c)
    
    # Solve the problem (minimize c1*x + c2*y)
    result = linprog(c=objective, A_ub=A, b_ub=B, method='highs')

    if result.success:
        return (result.x[0], result.x[1]), result.fun
    raise ValueError("No feasible solution found")


if __name__ == "__main__":

    objective = (-3, 10)  # Minimize -3x1 + 10x2
    constraints = [
        LinearConstraint(-2, 1, 12),
        LinearConstraint(-1, 3, 3),
        LinearConstraint(6, 7, 18),
        LinearConstraint(3, -12, -8),
        LinearConstraint(2, -7, 35),
        LinearConstraint(-1, 8, 29),
        LinearConstraint(2, -6, 9),
        LinearConstraint(-1, 0, 0),
        LinearConstraint(0, -1, 0)
    ]

    # optimal_point, optimal_value = solve_linear_programming(objective, constraints)
    # optimal_point, optimal_value = seidel_with_linprog(objective, constraints)
    optimal_point, optimal_value = seidel(objective, constraints)

    # Print results
    print("Optimal Point:", optimal_point)
    print("Optimal Value:", optimal_value)