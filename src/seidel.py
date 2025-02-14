from scipy.optimize import linprog
import matplotlib.pyplot as plt
import numpy as np



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

# # Seidel's algorithm for 2D linear programming
# # objective: (a, b) coefficients of the objective function ax + by
# # constraints: List of (a, b, c) where ax + by <= c
# def seidel(objective: tuple[int, int], constraints: list[LinearConstraint]) -> tuple:
#     # Check if a point satisfies all constraints
#     def satisfies_constraints(point: Point, constraints: list[LinearConstraint])-> bool:
#         for constraint_item in constraints:
#             a, b, c = constraint_item.variables
#             if a * point.x + b * point.y  > c:
#                 return False
#         return True

#     # Find the intersection of two lines
#     def line_intersection(cos1:LinearConstraint, cos2:LinearConstraint) -> Point:
#         a1, b1, c1 = cos1.variables
#         a2, b2, c2 = cos2.variables
#         det = a1 * b2 - b1 * a2
#         if abs(det) == 0 :  # Use a small threshold to account for floating point errors
#             return None
#         x = (b2 * c1 - b1 * c2) / det
#         y = (a1 * c2 - a2 * c1) / det
#         return Point(x, y)
    
#     def find_optimal_point(feasible_points: list[Point], objective: tuple[int, int]):
#         best_point = None
#         best_value = float("inf")
#         # Find the optimal point among the feasible points
#         for point in feasible_points:
#             value = objective[0] * point.x + objective[1] * point.y
#             if value < best_value:
#                 best_point = point
#                 best_value = value

#         return (best_point, best_value)
    
    
#     def custom_linprog(objective, constraints):
#         feasible_points = []
#         for i in range(len(constraints)):
#             for j in range(i + 1, len(constraints)):
#                 intersection_point = line_intersection(constraints[i], constraints[j])
#                 if intersection_point and satisfies_constraints(intersection_point, constraints):
#                     feasible_points.append(intersection_point)

#         if not feasible_points:
#             raise ValueError("No feasible points found")

#         return find_optimal_point(feasible_points, objective)


#     # Start with the first 3 constraints
#     current_constraints = constraints[:3]
#     best_point, best_value = custom_linprog(objective, current_constraints)

#     # Add one constraint at a time
#     for i in range(3, len(constraints)):
#         new_constraint = constraints[i]

#         # Check current feasible points against the new constraint
#         if satisfies_constraints(best_point, [new_constraint]):
#             continue

#         # Add the new contraint and reapeat
#         current_constraints.append(new_constraint)
#         best_point, best_value = custom_linprog(objective, current_constraints)

#     return best_point, best_value

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

    # Add remaining constraints one by one 
    # check if the current best point still satisfies the new constraint
    for i in range(3, len(constraints)):
        new_constraint = constraints[i]

        
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

def plot_feasible_region(constraints, best_point=None):
    # Create a grid of points in the x and y space
    x_vals = np.linspace(-10, 10, 400)
    y_vals = np.linspace(-10, 10, 400)
    
    # Create a mesh grid
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Initialize the feasible region mask (True means inside the feasible region)
    feasible_region = np.ones(X.shape, dtype=bool)
    
    # Loop through constraints and find the region that satisfies all inequalities
    for constraint in constraints:
        a, b, c = constraint.variables
        # Apply the constraint ax + by <= c
        feasible_region &= (a * X + b * Y <= c)
    
    # Plot the feasible region
    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, feasible_region, cmap='Greens', alpha=0.5)
    
    # Plot the constraint lines
    for constraint in constraints:
        a, b, c = constraint.variables
        # Create line y = (c - ax) / b (if b != 0)
        if b != 0:
            y_line = (c - a * x_vals) / b
            plt.plot(x_vals, y_line, label=f"{a}x + {b}y <= {c}")
        elif a != 0:  # In case b == 0, we have a vertical line
            plt.axvline(x=c / a, color='r', linestyle='--')
    
    # If the best point is provided, plot it
    if best_point:
        plt.plot(best_point[0], best_point[1], 'ro', label=f"Best Point: ({best_point[0]:.2f}, {best_point[1]:.2f})")
    
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Feasible Region and Constraints")
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.legend()
    plt.grid(True)
    plt.savefig("../images/feasible_region.png")


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

    optimal_point, optimal_value = seidel_with_linprog(objective, constraints)

    # Plot the feasible region and the best point
    plot_feasible_region(constraints, optimal_point)

    print(f"Optimal Point: ({optimal_point[0]:.3f}, {optimal_point[1]:.3f})")
    print(f"Optimal Value: {optimal_value:.3f}")