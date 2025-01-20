import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    
class Points:
    def __init__(self, points=None):
        self.points = points if points else []

    def __repr__(self):
        return f"Points({self.points})"
    
    def determinant(self, point1:Point, point2:Point):
        return point1.x * point2.y - point2.x * point1.y
    
    def add_point(self, point: Point):
        self.points.append(point)

    def lexicographical_sort(self):
        self.points.sort(key=lambda p: (p.x, p.y))

    # Converts the points into a numpy array for the ConvexHull 3D function
    def to_numpy(self):
        return np.array([(point.x, point.y) for point in self.points])
class Point3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

class Points3D:
    def __init__(self, points=None):
        self.points = points if points else []

    def __repr__(self):
        return f"Points({self.points})"

    def to_numpy(self):
        # Converts the points into a numpy array suitable for ConvexHull
        return np.array([(point.x, point.y, point.z) for point in self.points])
    
def orientation(p1: Point, p2: Point, p3: Point) -> float:
    """
    Return the orientation of the triplet (p1, p2, p3).
    > 0 if counter-clockwise
    = 0 if collinear
    < 0 if clockwise
    """
    return (p2.y - p1.y) * (p3.x - p2.x) - (p2.x - p1.x) * (p3.y - p2.y)


def plot_convex_hull(points, hull):
    """
    Visualizes the set of points and the convex hull.
    :param points: Points object containing all points.
    :param hull: Points object containing points on the convex hull.
    """
    # Extract x and y coordinates from points
    x_points = [p.x for p in points.points]
    y_points = [p.y for p in points.points]

    # Extract x and y coordinates from convex hull
    hull_x = [p.x for p in hull.points]
    hull_y = [p.y for p in hull.points]

    # Close the convex hull loop by appending the first point at the end
    hull_x.append(hull.points[0].x)
    hull_y.append(hull.points[0].y)

    # Plot all points
    plt.scatter(x_points, y_points, label="Points", color="blue")

    # Plot the convex hull
    plt.plot(hull_x, hull_y, label="Convex Hull", color="red")

    # Add legend and show the plot
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Convex Hull Visualization")
    plt.savefig("convex_hull.png")
    print("Plot saved as convex_hull.png")



# Implements the Graham Scan algorithm to find the convex hull.
# :param points: Points object containing all points.
# :return: Points object containing points on the convex hull.
def graham_scan(points: Points) -> Points:    
    def construct_half(points_list):
        # Helper function to construct one half of the hull.
        hull = []
        for p in points_list:
            while len(hull) >= 2 and orientation(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        return hull


    points.lexicographical_sort()

     # Build lower hull and upper hull
    lower_hull = construct_half(points.points)
    upper_hull = construct_half(reversed(points.points))

    # Remove the last point of each half to avoid duplication and combine
    return Points(lower_hull[:-1] + upper_hull[:-1])

# Implements the Gift Wrapping (Jarvis March) algorithm to find the convex hull.
# :param points: Points object containing all points.
# :return: Points object containing points on the convex hull.
def gift_wrapping(points: Points) -> Points:
    if len(points.points) < 3:
        raise ValueError("At least 3 points are required to compute a convex hull.")

    hull = []
    start = min(points.points, key=lambda p: (p.x, p.y))  # Leftmost point
    current = start

    while True:
        hull.append(current)
        # Select the next point in the hull
        next_point = points.points[0]  # Initialize with an arbitrary point

        for candidate in points.points:
            if candidate == current:
                continue
            # Check orientation: counter-clockwise turn means the candidate is more "left"
            if (next_point == current or orientation(current, next_point, candidate) < 0):
                next_point = candidate

        current = next_point

        # Break if we have wrapped around to the start
        if current == start:
            break

    return Points(hull)

def divide_and_conquer(points: Points) -> Points:
    """
    Implements the Divide and Conquer algorithm to find the convex hull.
    :param points: Points object containing all points.
    :return: Points object containing points on the convex hull.
    """
    def merge_hulls(left_hull, right_hull):
        """
        Merge two convex hulls into a single convex hull.
        :param left_hull: List of Points in the left hull.
        :param right_hull: List of Points in the right hull.
        :return: Merged convex hull as a list of Points.
        """
        # Find the rightmost point in the left hull and leftmost point in the right hull
        left_idx = max(range(len(left_hull)), key=lambda i: left_hull[i].x)
        right_idx = min(range(len(right_hull)), key=lambda i: right_hull[i].x)

        # Find the upper tangent
        i, j = left_idx, right_idx
        while True:
            changed = False
            while orientation(left_hull[i], right_hull[j], right_hull[(j + 1) % len(right_hull)]) < 0:
                j = (j + 1) % len(right_hull)
                changed = True
            while orientation(right_hull[j], left_hull[i], left_hull[(i - 1) % len(left_hull)]) > 0:
                i = (i - 1) % len(left_hull)
                changed = True
            if not changed:
                break
        upper_tangent = (i, j)

        # Find the lower tangent
        i, j = left_idx, right_idx
        while True:
            changed = False
            while orientation(left_hull[i], right_hull[j], right_hull[(j - 1) % len(right_hull)]) > 0:
                j = (j - 1) % len(right_hull)
                changed = True
            while orientation(right_hull[j], left_hull[i], left_hull[(i + 1) % len(left_hull)]) < 0:
                i = (i + 1) % len(left_hull)
                changed = True
            if not changed:
                break
        lower_tangent = (i, j)

        # Combine points from both hulls between the tangents
        merged_hull = []

        # Add points from left hull
        i = upper_tangent[0]
        while True:
            merged_hull.append(left_hull[i])
            if i == lower_tangent[0]:
                break
            i = (i + 1) % len(left_hull)

        # Add points from right hull
        j = lower_tangent[1]
        while True:
            merged_hull.append(right_hull[j])
            if j == upper_tangent[1]:
                break
            j = (j + 1) % len(right_hull)

        return merged_hull

    def divide(points_list):
        """
        Recursive division of points into halves to compute convex hulls.
        """
        if len(points_list) <= 3:
            # Handle base case: 3 or fewer points
            points_list.sort(key=lambda p: (p.x, p.y))  # Sort to maintain order
            return graham_scan(Points(points_list)).points

        # Divide points into two halves
        mid = len(points_list) // 2
        left_hull = divide(points_list[:mid])
        right_hull = divide(points_list[mid:])

        # Merge the two hulls
        return merge_hulls(left_hull, right_hull)

    # Sort points by x-coordinate
    points.points.sort(key=lambda p: (p.x, p.y))
    return Points(divide(points.points))

def quickhull(points: Points) -> Points:
    """
    Implements the QuickHull algorithm to find the convex hull.
    :param points: Points object containing all points.
    :return: Points object containing points on the convex hull.
    """
    if len(points.points) < 3:
        raise ValueError("At least 3 points are required to compute a convex hull.")

    def find_furthest_point(points, p1, p2):
        """
        Find the point furthest from the line formed by p1 and p2.
        """
        max_distance = -1
        furthest_point = None

        for point in points:
            distance = abs((p2.y - p1.y) * point.x - (p2.x - p1.x) * point.y + p2.x * p1.y - p2.y * p1.x)
            distance /= ((p2.y - p1.y)**2 + (p2.x - p1.x)**2)**0.5

            if distance > max_distance:
                max_distance = distance
                furthest_point = point

        return furthest_point

    def points_on_side(points, p1, p2):
        """
        Filter points that are on the left side of the line formed by p1 and p2.
        """
        side_points = []
        for point in points:
            if orientation(p1, p2, point) > 0:
                side_points.append(point)
        return side_points

    def find_hull(points, p1, p2, hull):
        """
        Recursive function to find hull points on one side of the line formed by p1 and p2.
        """
        if not points:
            return

        # Find the furthest point from the line formed by p1 and p2
        furthest = find_furthest_point(points, p1, p2)

        hull.append(furthest)

        # Partition points into two sets: left of (p1, furthest) and (furthest, p2)
        left_of_p1_furthest = points_on_side(points, p1, furthest)
        left_of_furthest_p2 = points_on_side(points, furthest, p2)

        find_hull(left_of_p1_furthest, p1, furthest, hull)
        find_hull(left_of_furthest_p2, furthest, p2, hull)

    # Step 1: Find the leftmost and rightmost points (guaranteed to be on the hull)
    leftmost = min(points.points, key=lambda p: p.x)
    rightmost = max(points.points, key=lambda p: p.x)

    # Step 2: Divide points into two sets: above and below the line (leftmost, rightmost)
    above = points_on_side(points.points, leftmost, rightmost)
    below = points_on_side(points.points, rightmost, leftmost)

    # Step 3: Find hull points recursively
    hull = [leftmost, rightmost]
    find_hull(above, leftmost, rightmost, hull)
    find_hull(below, rightmost, leftmost, hull)

    # Step 4: Remove duplicates and return the sorted hull
    unique_hull = sorted(set(hull), key=lambda p: (p.x, p.y))
    return Points(unique_hull)

# Function to compute the convex hull using QuickHull (via scipy)
def quickhull_3D(points: Points):
    # Convert Points object to numpy array
    points_array = points.to_numpy()

    # Compute the convex hull using SciPy's ConvexHull function
    hull = ConvexHull(points_array)

    # The result contains the indices of the points forming the convex hull
    convex_hull_points = [points.points[i] for i in hull.vertices]
    
    return convex_hull_points

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_convex_hull_3D(points: Points, convex_hull):
    # Convert Points object to numpy array for plotting
    points_array = points.to_numpy()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], color='blue', label='Points')

    # Plot the convex hull facets
    hull_points_array = np.array([(p.x, p.y, p.z) for p in convex_hull])
    for simplex in ConvexHull(points_array).simplices:
        # Get the vertices of the simplex
        vertices = points_array[simplex]
        # Plot the simplex as a triangle
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', alpha=0.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Convex Hull')

    # Add a legend and save the plot
    plt.legend()
    plt.savefig("convex_hull_3D.png")
    print("3D plot saved as convex_hull_3D.png")

from typing import List, Tuple

class LinearConstraint:
    def __init__(self, a: float, b: float, c: float):
        """
        Represents a linear constraint of the form:
        a * x + b * y <= c
        """
        self.a = a
        self.b = b
        self.c = c

    def __repr__(self):
        return f"{self.a} * x + {self.b} * y <= {self.c}"

def intersection(line1: LinearConstraint, line2: LinearConstraint) -> Tuple[float, float]:
    """
    Finds the intersection point of two lines (ignoring constraints for now).
    Solves the system:
        line1.a * x + line1.b * y = line1.c
        line2.a * x + line2.b * y = line2.c
    """
    det = line1.a * line2.b - line2.a * line1.b
    if det == 0:
        raise ValueError("Lines are parallel or coincident")

    x = (line2.b * line1.c - line1.b * line2.c) / det
    y = (line1.a * line2.c - line2.a * line1.c) / det
    return (x, y)

def is_point_feasible(point: Tuple[float, float], constraints: List[LinearConstraint]) -> bool:
    """
    Checks if a given point satisfies all the constraints.
    """
    for constraint in constraints:
        if constraint.a * point[0] + constraint.b * point[1] > constraint.c:
            return False
    return True

def solve_linear_program(constraints: List[LinearConstraint], objective: Tuple[float, float]) -> Tuple[float, float]:
    """
    Solves the 2D linear programming problem incrementally:
    - constraints: List of LinearConstraint (a, b, c)
    - objective: Coefficients of the objective function to maximize (cx, cy)
    Returns:
        The optimal point (x, y).
    """
    if not constraints:
        raise ValueError("No constraints provided")

    # Step 1: Start with the first two constraints
    feasible_region = []
    for i, constraint1 in enumerate(constraints):
        for j, constraint2 in enumerate(constraints):
            if i < j:
                try:
                    inter = intersection(constraint1, constraint2)
                    if is_point_feasible(inter, constraints):
                        feasible_region.append(inter)
                except ValueError:
                    continue

    if not feasible_region:
        raise ValueError("No feasible region exists")

    # Step 2: Evaluate the objective function at all feasible points
    cx, cy = objective
    optimal_point = max(feasible_region, key=lambda point: cx * point[0] + cy * point[1])
    return optimal_point


# Example usage:
if __name__ == "__main__":
    def read_points_from_file(file_path: str):
        points = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(float, line.split())
                points.append(Point(x, y))
        return Points(points)
    
    def read_points_from_file_3D(file_path: str):
        points = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y, z = map(float, line.split())
                points.append(Point3D(x, y, z))
        return Points3D(points)

    # Replace 'points.txt' with the actual file path
    points = read_points_from_file('points_2.txt')
    points = read_points_from_file_3D('points3D.txt')
    # convex_hull = graham_scan(points)
    # convex_hull = gift_wrapping(points)
    # convex_hull = quickhull(points)
    convex_hull = quickhull_3D(points)

    # print(f"Convex Hull: {convex_hull}")
    # plot_convex_hull(points, convex_hull) 
    plot_convex_hull_3D(points, convex_hull) 

     # Solve the linear programming problem
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

    # Define the objective function
    objective = (3, -10)

    # Solve using the incremental algorithm
    optimal_point = solve_linear_program(constraints, objective)

    print(f"Optimal Point: {optimal_point}")

    # Plot the feasible region and solution
    def plot_feasible_region(constraints, optimal_point):
        x_vals = np.linspace(-5, 20, 500)
        y_vals = np.linspace(-5, 20, 500)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        plt.figure(figsize=(10, 8))
        plt.title("Feasible Region and Optimal Solution")
        
        for constraint in constraints:
            Z = constraint.a * X + constraint.b * Y - constraint.c
            plt.contour(X, Y, Z, levels=[0], colors='blue', linestyles='dotted')
        
        plt.scatter(optimal_point[0], optimal_point[1], color='red', label='Optimal Solution')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid()
        plt.savefig("linear_prog.png")
        print("linear prog solved in linear_prog.png")

    plot_feasible_region(constraints, optimal_point)

