import matplotlib.pyplot as plt
import argparse
import math
import time
from typing import List


class Point:
    def __init__(self, x:float, y:float):
        self._x = x
        self._y = y

    # Getter for x
    @property
    def x(self) -> float:
        return self._x

    # Setter for x
    @x.setter
    def x(self, value: float):
        self._x = value

    # Getter for y
    @property
    def y(self) -> float:
        return self._y

    # Setter for y
    @y.setter
    def y(self, value: float):
        self._y = value

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

# Return the orientation of the triplet (p1, p2, p3).
# > 0 if counter-clockwise
# = 0 if collinear
# < 0 if clockwise
def orientation(p1: Point, p2: Point, p3: Point) -> float:
    return ((p2.x * p3.y) - (p2.y * p3.x)) - ((p1.x * p3.y) - (p1.y * p3.x)) + ((p1.x * p2.y) - (p1.y * p2.x))

# Returns the perpendicular distance of point p from the line joining points p1 and p2.
def line_distance(p1: Point, p2: Point, p: Point) -> float:
    return abs((p2.y - p1.y) * p.x - (p2.x - p1.x) * p.y + p2.x * p1.y - p2.y * p1.x) / (((p2.y - p1.y) ** 2 + (p2.x - p1.x) ** 2) ** 0.5)

def lexicographical_sort(points: List[Point]):
    points.sort(key=lambda p: (p.x, p.y))

# Visualizes the set of points and the convex hull.
# :param points: Points object containing all points.
# :param hull: Points object containing points on the convex hull.
def plot_convex_hull(points, hull):
    # Extract x and y coordinates from points
    x_points = [p.x for p in points]
    y_points = [p.y for p in points]

    # Extract x and y coordinates from convex hull
    hull_x = [p.x for p in hull]
    hull_y = [p.y for p in hull]

    # Close the convex hull loop by appending the first point at the end
    hull_x.append(hull[0].x)
    hull_y.append(hull[0].y)

    # Plot all points
    plt.scatter(x_points, y_points, label="Points", color="blue")

    # Plot the convex hull
    plt.plot(hull_x, hull_y, label="Convex Hull", color="red")

    # Add legend and show the plot
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Convex Hull")
    plt.savefig("convex_hull.png")
    print("Plot saved as convex_hull.png")

def graham_scan(points:List[Point]) -> List[Point]:    
    def construct_half(points_list):
        # Helper function to construct one half of the hull.
        hull = []
        for p in points_list:
            while len(hull) >= 2 and orientation(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        return hull


    lexicographical_sort(points)

     # Build lower hull and upper hull
    lower_hull = construct_half(points)
    upper_hull = construct_half(reversed(points))

    # Remove the last point of each half to avoid duplication and combine
    return (lower_hull[:-1] + upper_hull[:-1])

def gift_wrapping(points: List[Point]) -> List[Point]:
    if len(points) < 3:
        raise ValueError("At least 3 points are required to compute a convex hull.")

    hull = []
    start = min(points, key=lambda p: (p.x, p.y))  # Leftmost point
    current = start

    while True:
        hull.append(current)
        # Select the next point in the hull
        next_point = points[0]  # Initialize with an arbitrary point

        for candidate in points:
            if candidate == current:
                continue
            # Check orientation: counter-clockwise turn means the candidate is more "left"
            if (next_point == current or orientation(current, next_point, candidate) < 0):
                next_point = candidate

        current = next_point

        # Break if we have wrapped around to the start
        if current == start:
            break

    return hull

def divide_and_conquer(points: List[Point]) -> List[Point]:
    def merge_hulls(left_hull, right_hull):

        left_hull_len = len(left_hull)
        right_hull_len = len(right_hull)
        
        left_idx = max(range(left_hull_len), key=lambda i: left_hull[i].x)
        right_idx = min(range(right_hull_len), key=lambda i: right_hull[i].x)

        # Find the upper tangent
        i, j = left_idx, right_idx
        while True:
            changed = False
            if orientation(left_hull[i], left_hull[(i + 1) % left_hull_len], right_hull[j]) < 0:
                i = (i + 1) % left_hull_len
                changed = True
            if orientation(right_hull[j], right_hull[ (j - 1) % right_hull_len], left_hull[i]) > 0:
                j = (j - 1) % right_hull_len
                changed = True
            if not changed:
                break
        upper_tangent = (i, j)

        # Find the lower tangent
        i, j = left_idx, right_idx
        while True:
            changed = False
            if orientation(right_hull[j], right_hull[(j + 1) % right_hull_len], left_hull[i]) < 0:
                j = (j + 1) % right_hull_len
                changed = True
            if orientation(left_hull[i], left_hull[(i - 1) % left_hull_len], right_hull[j]) > 0:
                i = (i - 1) % left_hull_len
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
            i = (i + 1) % left_hull_len

        # Add points from right hull
        i = lower_tangent[1]
        while True:
            merged_hull.append(right_hull[i])
            if i == upper_tangent[1]:
                break
            i = (i + 1) % right_hull_len

        return merged_hull

    
    def divide(points_list):
        # Handle base case: 5 or fewer points
        if len(points_list) <= 5:
            return graham_scan(points_list)

        # Divide points into two halves
        mid = len(points_list) // 2
        left_hull = divide(points_list[:mid])
        right_hull = divide(points_list[mid:])

        # Merge the two hulls
        return merge_hulls(left_hull, right_hull)
    
    # Sort points by x-coordinate
    lexicographical_sort(points)
    return divide(points)

def sort_points_by_angle(points: List[Point]) -> List[Point]:
    def angle(p: Point) -> float:
        return math.atan2(p.y - centroid.y, p.x - centroid.x)
    hull_len = len(points)
    avg_x = sum(p.x for p in points) / hull_len
    avg_y = sum(p.y for p in points) / hull_len
    centroid = Point(avg_x, avg_y)
    return sorted(points, key=angle)

def quickhull(points: List[Point]) -> List[Point]:
    # Find the points with the minimum and maximum x-coordinates
    min_point = min(points, key=lambda p: p.x)
    max_point = max(points, key=lambda p: p.x)

    hull = []
    # Add the line endpoints to the hull
    hull.append(min_point)
    hull.append(max_point)

    # Partition the points into two sets: left of line and right of line
    left_set = [p for p in points if orientation(min_point, max_point, p) > 0]
    right_set = [p for p in points if orientation(min_point, max_point, p) < 0]

    def find_hull(p1: Point, p2: Point, points: List[Point], hull: List[Point]):
        if not points:
            return

        # Find the point farthest from the line (p1, p2),
        # and add it to the hull
        farthest_point = max(points, key=lambda p: line_distance(p1, p2, p))
        hull.append(farthest_point)

        # Partition the points into two subsets
        left_of_line1 = [p for p in points if orientation(p1, farthest_point, p) > 0]
        left_of_line2 = [p for p in points if orientation(farthest_point, p2, p) > 0]

        # Recur for the two subsets
        find_hull(p1, farthest_point, left_of_line1, hull)
        find_hull(farthest_point, p2, left_of_line2, hull)

    # Recursively find hull points
    find_hull(min_point, max_point, left_set, hull)
    find_hull(max_point, min_point, right_set, hull)

    # Sort the points in counter-clockwise order based on their angle to the centroid
    sorted_hull = sort_points_by_angle(hull)

    return sorted_hull

def find_convex_hull(algorithm, points: List[Point]):
    start_time = time.time()
    result = algorithm(points)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time



if __name__ == "__main__":
    def read_points_from_file(file_path: str):
        points = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(float, line.split())
                points.append(Point(x, y))
        return points
    
    parser = argparse.ArgumentParser(description="Run convex hull algorithms.")
    parser.add_argument(
        "algorithm",
        choices=["graham", "giftwrapping", "divideandconquer", "quickhull"],
        help="The convex hull algorithm to run (graham, giftwrapping, divideandconquer, quickhull).",
    )
    parser.add_argument(
        "num_points",
        type=int,
        choices=[50, 100, 500, 1000],
        help="The number of points to generate (50, 100, 500, 1000).",
    )
    args = parser.parse_args()

    algorithms = {
        "graham": graham_scan,
        "giftwrapping": gift_wrapping,
        "divideandconquer": divide_and_conquer,
        "quickhull": quickhull,
    }

    # Run the selected algorithm
    selected_algorithm = algorithms[args.algorithm]
    file_name = f"../points/points_{args.num_points}.txt"
    points = read_points_from_file(file_name)
    hull, time_taken = find_convex_hull(selected_algorithm, points)

    # Output results
    print(f"Algorithm: {args.algorithm}")
    print(f"Number of points: {args.num_points}")
    print(f"Execution time: {time_taken:.6f} seconds")
    plot_convex_hull(points, hull) 
   

    
