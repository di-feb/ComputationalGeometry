import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from typing import List, Tuple


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
    
    
class Points:
    def __init__(self, points=None):
        self.points = points if points else []

    def __repr__(self):
        return f"Points({self.points})"

    def lexicographical_sort(self):
        self.points.sort(key=lambda p: (p.x, p.y))
    
class Point3D:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        self._z = value

    def __repr__(self):
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"    

class Points3D:
    def __init__(self, points=None):
        self.points = points if points else []

    # Converts the points into a numpy array suitable for ConvexHull
    def to_numpy(self):
        return np.array([(point.x, point.y, point.z) for point in self.points])
    

# Return the orientation of the triplet (p1, p2, p3).
# > 0 if counter-clockwise
# = 0 if collinear
# < 0 if clockwise
def orientation(p1: Point, p2: Point, p3: Point) -> float:
    return ((p2.x * p3.y) - (p2.y * p3.x)) - ((p1.x * p3.y) - (p1.y * p3.x)) + ((p1.x * p2.y) - (p1.y * p2.x))

# Visualizes the set of points and the convex hull.
# :param points: Points object containing all points.
# :param hull: Points object containing points on the convex hull.
def plot_convex_hull(points, hull):
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
    plt.title("Convex Hull")
    plt.savefig("convex_hull.png")
    print("Plot saved as convex_hull.png")

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
    def merge_hulls(left_hull, right_hull):

        left_hull_len = len(left_hull)
        right_hull_len = len(right_hull)
        
        left_idx = max(range(left_hull_len), key=lambda i: left_hull[i].x)
        right_idx = min(range(right_hull_len), key=lambda i: right_hull[i].x)

        # Find the upper tangent
        i, j = left_idx, right_idx
        while True:
            changed = False
            if(orientation(right_hull[j], left_hull[i], left_hull[(i + 1) % left_hull_len]) > 0):
                i = (i + 1) % left_hull_len
                changed = True
            if(orientation(left_hull[i], right_hull[j], right_hull[ (j - 1) % right_hull_len]) < 0):
                j = (j - 1) % right_hull_len
                changed = True
            if not changed:
                break
        upper_tangent = (i, j)

        # Find the lower tangent
        i, j = left_idx, right_idx
        while True:
            changed = False
            if(orientation(left_hull[i], right_hull[j], right_hull[(j + 1) % right_hull_len] ) > 0 ):
                j = (j + 1) % right_hull_len
                changed = True
            if(orientation(right_hull[j], left_hull[i], left_hull[(i - 1) % left_hull_len]) < 0):
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
        # Handle base case: 3 or fewer points
        if len(points_list) <= 3:
            return graham_scan(Points(points_list)).points

        # Divide points into two halves
        mid = len(points_list) // 2
        left_hull = divide(points_list[:mid])
        right_hull = divide(points_list[mid:])

        # Merge the two hulls
        return merge_hulls(left_hull, right_hull)
    
    # Sort points by x-coordinate
    points.lexicographical_sort()
    return Points(divide(points.points))

     
def plot_convex_hull_3D(points: Points3D):
    # Convert Points3D object to NumPy array for plotting
    points_array = points.to_numpy()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], color='blue', label='Points')

    # Plot the convex hull facets
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
    # points3D = read_points_from_file_3D('points3D.txt')
    # convex_hull = graham_scan(points)
    # convex_hull = gift_wrapping(points)
    convex_hull = divide_and_conquer(points)
    # convex_hull = quickhull(points)

    plot_convex_hull(points, convex_hull) 
    # plot_convex_hull_3D(points3D) 

    
