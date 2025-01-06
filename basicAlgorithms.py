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

# Example usage:
if __name__ == "__main__":
    def read_points_from_file(file_path: str):
        points = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(float, line.split())
                points.append(Point(x, y))
        return Points(points)

    # Replace 'points.txt' with the actual file path
    points = read_points_from_file('points_2.txt')
    convex_hull = graham_scan(points)
    print(f"Convex Hull: {convex_hull}")
    plot_convex_hull(points, convex_hull) 
