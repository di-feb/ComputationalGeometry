import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


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
    

def create_plot_convex_hull_3D(points: Points3D):
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
    plt.savefig("../images/convex_hull_3D.png")
    print("3D plot saved as convex_hull_3D.png")

if __name__ == "__main__":
    def read_points_from_file_3D(file_path: str):
        points = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y, z = map(float, line.split())
                points.append(Point3D(x, y, z))
        return Points3D(points)
    
    file_name = f"../points/points3D.txt"
    points = read_points_from_file_3D(file_name)
    create_plot_convex_hull_3D(points) 