import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d, delaunay_plot_2d

def plot_diagram(plot_function, diagram, side, color, title) -> None:
    if diagram == voronoi:
        plot_function(diagram, ax=side, show_vertices=False)
    else:
        plot_function(diagram, ax=side)
    side.scatter(points[:,0], points[:,1], color=color, label='Points')
    side.set_title(title)
    side.legend()

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="Generate a kd-tree from random points.")
    parser.add_argument("num_points", type=int, help="Number of random points to generate")
    args = parser.parse_args()

    np.random.seed(42)  
    points = np.random.rand(args.num_points, 2) 

    # Compute Voronoi Diagram and Delaunay Triangulation
    voronoi = Voronoi(points)
    delaunay = Delaunay(points)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Voronoi Diagram
    plot_diagram(voronoi_plot_2d, voronoi, ax[0], 'red', 'Voronoi Diagram')

    # Plot Delaunay Triangulation
    plot_diagram(delaunay_plot_2d, delaunay, ax[1], 'cyan', 'Delaunay Triangulation')

    plt.savefig("../images/voronoi_delaunay.png")