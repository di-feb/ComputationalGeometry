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

    plt.savefig("voronoi_delaunay.png")

# So now beased on all of we have been tell give me a readme.md file
# The name of the course is computational geometry 
# In the first exercise we implemented the grahams scan the gift wrappiong the devide and conquer and the quickhull algorithms for 2d 
# those implementrations are inside the basicAlgorithms.py 
# you can run the algorithms if your right  python3 ./basicAlgorithms.py and then give the name of the algorithm and the number of points you can choose bettewn 50 100 500 1000 poiints . there are 5 files .txt that contains those points each one for every argument.
# an example is 
# python3 ./basicAlgorithms.py quickhull  100
# when you run this script the convex hull will be created and saved inside a png with the anme convex_hull.png
# I want you to create me a board that you will have the avg time that each algorithm took to run 
# and say a few things about it
# we also create a convexhull for 3d using the from scipy.spatial import ConvexHull library i want you to tell me what algorithm it runs behind the scenes and explain its complexity
# when you run this algorithm a 3d image of a convex hull of 80 points will be created and saved inside the  convex_hull_3D.png  
# ---------
# In the next exercise we created th seidels algorithm to solve and linear programming problem with some contraints we use it the linprog library to solve it i want you to tell me a few things about this function
# you can run it with python3 seidels.py when you run it a ong will be createed that it will show you the critical area c inside the critical_area.png
# -------
# In the next exercise we show what is the realtionship beetween voronoi diagram and delaunay triangle you can run it writing python3 voronoi-delaynay num_of_points which is the number of points. we used from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d, delaunay_plot_2d
# those function to build that. when you run it two diagrams of voronoi and delaunay will be created in voronoi_dealynay.png
# ------
# Last exercise was about create a kd tree and do a range search we just did so you know what to write about those.