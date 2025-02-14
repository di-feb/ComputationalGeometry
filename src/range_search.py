import argparse
import numpy as np
import matplotlib.pyplot as plt

from kd_tree import KDNode, build_kdtree, plot_kdtree

def range_search(node: KDNode | None, xmin: float, xmax: float, ymin: float, ymax: float, found: list) -> None:
    if node is None:
        return
    
    x, y = node.point

    # Check if the point is inside our teritory
    if xmin <= x <= xmax and ymin <= y <= ymax:
        found.append(node.point)

    # Check the axes to choose in what branch you will go
    if node.axis == 0:
        if xmin <= x:
            range_search(node.left, xmin, xmax, ymin, ymax, found)
        if xmax >= x:
            range_search(node.right, xmin, xmax, ymin, ymax, found)
    else:
        if ymin <= y:
            range_search(node.left, xmin, xmax, ymin, ymax, found)
        if ymax >= y:
            range_search(node.right, xmin, xmax, ymin, ymax, found)

if __name__ == "__main__":  
    
    parser = argparse.ArgumentParser(description="Build a KD-Tree and perform range search.")
    
    parser.add_argument("seed", type=int, help="Random seed for generating points")
    parser.add_argument("xmin", type=float, help="Minimum x-coordinate for search region")
    parser.add_argument("xmax", type=float, help="Maximum x-coordinate for search region")
    parser.add_argument("ymin", type=float, help="Minimum y-coordinate for search region")
    parser.add_argument("ymax", type=float, help="Maximum y-coordinate for search region")

    args = parser.parse_args()

    np.random.seed(args.seed)
    points = np.random.rand(150, 2)

    # Build KD-Tree
    kd_tree = build_kdtree(points.tolist())

    print(f"KD-Tree built with {len(points)} points.")
    print(f"Search Region: xmin={args.xmin}, xmax={args.xmax}, ymin={args.ymin}, ymax={args.ymax}")

    kd_tree = build_kdtree(points.tolist())


    found_points = []
    range_search(kd_tree, args.xmin, args.xmax, args.ymin, args.ymax, found_points)

    print(f"Points inside teritory ({args.xmin} ≤ x ≤ {args.xmax}, {args.ymin} ≤ y ≤ {args.ymax}):")
    for point in found_points:
        print(point)

    plt.figure(figsize=(8, 6))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plot_kdtree(kd_tree, 0, 1, 0, 1)

    # Plot teritory
    plt.plot([args.xmin, args.xmin, args.xmax, args.xmax, args.xmin], [args.ymin, args.ymax, args.ymax, args.ymin, args.ymin], 'g--', lw=2)

    # Plot points in the teritory
    found_points = np.array(found_points)
    if found_points.size > 0:
        plt.scatter(found_points[:, 0], found_points[:, 1], c='blue', s=80, label="Found Points")

    plt.title("Kd-tree and range search")
    plt.legend()
    plt.savefig("range_search.png")