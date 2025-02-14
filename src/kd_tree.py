import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


class KDNode:
    def __init__(self, point: Tuple[float, float], left: Optional["KDNode"] = None, right: Optional["KDNode"] = None, axis: int = 0):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis

def build_kdtree(points: List[Tuple[float, float]], depth: int = 0) -> Optional[KDNode]:
    if not points:
        return None
    
    axis = depth % 2  # 0 for x-axis, 1 for y-axis

    # Sort and find median
    points.sort(key=lambda x: x[axis])
    median_idx = len(points) // 2

    # Left subset includes points ≤ median
    left_set = points[:median_idx]  
    right_set = points[median_idx + 1:] 

    left_node = build_kdtree(left_set, depth + 1)
    right_node = build_kdtree(right_set, depth + 1)

    return KDNode(point=points[median_idx], left=left_node, right=right_node, axis=axis)

# This function visualizes the construction of a kd-tree
# by recursively plotting the splitting lines and points.
def plot_kdtree(node: Optional[KDNode], min_x: float, max_x: float, min_y: float, max_y: float, depth: int = 0) -> None:
    if node is None:
        return

    x, y = node.point
    axis = node.axis # 0 = vertical split, 1 = horizontal split

    if axis == 0:
        # Split vertically x'x
        plt.plot([x, x], [min_y, max_y], 'r--', lw=1)
        plot_kdtree(node.left, min_x, x, min_y, max_y, depth + 1)
        plot_kdtree(node.right, x, max_x, min_y, max_y, depth + 1)
    else:
        # Split horizontally y'y
        plt.plot([min_x, max_x], [y, y], 'b--', lw=1)
        plot_kdtree(node.left, min_x, max_x, min_y, y, depth + 1)
        plot_kdtree(node.right, min_x, max_x, y, max_y, depth + 1)

    plt.scatter(x, y, c='black', s=20, zorder=3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a kd-tree from random points.")
    parser.add_argument("num_points", type=int, help="Number of random points to generate")
    args = parser.parse_args()

    np.random.seed(42)
    points = np.random.rand(args.num_points, 2)

    kd_tree = build_kdtree(points.tolist())

    plt.figure(figsize=(8, 6))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plot_kdtree(kd_tree, 0, 1, 0, 1)
    plt.title("Kd-tree")
    plt.savefig("Kd-tree.png")

