# Computational Geometry

This repository contains implementations of various computational geometry algorithms, including convex hull algorithms, linear programming, Voronoi diagrams, and k-d trees for range searching.


## **Exercise 1: Convex Hull Algorithms (2D & 3D)**

In this exercise, we implemented four different algorithms for computing the **convex hull** of a set of 2D points:

- **Graham's scan**
- **Gift Wrapping (Jarvis March)**
- **Divide and Conquer**
- **QuickHull**

These implementations can be found in `basicAlgorithms.py`.

### **How to Run:**

You can execute the script with:

```sh
python3 ./basicAlgorithms.py <algorithm> <num_points>
```

- `<algorithm>` can be `graham`, `giftwrapping`, `divideandconquer`, or `quickhull`.
- `<num_points>` can be `50`, `100`, `500`, or `1000`.

### **Example:**

```sh
python3 ./basicAlgorithms.py quickhull 100
```

This will compute the convex hull using QuickHull on 100 points and save the result as `convex_hull.png`.

### **Performance Comparison:**

| Algorithm        | 50 Points | 100 Points | 500 Points | 1000 Points |
|------------------|-----------|------------|------------|-------------|
| Graham's Scan    | 0.000142       | 0.000257        | 0.001304        | 0.002945        |
| Gift Wrapping    | 0.000426       | 0.000535        | 0.004335        | 0.010895        |
| Divide & Conquer | 0.000207       | 0.000445        | 0.002025        | 0.003687        | 
| QuickHull        | 0.000290       | 0.000425        | 0.002171        | 0.003499        | 

- **Graham's Scan:** Runs in **O(n log n)** time due to sorting.
- **Gift Wrapping:** Runs in **O(nh)** time, where *h* is the number of hull points (slower for large sets).
- **Divide & Conquer:** Runs in **O(n log n)** time.
- **QuickHull:** Runs in **O(n log n)** on average, but worst-case **O(n²)**.

As you will se all the algorithms returns the same convex hull.  

### **3D Convex Hull**

We also implemented a **3D convex hull** using `scipy.spatial.ConvexHull`.  
The underlying algorithm used is **QuickHull** for 3D, which has an average time complexity of **O(n log n)** but can degrade to **O(n²)** in the worst case.

#### **How to Run:**

```sh
python3 convexHull3D.py
```

This will compute the **3D convex hull** of 80 random points and save it as `convex_hull_3D.png`.

## **Exercise 2: Seidel's Algorithm for Linear Programming**

We implemented **Seidel’s Algorithm** to solve linear programming problems efficiently in low dimensions.

Additionally, we used `scipy.optimize.linprog` to compare with a standard LP solver.

#### **How to Run:**

```sh
python3 seidels.py
```

This will generate an image of the **critical area** of the solution space, saved as `critical_area.png`.

### **About linprog from SciPy:**

- Uses **interior-point methods** or **simplex methods**.
- The method automatically chooses the best approach based on the problem structure.
- Complexity depends on the method chosen:
  - Simplex: **Worst-case exponential, but works well in practice**.
  - Interior-Point: **Polynomial time** (approximately O(n³) for dense problems).


## **Exercise 3: Voronoi Diagram & Delaunay Triangulation**

We explored the relationship between **Voronoi Diagrams** and **Delaunay Triangulations** using `scipy.spatial`.
- A **Voronoi diagram** partitions the plane into regions, where each region corresponds to the area closest to a given point.
- A **Delaunay triangulation** is a triangulation of the points such that no point is inside the circumcircle of any triangle.

#### **How to Run:**

```sh
python3 voronoi-delaunay.py <num_points>
```

where `<num_points>` is the number of random points to generate.

This script will generate both:

- **Voronoi Diagram**
- **Delaunay Triangulation**

and save them in `voronoi_delaunay.png`.

### **Functions Used:**

- `Voronoi()` – Constructs the Voronoi diagram.
- `Delaunay()` – Constructs the Delaunay triangulation.
- `voronoi_plot_2d()` – Plots the Voronoi diagram.
- `delaunay_plot_2d()` – Plots the Delaunay triangulation.

### **Complexity Analysis of Voronoi Diagram and Delaunay Triangulation**

### 1. Voronoi Diagram Complexity
- **Time Complexity**: The best-known algorithms for computing the Voronoi diagram has **`O(n log n)`** complexity one of them is divide-and-conquer.
  - Other methods, such as divide-and-conquer, also achieve **`O(n log n)`** complexity.
- **Space Complexity**:
  - Stores approximately **`O(n)`** edges and **`O(n)`** regions, so the space complexity is **`O(n)`**.

### 2. Delaunay Triangulation Complexity
- **Time Complexity**: The most efficient Delaunay triangulation algorithms run in **`O(n log n)`** time.
  - In the worst case (for special degenerate cases), some algorithms may reach **`O(n^2)`**, but this is rare in practical scenarios.
- **Space Complexity**:
  - A Delaunay triangulation consists of **`O(n)`** triangles and edges, leading to **`O(n)`** space complexity.

### Effect of Increasing Number of Points \( n \)
- **For small \( n \)**, both structures are computed efficiently.
- **As \( n \) grows**, both the Voronoi diagram and Delaunay triangulation scale at **`O(n log n)`**, meaning:
  - **Doubling \( n \) only increases runtime slightly more than double**, making it feasible for large datasets.
  - **Memory usage increases linearly** with \( n \).


### Final Complexity Summary

| Algorithm              | Time Complexity | Space Complexity |
|------------------------|-----------------|------------------|
| Voronoi Diagram        | `O(n log n)`    | `O(n)`           |
| Delaunay Triangulation | `O(n log n)`    | `O(n)`           |

Since both algorithms have similar complexity, their performance depends more on implementation details and dataset properties.

## **Exercise 4: KD-Tree & Range Searching**

We implemented a **kd-tree** to perform efficient **range searching**.

### **How to Run:**

```sh
python3 range_search.py <seed> <xmin> <xmax> <ymin> <ymax>
```

- Seed is used for generating pseudo-random points.  
  So if you want your random points to be the same you **NEED** to keep the same seed all the time.

Example:

```sh
python3 range_search.py 42 0.2 0.8 0.3 0.7
```

1. Generate **150 random points**.
2. Build a **kd-tree**.
3. Perform a **range search** within the given rectangle.
4. Save the output in `range_search.png`.

### **KD-Tree Complexity:**

- **Construction:** **O(n log n)** (there is sorting at each level).
- **Range Search:** **O(√n) to O(n)**, depending on query size.

---

## **Dependencies**

Make sure you have the following Python libraries installed in order to be able to run and play with this project:

```sh
pip install numpy matplotlib scipy argparse
```

---

## **Conclusion**

This project covers various fundamental computational geometry problems, from convex hulls to range searching.  
We implemented both classical algorithms and leveraged existing scientific libraries to explore their efficiency and performance in practice.

