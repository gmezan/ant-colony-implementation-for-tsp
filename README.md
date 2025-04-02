# Ant Colony Optimization for Weighted Traveling Salesman Problem (WTSP)

## 📌 Overview
This project implements the **Ant Colony Optimization (ACO)** algorithm to solve the **Weighted Traveling Salesman Problem (WTSP)**. The algorithm simulates the behavior of ants finding the shortest path in a graph, optimizing the route for a salesman who must visit all cities with different weighted distances.

## 🧠 Algorithm Explanation
- Ants explore the graph, moving between cities based on **pheromone levels** and **heuristic information**.
- Each ant builds a tour, considering edge weights and probabilistic choices.
- After completing a tour, ants deposit **pheromones** on visited edges.
- Over multiple iterations, the pheromone trails help find an optimized path.

## 🔧 Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/gmezan/ant-colony-implementation-for-tsp.git
    cd ant-colony-implementation-for-tsp
    ```

3. Create a python env and install requirements:
    ```bash
    pip install -r requirements.txt
    ```
    
4. Run the script with a sample dataset:
    ```bash
    python aco_for_tsp.py # takes default dataset

    python aco_for_tsp.py resources/instance2.csv # takes another dataset

    python aco_for_tsp.py custom.csv # custom dataset
    ```
5. Example taking `resources/instance1.csv`:
![alt text](https://raw.githubusercontent.com/gmezan/ant-colony-implementation-for-tsp/main/example/example.png)
