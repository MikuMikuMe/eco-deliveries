Creating a comprehensive Python program for "eco-deliveries" involves several components, including data preparation, machine learning model training, and route optimization. Below is a simplified example detailing the steps you might take to implement such a platform. This example assumes you have historical delivery data and want to predict and optimize future routes to reduce emissions and costs.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import itertools
import networkx as nx

# Sample delivery data, usually this would be loaded from a database or a CSV file
# For this example, let's assume a simple structure
# Data includes delivery route name, distance, time taken, and fuel used
data = {
    'route_id': [1, 2, 3, 4, 5],
    'distance_km': [100, 150, 200, 250, 300],
    'time_hours': [2, 3, 4, 5, 6],
    'fuel_used_liters': [10, 12, 15, 18, 22]
}
df = pd.DataFrame(data)

# Separate features and target
X = df[['distance_km', 'time_hours']]
y = df['fuel_used_liters']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

def optimize_route(locations):
    """
    Finds the optimal route to minimize distance using a simple approach.
    Here locations are given by their distance matrix.

    :param locations: List of tuples containing distances between locations
    :return: Optimal route and total minimum distance
    """
    try:
        min_distance = float('inf')
        optimal_route = []

        for perm in itertools.permutations(locations):
            current_distance = sum(perm[i][i + 1] for i in range(len(perm) - 1))

            if current_distance < min_distance:
                min_distance = current_distance
                optimal_route = perm

        return optimal_route, min_distance
    except Exception as e:
        print(f"An error occurred while optimizing the route: {e}")
        return None, None

# Example distance matrix
locations = [(0, 1, 20), (1, 2, 30), (2, 0, 25)]  # (from, to, distance)

optimal_route, min_distance = optimize_route(locations)
print(f"Optimal Route: {optimal_route}, Min Distance: {min_distance}")

# Network Analysis for more complex route optimization
def create_graph(locations):
    """
    Creates a graph with edges as distances between 'locations'.
    
    :param locations: List of tuples as above
    :return: A NetworkX graph
    """
    G = nx.Graph()
    
    try:
        for from_loc, to_loc, distance in locations:
            G.add_edge(from_loc, to_loc, weight=distance)
    except Exception as e:
        print(f"An error occurred while creating a graph: {e}")
    
    return G

def shortest_path(G, start_node, end_node):
    """
    Finds the shortest path in a graph using Dijkstra's algorithm.
    
    :param G: The NetworkX graph
    :param start_node: Starting node
    :param end_node: End node
    :return: Shortest path and its distance
    """
    try:
        path = nx.dijkstra_path(G, source=start_node, target=end_node, weight='weight')
        distance = nx.dijkstra_path_length(G, source=start_node, target=end_node, weight='weight')
        return path, distance
    except Exception as e:
        print(f"An error occurred while finding the shortest path: {e}")
        return None, None

G = create_graph(locations)
path, distance = shortest_path(G, 0, 2)
print(f"Shortest path: {path} with distance: {distance}")
```

### Comments on the Program:
1. **Data Simulation**: This program starts by simulating some historical delivery data to predict fuel usage, which is integral to estimating emissions.
   
2. **Random Forest Regressor**: Uses a machine learning model to predict fuel usage based on route characteristics.

3. **Route Optimization**: Implements a simple permutation-based method to find the optimal delivery route with a minimum distance and incorporates error handling.

4. **Graph-based Optimization**: Uses NetworkX for more complex scenarios requiring graph-based shortest path algorithms.

5. **Error Handling**: Includes basic error handling to ensure graceful degradation in case of errors in processing.

### Additional Considerations:
- Real-world implementation would require extensive considerations for scaling, real-time data processing, and handling larger datasets with more features.
- Integrate proper logging, user authentication, real-time traffic data, and a GUI for a fully interactive platform.
- Consider incorporating environmental factors that may influence emissions, such as vehicle type and weather conditions.