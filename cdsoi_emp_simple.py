import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans, vq
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt

class InventionSpace:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.inventions = {}
        
    def add_invention(self, invention_id, features):
        self.inventions[invention_id] = np.array(features)
        
    def get_invention(self, invention_id):
        return self.inventions[invention_id]

def calculate_empowerment(invention, actions, n_steps):
    # Simplified empowerment calculation
    # In a real implementation, this would involve complex channel capacity calculations
    return np.sum(np.abs(invention + np.sum(actions, axis=0)))

def empowerment_field(invention_space, n_actions, n_steps):
    field = {}
    for inv_id, inv_features in invention_space.inventions.items():
        actions = np.random.randn(n_actions, invention_space.dimensions)
        field[inv_id] = calculate_empowerment(inv_features, actions, n_steps)
    return field

def empowerment_gradient(invention_space, emp_field):
    gradient = {}
    for inv_id, inv_features in invention_space.inventions.items():
        grad = np.zeros(invention_space.dimensions)
        for d in range(invention_space.dimensions):
            perturbed = inv_features.copy()
            perturbed[d] += 0.01
            emp_perturbed = calculate_empowerment(perturbed, np.random.randn(10, invention_space.dimensions), 1)
            grad[d] = (emp_perturbed - emp_field[inv_id]) / 0.01
        gradient[inv_id] = grad
    return gradient

def categorize_inventions(emp_field, n_categories):
    emp_values = np.array(list(emp_field.values())).reshape(-1, 1)
    centroids, _ = kmeans(emp_values, n_categories)
    categories, _ = vq(emp_values, centroids)
    return dict(zip(emp_field.keys(), categories))

def dic_to_dfcsv(dictionary, count):
    df = pd.DataFrame(list(dictionary.items()), columns=['Key', 'Value'])
    df.to_csv('csv_saved/empgradient_{}.csv'.format(count), index=False)
    print("--- Empowerment Gradient saved as [empgradient_{}.csv] ---\n".format(count))

def simulate_invention_trajectory(invention_space, emp_gradient, start_inv_id, steps):
    trajectory = [start_inv_id]
    current_inv = invention_space.get_invention(start_inv_id)
    # count = 2   # next = 3
    # dic_to_dfcsv(emp_gradient, count)
    for _ in range(steps):
        grad = emp_gradient[start_inv_id]
        current_inv += 0.1 * grad  # Step size of 0.1
        new_inv_id = f"inv_{len(invention_space.inventions)}"
        invention_space.add_invention(new_inv_id, current_inv)
        trajectory.append(new_inv_id)
        start_inv_id = new_inv_id
    return trajectory

def visualize_invention_space(invention_space, categories, trajectories):
    # Use t-SNE for dimensionality reduction
    features = np.array(list(invention_space.inventions.values()))
    tsne = TSNE(n_components=2, random_state=42)
    coords = tsne.fit_transform(features)
    
    # Plot inventions
    plt.figure(figsize=(12, 8))
    for cat in set(categories.values()):
        cat_coords = [coords[i] for i, (inv_id, c) in enumerate(categories.items()) if c == cat]
        cat_coords = np.array(cat_coords)
        plt.scatter(cat_coords[:, 0], cat_coords[:, 1], label=f'Category {cat}')
    
    # Plot trajectories
    for traj in trajectories:
        traj_coords = [coords[list(invention_space.inventions.keys()).index(inv_id)] for inv_id in traj]
        traj_coords = np.array(traj_coords)
        plt.plot(traj_coords[:, 0], traj_coords[:, 1], 'r-')
    
    plt.legend()
    plt.title('Invention Space Visualization')
    plt.show()

if __name__ == "__main__":
    # Initialize invention space
    #
    inv_space = InventionSpace(dimensions=10)
    for i in range(100):
        inv_space.add_invention(f"inv_{i}", np.random.randn(10))
    print("Invention Space: {}\n".format(inv_space.inventions))
    
    # Calculate empowerment field
    #
    emp_field = empowerment_field(inv_space, n_actions=10, n_steps=1)
    print("Empowerment Field: {}\n".format(emp_field))
    
    # Calculate empowerment gradient
    #
    emp_gradient = empowerment_gradient(inv_space, emp_field)
    print("Empowerment Gradient: {}\n".format(emp_gradient))

    # Categorize inventions
    #
    categories = categorize_inventions(emp_field, n_categories=5)
    print("Identified Categories (by kmeans): {}\n".format(categories))
    
    # Simulate invention trajectories
    #
    trajectories = [
        simulate_invention_trajectory(inv_space, emp_gradient, "inv_0", steps=10),
        simulate_invention_trajectory(inv_space, emp_gradient, "inv_50", steps=10)
    ]
    print("Tragenctories: {}\n".format(trajectories))
    
    # Visualize results
    # 
    visualize_invention_space(inv_space, categories, trajectories)
    
    # Evaluate effectiveness (simplified)
    #
    print("Number of inventions in each category:")
    print(np.bincount(list(categories.values())))
    
    print("\nAverage empowerment for each category:")
    for cat in set(categories.values()):
        cat_emp = [emp_field[inv_id] for inv_id, c in categories.items() if c == cat]
        print(f"Category {cat}: {np.mean(cat_emp):.2f}")