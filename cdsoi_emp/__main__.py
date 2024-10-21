import numpy as np
import json
import os
from cdsoi_emp.invention_space import InventionSpace 
from cdsoi_emp.empowerment import empowerment
from cdsoi_emp.operations import Operations

directory = "cdsoi_emp\\data"
os.makedirs(directory)

if __name__ == "__main__":

    def ndarray_to_list_dict(ndarray_dict):
        return {k: v.tolist() for k, v in ndarray_dict.items()}

    # Initialize invention space
    #
    inv_space = InventionSpace(dimensions=10)
    for i in range(100):
        inv_space.add_invention(f"inv_{i}", np.random.randn(10))
    print("Invention Space: {}\n".format(inv_space.inventions))
    with open(os.path.join(directory, "invention_space_sample.json"), 'w') as f: 
        json.dump(ndarray_to_list_dict(inv_space.inventions), f)
    print("Invention Space saved to {}\n".format(f.name))

    emp = empowerment(inv_space.inventions, inv_space.dimensions, n_actions=10, n_steps=1)
    # Calculate empowerment field
    #
    emp_field = emp.empowerment_field(inv_space.inventions, n_actions=10, n_steps=1)
    print("Empowerment Field: {}\n".format(emp_field))
    with open(os.path.join(directory, "emp_field_sample.json"), 'w') as f:
        json.dump(ndarray_to_list_dict(emp_field), f)
    print("Empowerment Field saved to {}\n".format(f.name))

    # Calculate empowerment gradient
    #
    emp_gradient = emp.empowerment_gradient(inv_space.inventions, emp_field)
    print("Empowerment Gradient: {}\n".format(emp_gradient))
    with open(os.path.join(directory, "emp_grad_sample.json"), 'w') as f:
        json.dump(ndarray_to_list_dict(emp_gradient), f)
    print("Empowerment Gradients saved to {}\n".format(f.name))

    op = Operations()
    # Categorize inventions
    #
    categories = op.categorize_inventions(emp_field, 5)
    print("Identified Categories (by kmeans): {}\n".format(categories))
    with open(os.path.join(directory, "cluster_categories_sample.json") , 'w') as f:
        json.dump(ndarray_to_list_dict(categories), f)
    print("Clustered Categories saved to {}\n".format(f.name))

    """
    
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

    """