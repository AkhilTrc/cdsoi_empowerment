import numpy as np
import json
import os
from cdsoi_emp.invention_space import InventionSpace 
from cdsoi_emp.empowerment import empowerment
from cdsoi_emp.operations import Operations
from cdsoi_emp.visualizations import Visualize

directory = "cdsoi_emp\\data"
os.makedirs(directory, exist_ok=True)

def ndarray_to_list_dict(ndarray_dict):
    return {k: v.tolist() for k, v in ndarray_dict.items()}

def json_dump(name, x):
    with open(os.path.join(directory, name+".json") , 'w') as f:
        json.dump(ndarray_to_list_dict(x), f)
    print("{} saved to {}\n".format(name, f.name))

if __name__ == "__main__":

    # Initialize invention space
    #
    inv_space = InventionSpace(dimensions=10)
    for i in range(100):
        inv_space.add_invention(f"inv_{i}", np.random.randn(10))
    print("Invention Space: {}\n".format(inv_space.inventions))
    json_dump("invention_space_sample", inv_space.inventions)

    emp = empowerment(inv_space.dimensions)
    # Calculate empowerment field
    #
    emp_field = emp.empowerment_field(inv_space.inventions, n_actions=10, n_steps=1)
    print("Empowerment Field: {}\n".format(emp_field))
    json_dump("emp_field_sample", emp_field)

    # Calculate empowerment gradient
    #
    emp_gradient = emp.empowerment_gradient(inv_space.inventions, emp_field)
    print("Empowerment Gradient: {}\n".format(emp_gradient))
    json_dump("emp_grad_sample", emp_gradient)

    op = Operations()
    # Categorize inventions
    #
    categories = op.categorize_inventions(emp_field, 5)
    print("Identified Categories (by kmeans): {}\n".format(categories))
    json_dump("cluster_categories_sample", categories)
    
    # Simulate invention trajectories
    #
    trajectories = [
        op.simulate_invention_trajectory(inv_space, emp_gradient, "inv_0", steps=10),
        op.simulate_invention_trajectory(inv_space, emp_gradient, "inv_50", steps=10)
    ]
    print("Tragenctories: {}\n".format(trajectories))
    
    viz = Visualize()
    # Visualize results
    # 
    viz.visualize_invention_space(inv_space, categories, trajectories)

    """
    # Evaluate effectiveness (simplified)
    #
    print("Number of inventions in each category:")
    print(np.bincount(list(categories.values())))

    print("\nAverage empowerment for each category:")
    for cat in set(categories.values()):
        cat_emp = [emp_field[inv_id] for inv_id, c in categories.items() if c == cat]
        print(f"Category {cat}: {np.mean(cat_emp):.2f}")

    """