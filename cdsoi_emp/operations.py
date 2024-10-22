import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq
from cdsoi_emp.invention_space import InventionSpace 

class Operations():
    def __int__():
        pass

    def categorize_inventions(self, emp_field, n_categories):
        emp_values = np.array(list(emp_field.values())).reshape(-1, 1)
        centroids, _ = kmeans(emp_values, n_categories)
        categories, _ = vq(emp_values, centroids)

        return dict(zip(emp_field.keys(), categories))

    def dic_to_dfcsv(self, dictionary, count):
        df = pd.DataFrame(list(dictionary.items()), columns=['Key', 'Value'])
        df.to_csv('csv_saved/empgradient_{}.csv'.format(count), index=False)
        print("--- Empowerment Gradient saved as [empgradient_{}.csv] ---\n".format(count))

    def simulate_invention_trajectory(self, inventions, emp_gradient, start_inv_id, steps):
        trajectory = [start_inv_id]

        current_inv = inventions.get_invention(start_inv_id)
        # dic_to_dfcsv(emp_gradient, count)
        for _ in range(steps):
            if start_inv_id in emp_gradient:
                grad = emp_gradient[start_inv_id]
                current_inv += 0.1 * grad  # Step size of 0.1
                new_inv_id = f"inv_{len(inventions.inventions)}"
                inventions.add_invention(new_inv_id, current_inv)
                trajectory.append(new_inv_id)
                start_inv_id = new_inv_id
            else:
                continue

        return trajectory