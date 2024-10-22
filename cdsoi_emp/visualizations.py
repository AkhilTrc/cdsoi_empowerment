import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class Visualize():
    def __init__(self):
        pass

    def visualize_invention_space(self, invention_space, categories, trajectories):
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
        plt.title('invention_space_visualization')
        title_str = plt.gca().get_title()
        plt.tight_layout()
        # plt.show()
        plt.savefig('cdsoi_emp/data/figures/{}'.format(title_str))
        plt.close()