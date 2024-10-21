import numpy as np

class empowerment():
    def __init__(self, inv_space, dimensions, n_actions=10, n_steps=1):
        self.inv_space = inv_space
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.dimensions = dimensions

    def calculate_empowerment(self, invention, actions, n_steps):
        # Simplified empowerment calculation
        # In a real implementation, this would involve complex channel capacity calculations
        return np.sum(np.abs(invention + np.sum(actions, axis=0)))

    def empowerment_field(self, inventions, n_actions, n_steps):
        field = {}
        for inv_id, inv_features in inventions.items():
            actions = np.random.randn(n_actions, self.dimensions)
            field[inv_id] = self.calculate_empowerment(inv_features, actions, n_steps)

        return field

    def empowerment_gradient(self, invention_space, emp_field):
        gradient = {}
        for inv_id, inv_features in invention_space.items():
            grad = np.zeros(self.dimensions)
            for d in range(self.dimensions):
                perturbed = inv_features.copy()
                perturbed[d] += 0.01
                emp_perturbed = self.calculate_empowerment(perturbed, np.random.randn(10, self.dimensions), 1)
                grad[d] = (emp_perturbed - emp_field[inv_id]) / 0.01
            gradient[inv_id] = grad
            
        return gradient