import numpy as np

class InventionSpace():
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.inventions = {}
        
    def add_invention(self, invention_id, features):
        self.inventions[invention_id] = np.array(features)
        
    def get_invention(self, invention_id):
        return self.inventions[invention_id]