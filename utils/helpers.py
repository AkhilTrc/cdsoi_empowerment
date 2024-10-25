import random
import warnings

import numpy as np

def split_numbers(n_elements, split):
    """Splits elements into groups of (nearly) equal size.
    """
    
    x = np.arange(n_elements)   # Generates the index values. 
    random.shuffle(x)   # List of randomly shuffled index values. 

    return np.array_split(x, split)