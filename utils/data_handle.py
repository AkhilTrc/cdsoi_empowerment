import json

import cdsoi_empowerment.utils.helpers as helpers
import numpy as np
import pandas as pd
import scipy.io

def get_combination_table(csv=True):
    """Returns a combination table with four columns:
    """
    if csv is True:
        combination_table = pd.read_csv('cdsoi_empowerment/alldata/CombinationTable.csv', dtype={'first': int, 'second': int, 'success': int, 'result': int})
    else:
        with open('cdsoi_empowerment/alldata/CombinationTable.json', encoding='utf8') as infile:
            combination_table = json.load(infile, object_hook=helpers.jsonKeys2int)

    return combination_table