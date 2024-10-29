import json

import utils.helpers as helpers
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

def get_wordvectors(vector_version='crawl300'):
    """Returns wordvectors for given version. 
    """

    if vector_version == 'ccen100' or vector_version == 'ccen300' or vector_version == 'crawl100' or vector_version == 'crawl300' or vector_version == 'wiki100' or vector_version == 'wiki300':
        vectors = np.loadtxt('semantimodel/data/ElementVectors-{}.txt'.format(vector_version))
    else:
        raise ValueError('Undefined vector_version: "{}". Use "ccen100", "ccen300", "crawl100", "crawl300", "wiki100" or "wiki300" instead.'.format(vector_version))
   
    return vectors