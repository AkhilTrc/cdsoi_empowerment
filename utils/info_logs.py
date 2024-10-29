import csv
import json
import os

import numpy as np
import pandas as pd


def create_directory(directory_name):
    """Creates directory if it was not existent before.

    Args:
        directory_name (str): Directory path.
    """
    os.makedirs(os.path.dirname(directory_name), exist_ok=True)
    
def create_gametreetable_file(time=None, prediction_model=0, n_elements=None, split_version='data', vector_version='crawl300'):
    """Creates file for later result logging depending on model type (link or element prediction model).
    """

    if prediction_model == 0:
        append_gametreetable_file(['first', 'second', 'trueSuccess', 'trueResult', 'predSuccess'], time=time, prediction_model=prediction_model, first_line=True, split_version=split_version, vector_version=vector_version)
    elif prediction_model == 1:
        first_line = ['first', 'second', 'trueSuccess', 'trueResult']
        for element_idx in range(n_elements):
            first_line.append(str(element_idx))
        append_gametreetable_file(first_line, time=time, prediction_model=prediction_model, first_line=True, split_version=split_version, vector_version=vector_version)
    else:
        append_gametreetable_file(['first', 'second', 'trueSuccess', 'trueResult', 'predEmp'], time=time, prediction_model=prediction_model, first_line=True, split_version=split_version, vector_version=vector_version)


def append_gametreetable_file(*data, time=None, prediction_model=0, first_line=False, split_version='data', vector_version='crawl300'):
    """Writes test results to csv file continuously.
    """

    # write first line
    if first_line is False:
        data_new = np.concatenate((data[0], data[1]), axis=1)
    else:
        data_new = data

    # append data
    for line in data_new:
        if prediction_model == 0:
            with open('cdsoi_empowerment/alldata/gametree/{}/LinkPredTable-{}-{}.csv'.format(time, split_version, vector_version), 'a+', newline='') as outfile:
                # = '/235545/tinyalchemyLinkPredTable-data-crawl300.csv' for this example. 
                writer = csv.writer(outfile)
                writer.writerow(line)
        elif prediction_model == 1:
            with open('cdsoi_empowerment/alldata/gametree/{}/ElemPredTable-{}-{}.csv'.format(time, split_version, vector_version), 'a+', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(line)
        else:
            with open('cdsoi_empowerment/alldata/gametree/{}/EmpPredTable-{}-{}.csv'.format(time, split_version, vector_version), 'a+', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(line)