import json
import csv
import utils.data_handle as data_handle
import numpy as np
import pandas as pd

class CustomEmpowerment():
    def __init__(self, vector_version, split_version, n_elements):

        self.split_version = split_version
        self.vector_version = vector_version
        self.n_elements = n_elements

        self.calcualte_empowerment()

    def calculate_empowerment(self):
        """From Custom-designed Gametree, empowerment values are calculated.
        """

        # get probability table
        try:
            probability_table = pd.read_hdf('cdsoi_empowerment/alldata/gametree/GametreeTable-{}-{}.h5'.format(self.split_version, self.vector_version))
        except:
            raise ValueError('Corresponding custom gametree table not found. Check if input was correct or create the needed table using "gametree" package')

        # print info for user
        print('\nGet single element empowerment info.')

        # initialize storage for empowerment info
        empowerment_outgoing_combinations = dict()
        empowerment_children = dict()

        # get empowerment info for each element
        for element in range(self.n_elements):
            empowerment_outgoing_combinations[element] = self.get_empowerment(probability_table, float(element), True)
            empowerment_children[element] = self.get_empowerment(probability_table, float(element), False)

        # write to JSON file
        with open('cdsoi_empowerment/semanticmodel/data/OutgoingCombinationsEmpowermentTable-{}-{}.json'.format(self.split_version, self.vector_version), 'w') as filehandle:
            json.dump(empowerment_outgoing_combinations, filehandle, indent=4, sort_keys=True)

        with open('cdsoi_empowerment/semanticmodel/data/ChildrenEmpowermentTable-{}-{}.json'.format(self.split_version, self.vector_version), 'w') as filehandle:
            json.dump(empowerment_children, filehandle, indent=4, sort_keys=True)

        # print info for user
        print('\nGet combination empowerment info.')

        # initialize logging to csv file
        self.append_empowermenttable_file(['first', 'second', 'predResult', 'empComb', 'empChild', 'binComb', 'binChild'], first_line=True)

        # get empowerment info for each combination
        for element in range(self.n_elements):
            # get combination probabilities
            combination_probabilities = probability_table.query('first == @element')

            # get combination elements
            combination_elements = combination_probabilities.index.values
            combination_elements = [list(x) for x in combination_elements]
            combination_elements = np.array(combination_elements)

            # get predicted result
            result = combination_probabilities['predResult'].values

            # get probability
            empowerment_probability = self.get_combination_empowerment(empowerment_outgoing_combinations, empowerment_children, combination_probabilities)

            # join arrays
            element_empowerment_info = np.zeros((combination_probabilities.shape[0],7))
            element_empowerment_info[:,:2] = combination_elements
            element_empowerment_info[:,2] = result
            element_empowerment_info[:,3:] = empowerment_probability

            # write to file
            self.append_empowermenttable_file(element_empowerment_info, first_line=False)

        print('\nDone.')

    def get_empowerment(self, probability_table, element, outgoing_combinations):
        """Returns predicted empowerment for single element depending on the calculation type. 
        """

        # check how many outgoing combinations with element are possible
        combinations = probability_table.query('first==@element | second==@element')
           
        # get successful combinations (probability of success > 0.5)
        successful_combination = combinations.query('predSuccess >= 0.5')
                 
        if outgoing_combinations is True:
            # get number of successful combinations
            element_empowerment_info = len(successful_combination.index)
        else:            
            # extract only element probability columns
            element_probabilities = successful_combination[successful_combination.columns[-self.n_elements:-1]]
            
            # get elements with maximum probability
            element_empowerment_info = element_probabilities.idxmax(axis='columns').astype(int).unique().tolist()
            
        return element_empowerment_info

    def get_combination_empowerment(self, empowerment_outgoing_combinations, empowerment_children, combination_probabilities):       
        """Returns predicted empowerment for combinations.
        """

        # get probability of success
        p_success = combination_probabilities['predSuccess'].values
        
        # initialize summed element probabilities
        p_sum = np.zeros((len(p_success), 4))
        
        for element in range(self.n_elements):
            # get empowerment value
            emp_out = empowerment_outgoing_combinations[element]
            emp_chi = len(empowerment_children[element])
            
            # get binary value
            if emp_out > 0:
                bin_out = 1
            else:
                bin_out = 0
                
            if emp_chi > 0:
                bin_chi = 1
            else:
                bin_chi= 0
            
            # get probability that this element is the result
            avg_element_probability = combination_probabilities[str(element)].values
            
            # multiply and add to sum
            p_sum[:,0] = p_sum[:,0] + avg_element_probability*emp_out
            p_sum[:,1] = p_sum[:,1] + avg_element_probability*emp_chi
            p_sum[:,2] = p_sum[:,2] + avg_element_probability*bin_out
            p_sum[:,3] = p_sum[:,3] + avg_element_probability*bin_chi
        
        # set empowerment value for combination
        return p_sum * p_success[:, None]

    def append_empowermenttable_file(self, *data, first_line=False):
        """Writes test results to csv file continuously.
        """

        # write first line
        if first_line is True:
            data_new = data
        else:
            data_new = data[0]

        # append data
        for line in data_new:
            with open('cdsoi_empowerment/semanticmodel/data/EmpowermentTable-{}-{}.csv'.format(self.split_version, self.vector_version), 'a+', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(line)
