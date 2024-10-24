import cdsoi_empowerment.utils.data_handle as data_handle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PreparedDataset():
    """Dataset containing train, test and (if wanted) validation and prediction set.
    """
    def __init__(self, data, prediction_model=0, exclusion_elements=None, manual_validation=None,
                 oversampling=None, custom_class_weight=None, vector_version='crawl300'):
        """Initializes prepared datasets that will be input to an element or link prediction model.
        """
        # print info for user
        print('\nGet data set.') 

        # set general attributes
        self.n_elements = 1000  # 1000 for now

        self.vector_version = vector_version
        self.prediction_model = prediction_model
        self.exclusion_elements = exclusion_elements
        self.manual_validation = manual_validation

        # split into train, test and validation set
        train_data, test_data, val_data = self.split_train_val_test_sets(data)

        if self.prediction_model == 0:
            # analyze imbalance
            self.output_bias, self.class_weight  = self.analyze_imbalance(train_data, test_data, val_data)
            if custom_class_weight is not None:
                self.class_weight = custom_class_weight

            # apply oversampling
            self.oversampling = oversampling
            if self.oversampling is not None:
                self.output_bias, self.class_weight = None, None
                if custom_class_weight is not None:
                    self.class_weight = custom_class_weight
                train_data = self.apply_oversampling(train_data)

        # transform data into vectors
        self.X_train, self.y_train, self.idx_train = self.get_feature_and_label_vector(train_data, type=0)
        self.X_test, self.y_test, self.idx_test = self.get_feature_and_label_vector(test_data, type=1)
        if val_data is not None:
            self.X_val, self.y_val, self.idx_val = self.get_feature_and_label_vector(val_data, type=2)

        # normalize
        self.normalize()

    def split_train_val_test_sets(self, data): #TODOOOOOO
        """Returns train, test and validation sets.
        """
        # initialize val_table
        val_table = None

        if self.exclusion_elements is None:
            # extract info from argument
            train_table = data[0]
            test_table = data[1]

            # get total size of data entries
            n_data = len(train_table.index) + len(test_table.index)

            # split train set further into test and validation set depending on manual_validation
            if self.manual_validation is not None:
                split = np.split(train_table.sample(frac=1), [int(self.manual_validation*len(train_table.index))])
                val_table = split[0]
                train_table = split[1]
        else:
            # print info
            print('\nExclude from test set: {}'.format(self.exclusion_elements[0]))

            # extract info from argument
            table = data

            # get total size of data entries
            n_data = len(table.index)

            # split into train and test sets
            first_exclusion_test = table.eval('first in @self.exclusion_elements[0]')
            second_exclusion_test = table.eval('second in @self.exclusion_elements[0]')
            if self.prediction_model != 1:
                test_condition = (first_exclusion_test | second_exclusion_test)
            else:
                result_exclusion_test = table.eval('result in @self.exclusion_elements[0]')
                test_condition = (first_exclusion_test | second_exclusion_test | result_exclusion_test)
            train_table = table.loc[~test_condition]
            test_table = table.loc[test_condition]

            # split train set further into test and validation set
            if len(self.exclusion_elements) == 2:
                # print info
                print('\nExclude from validation set: {}'.format(self.exclusion_elements[1]))

                first_exclusion_val = table.eval('first in @self.exclusion_elements[1]')
                second_exclusion_val = table.eval('second in @self.exclusion_elements[1]')
                if self.prediction_model != 1:
                    val_condition = (first_exclusion_val | second_exclusion_val)
                else:
                    result_exclusion_val = table.eval('result in @self.exclusion_elements[1]')
                    val_condition = (first_exclusion_val | second_exclusion_val | result_exclusion_val)
                val_table = train_table.loc[val_condition]
                train_table = train_table.loc[~val_condition]
            elif self.manual_validation is not None:
                split = np.split(train_table.sample(frac=1), [int(self.manual_validation*len(train_table.index))])
                val_table = split[0]
                train_table = split[1]

        # print info
        print('\nTrain set: {:.2f}%'.format(100 * len(train_table.index)/ n_data))
        print('\nTest set: {:.2f}%'.format(100 * len(test_table.index)/ n_data))
        if val_table is not None:
            print('\nValidation set: {:.2f}%'.format(100 * len(val_table.index)/ n_data))

        # return data sets
        return train_table, test_table, val_table

    def analyze_imbalance(self, *tables):
        """Analyzes class values and returns initial bias and class weights.
        """
        # print info
        print('\nAnalyze imbalance:')

        class_values = list()
        for table in tables:
            if table is not None:
                class_values += table['success'].values.tolist()

        # get counts for each class and total
        class_zero, class_one = np.bincount(class_values)   # Number of succesfull and failed combinations. 
        total = class_zero + class_one
        print('\t- Class 0: {:.2f}%\n\t- Class 1: {:.2f}%'.format(100 * class_zero / total, 100 * class_one / total))

        # get initial bias
        initial_bias = np.log([class_one/class_zero])
        print('\t- Initial bias: {}'.format(initial_bias))

        # get class weights
        weight_for_zero = (1 / class_zero)*(total)/2.0
        weight_for_one = (1 / class_one)*(total)/2.0
        class_weight = {0: weight_for_zero, 1: weight_for_one}
        print('\t- Weight for Zero: {}\n\t- Weight for Once: {}\n\t- Class Weight: {}\n'.format(weight_for_zero, weight_for_one, class_weight))

        return initial_bias, class_weight

    def apply_oversampling(self, table):
        """Applies oversampling on data.
        """
        # print info
        print('\nApply oversampling.')

        # extract tables for success or no success
        success_condition = table.eval('success == 1')
        class_one_table = table.loc[success_condition].values
        class_zero_table= table.loc[~success_condition].values

        # oversample: balance out both classes by randomly resampling success group
        ids = np.arange(class_one_table.shape[0])
        choices = np.random.choice(ids, int(class_zero_table.shape[0]*self.oversampling))
        class_one_table_oversampled = class_one_table[choices]

        # join both tables
        table = np.concatenate([class_one_table_oversampled, class_zero_table], axis=0)
        table = pd.DataFrame({'first': table[:, 0], 'second': table[:, 1], 'success': table[:,2], 'result': table[:,3]})

        # return shuffled joined table with balanced classes
        return table.sample(frac=1)

    def normalize(self):
        """Normalizes input vectors.
        """
        # print info for user
        print('\nNormalize data.')

        # normalize
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        if hasattr(self, 'X_val'):
            self.X_val = scaler.transform(self.X_val)

    def get_feature_and_label_vector(self, data, type=0):
        """Returns feature and corresponding label vectors. Feature vectors are made up of concatenated word vectors, labels are either 0 or 1.
        """
        # print info for user
        print('\nGet feature and label vectors.')

        self.parent_table = data_handle.get_parent_table(self.game_version)
        self.combination_table = data_handle.get_combination_table(self.game_version, csv=False)

        # get list of element vectors
        element_vectors = data_handle.get_wordvectors(self.game_version, self.vector_version)

        # for element prediction model, empowerment prediction model and train and val data: reduce data to only successful data
        if self.prediction_model != 0 and (type == 0 or type == 2):
            data.query('success == 1', inplace=True)
        data = data.values      # Reduces the data to only the combinations that are succesful (i.e = 1)

        # initialize feature vector storage
        combination_count = data.shape[0]       # All succesful Combinations count.  
        vector_dim = element_vectors.shape[1]       # Length of Word Vector. Size of crawl300 in this case. 
        feature_vectors = np.empty((2*combination_count,2*vector_dim))      # Create empty matrix with the 2*dimensions of above measures.

        # initialize label vector storage
        if self.prediction_model != 1:
            label_vector = np.empty(2*combination_count)
        else:
            label_vector = np.empty((2*combination_count, vector_dim))

        # initialize element index storage
        indices = np.empty((2*combination_count,4))

        # create combination features and get corresponding result vectors
        for idx, combination in enumerate(data):
            combination_success = [int(x) for x in combination[:3]]  # I don't understand this!!
            result = combination[3]
            feature_vectors[idx,:] = np.concatenate([element_vectors[combination_success[0]], element_vectors[combination_success[1]]], axis = 0)    # Contains word vectors for the first and second elements for the combination.
            feature_vectors[idx+combination_count,:] = np.concatenate([element_vectors[combination_success[1]], element_vectors[combination_success[0]]], axis = 0)    # # Contains word vectors for the first and second elements for the combination reversed.

            if self.prediction_model == 0:  # Means Link prediction. 
                label_vector[idx] = combination_success[2]      # 0/1 based on success/failure of combination.
                label_vector[idx+combination_count] = combination_success[2]    # 0/1 based on s/f of combination reversed. 
            
            elif self.prediction_model ==2:     # Empowerment prediction model. 
                #empvalue of result

                if combination[0] in self.combination_table and combination[1] in self.combination_table[combination[0]]:
                    result_elements = self.combination_table[combination[0]][combination[1]]
                else:
                    result_elements = list()

                # initialize empowerment depending on how it is calculated
                empowerment = set()

                # calculate empowerment value iteratively for each result
                for r in result_elements:
                    if r in self.parent_table:
                        empowerment.update(self.parent_table[r])

                label_vector[idx] = len(empowerment)
                label_vector[idx+combination_count] = len(empowerment)
            else:
                if not np.isnan(result):
                    label_vector[idx,:] = element_vectors[int(result)]
                    label_vector[idx+combination_count,:] = element_vectors[int(result)]

            indices[idx,:] = combination
            indices[idx+combination_count,:] = combination

        print('\t\t... shape input vectors:', feature_vectors.shape)
        print('\t\t... shape label vector:', label_vector.shape)

        # return feature vectors and corresponding label and index vectors
        return feature_vectors, label_vector, indices
