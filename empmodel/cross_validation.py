import random
import pandas as pd

import cdsoi_empowerment.utils.data_handle as data_handle
import cdsoi_empowerment.utils.helpers as helpers
import cdsoi_empowerment.utils.info_logs as log

from element_prediction_model import ElementPredictionModel
from link_prediction_model import LinkPredictionModel
from empowerment_prediction_model import EmpowermentPredictionModel
from prepare_dataset import PreparedDataset
from sklearn.model_selection import KFold


class CrossValidation():
    def __init__(self, k, time,
                 prediction_model=0, epochs=1, batch_size=32, steps_per_epoch=None,
                 exclude_elements_test=True, exclude_elements_val=True, manual_validation=None,
                 oversampling=None, custom_class_weight=None,
                 vector_version='crawl', dim=300):
        """Initialize k-fold cross validation.
        """

        # set info on cross validation
        self.k = k

        # set vector info
        self.vector_version = '{}{}'.format(vector_version, dim)    # = 'crawl300' in this example.

        # set general info
        self.time = time
        self.round = 1

        # set info on game version, number of elements = 1000 for now
        self.n_elements = 1000

        # set info on model
        self.prediction_model = prediction_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        # set info for data generation
        self.exclude_elements_test = exclude_elements_test
        self.exclude_elements_val = exclude_elements_val
        if self.exclude_elements_test is True or exclude_elements_val is True:
            self.split_version = 'element'
        else:
            self.split_version = 'data'
        self.manual_validation = manual_validation
        self.oversampling = oversampling
        self.custom_class_weight = custom_class_weight

    def run_cross_validation(self):
        """Run cross validation.
        """
        # print info for user
        print('\nRun cross validation.')

        # create file for predicted probabilities
        log.create_gametreetable_file(time=self.time, prediction_model=self.prediction_model, n_elements=self.n_elements, split_version=self.split_version, vector_version=self.vector_version)

        # load and shuffle dataset
        data_table = data_handle.get_combination_table()
        data_table = data_table.sample(frac=1)      

        #if self.prediction_model == 0:
        if self.prediction_model != 1:
            data_table = data_table.drop_duplicates(subset=['first', 'second'])

        # initialize storage for results
        result_metrics = list()

        if self.exclude_elements_test is True:
            # make a k-fold cross-validation that has distinct element groups for each set
            element_groups = helpers.split_numbers(self.n_elements, self.k)     

            for idx, test_group in enumerate(element_groups):
                # print info for user
                print('\nRound {} of {}.'.format(self.round, self.k))

                # get data
                if self.exclude_elements_val is True:
                    # choose a random group of elements to make up the validation set
                    val_group = element_groups[random.choice([group for group in range(len(element_groups)) if group != idx])]

                    data = PreparedDataset(data_table, prediction_model=self.prediction_model,
                                           exclusion_elements=(test_group,val_group),
                                           oversampling=self.oversampling, custom_class_weight=self.custom_class_weight,
                                           vector_version=self.vector_version)
                else:
                    data = PreparedDataset(data_table, prediction_model=self.prediction_model,
                                           exclusion_elements=(test_group,), manual_validation=self.manual_validation,
                                           oversampling=self.oversampling, custom_class_weight=self.custom_class_weight,
                                           vector_version=self.vector_version)

                # run models and update result storage
                test_metrics = self.run_cross_validation_round(data)
                result_metrics.append(test_metrics)
        else:
            # make a random k-fold cross-validation
            kf = KFold(n_splits=self.k)

            for train_index, test_index in kf.split(data_table):
                # get data
                data = PreparedDataset((data_table.iloc[train_index], data_table.iloc[test_index]), prediction_model=self.prediction_model,
                                       manual_validation=self.manual_validation,
                                       oversampling=self.oversampling, custom_class_weight=self.custom_class_weight,
                                       vector_version=self.vector_version)

                # run models and update result storage
                test_metrics = self.run_cross_validation_round(data)
                result_metrics.append(test_metrics)

        # save performance
        results = pd.DataFrame(result_metrics)
        if self.prediction_model == 0:
            results.to_csv('cdsoi_empowerment/alldata/gametree/{}/LinkPred-metrics.csv'.format(self.time), index=False)
        elif self.prediction_model == 1:
            results.to_csv('cdsoi_empowerment/alldata/gametree/{}/ElemPred-metrics.csv'.format(self.time), index=False)
        else:
            results.to_csv('cdsoi_empowerment/alldata/gametree/{}/EmpPred-metrics.csv'.format(self.time), index=False)

    def run_cross_validation_round(self, data):
        """Runs one cross validation round with given data.
        """

        # evaluate model
        if self.prediction_model == 0:
            model = LinkPredictionModel(self.time, self.round, self.epochs, self.batch_size, self.steps_per_epoch, data.class_weight, data.output_bias)
            if self.manual_validation is not None or self.exclude_elements_val is True:
                predictions, test_metrics = model.evaluate_model((data.X_train, data.X_test, data.X_val), (data.y_train, data.y_test, data.y_val))
            else:
                predictions, test_metrics = model.evaluate_model((data.X_train, data.X_test), (data.y_train, data.y_test), validation_split=0.2)
        elif self.prediction_model == 1:
            model = ElementPredictionModel(self.time, self.round, self.epochs, self.batch_size, self.steps_per_epoch, self.vector_version)
            if self.manual_validation is not None or self.exclude_elements_val is True:
                predictions, test_metrics = model.evaluate_model((data.X_train, data.X_test, data.X_val),
                                                                 (data.y_train, data.y_test, data.y_val), (data.idx_train, data.idx_test))
            else:
                predictions, test_metrics = model.evaluate_model((data.X_train, data.X_test),
                                                                 (data.y_train, data.y_test), (data.idx_train, data.idx_test), validation_split=0.2)
        else:
            model = EmpowermentPredictionModel(self.time, self.round, self.epochs, self.batch_size, self.steps_per_epoch) # data.output_bias
            if self.manual_validation is not None or self.exclude_elements_val is True:
                predictions, test_metrics = model.evaluate_model((data.X_train, data.X_test, data.X_val), (data.y_train, data.y_test, data.y_val))
            else:
                predictions, test_metrics = model.evaluate_model((data.X_train, data.X_test), (data.y_train, data.y_test), validation_split=0.2)

        # write to file
        log.append_gametreetable_file(data.idx_test, predictions, time=self.time, prediction_model=self.prediction_model, split_version=self.split_version, vector_version=self.vector_version)

        # increment round
        self.round += 1

        return test_metrics
