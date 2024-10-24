import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from visualization import Visualization
from tensorflow import keras
from tensorflow.keras import layers


class EmpowermentPredictionModel():
    """Model that predicts empowerments of combinations.
    """
    def __init__(self, time, step, epochs=1, batch_size=32, steps_per_epoch=None): #output_bias
        """Initializes an empowerment prediction model.
        """
        # print info for user
        print('\nInitialize empowerment prediction model.')

        # set attributes
        self.time = time
        self.step = step
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        #self.output_bias = output_bias

    def evaluate_model(self, X, y, validation_split=None):
        """Get, train and evaluate model for predicting links between two elements.
        """
        # extract train and test data
        X_train, y_train = X[0], y[0]
        X_test, y_test = X[1], y[1]

        # get input size
        n_inputs = X_train.shape[1]


        # build model
        model = self.build_model(n_inputs)
        model.summary()

        # apply early stopping
        # early_stop = keras.callbacks.EarlyStopping(monitor='val_auc',
        #                                           verbose=1, mode='max',
        #                                           patience=15, restore_best_weights=True)

        # print info for user
        print('\nFit empowerment prediction model.')

        if validation_split is not None:
            # train model
            history = model.fit(X_train, y_train,
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                validation_split=validation_split,
                                verbose=0,
                                callbacks=[tfdocs.modeling.EpochDots()])
                                #          tf.keras.callbacks.TensorBoard(log_dir='./logs')],

        else:
            X_val, y_val = X[2], y[2]

            # train model
            history = model.fit(X_train, y_train,
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                validation_data = (X_val, y_val),
                                verbose=0,
                                callbacks=[tfdocs.modeling.EpochDots()])
                                #          tf.keras.callbacks.TensorBoard(log_dir='./logs')],


        # visualization of training progress
        visualization = Visualization(self.time, 'EmpPred', self.step)
        visualization.plot_progress(history, ['loss', 'mse', 'mae'])

        # make predictions and plot confusion matrix and ROC
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # check generalization on test data
        test_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

        # print info for user
        print('\nReturn results from empowerment prediction model.')

        # return test predictions and test metrics
        return test_predictions, test_metrics


    def build_model(self, n_inputs):
        """Creates and returns model for predicting links between two elements.
        """
        # print info for user
        print('\nBuild empowerment prediction model.')

        # set output bias
        #if self.output_bias is not None:
            #output_bias = tf.keras.initializers.Constant(self.output_bias)
        #else:
            #output_bias = None

        # define architecture
        model = keras.Sequential([
            layers.Dense(16, activation='relu', input_shape=[n_inputs]),
            layers.Dense(1, activation='relu')
        ])

        # define learning rate
        lr = 1e-3

        # define optimizer
        optimizer = tf.keras.optimizers.Adam(lr)

        # define metrics
        metrics = [
            keras.metrics.MeanSquaredError(name='mse'),
            keras.metrics.MeanAbsoluteError(name='mae'),
            #keras.metrics.Accuracy(name='accuracy'),
        ]

        # compile model
        model.compile(loss= keras.losses.MeanSquaredError(),
                      optimizer=optimizer,
                      metrics=metrics)

        return model
