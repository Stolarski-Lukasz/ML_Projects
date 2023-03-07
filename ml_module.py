import pandas as pd
import numpy as np
from abc import ABC, abstractclassmethod

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import settings as s



# Data processing
#################
class IDataProcessor(ABC):

    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_features = None
        self.n_label_levels = None
        self.model_type = None
        self.evaluation_metric = None

    @abstractclassmethod
    def prepare_data(self, model_type):
        pass


class DataProcessor(IDataProcessor):

    def __init__(self, data):
        super().__init__(data)

    def prepare_data(self, model_type):
        self.data = shuffle(self.data)
        self.n_features = len(s.X)
        self.X = self.data[s.X]
        # one-hot encoding performed for classification models
        if model_type == "classification":
            self.n_label_levels = self.data[s.y].nunique()
            encoder = LabelEncoder()
            labels = self.data[s.y]
            encoder.fit(labels)
            encoded_labels = encoder.transform(labels)
            final_labels = np_utils.to_categorical(encoded_labels)
            self.y = final_labels
            self.model_type = "classification"
            self.evaluation_metric = "accuracy"
        elif model_type == "regression":
            self.y = self.data[s.y]
            self.model_type = "regression"
            self.evaluation_metric = "mean absolute error"
        else:
            raise ValueError('model_type can only be either "classification" or "regression"')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=s.TEST_SIZE)


class SklDataProcessor(IDataProcessor):

    def __init__(self, data):
        super().__init__(data)

    def prepare_data(self, model_type):
        self.X = self.data[s.X]
        self.y = self.data[s.y]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=s.TEST_SIZE)


    
        

# Model evaluation
##################
class IModelEvaluator(ABC):

    def __init__(self, data_processor: IDataProcessor, model):
        self.data_processor = data_processor
        self.model = model

    @abstractclassmethod
    def validate_model_single_fold(self, is_train_test_split=True):
        pass

    @abstractclassmethod
    def validate_model_k_fold(self, is_train_test_split=True):
        pass

    @abstractclassmethod
    def test_model(self):
        pass


class DeepModelEvaluator(IModelEvaluator):

    def __init__(self, data_processor: IDataProcessor, model):
        super().__init__(data_processor, model)

    def validate_model_single_fold(self, is_train_test_split=True):
        if is_train_test_split == True:
            print(f"Starting training the model with the last {int(s.VALIDATION_SPLIT * 100)}% of the training data as the validation set.")
            self.model.fit(self.data_processor.X_train, 
                                    self.data_processor.y_train, 
                                    batch_size=s.BATCH_SIZE, 
                                    epochs=s.N_EPOCHS, 
                                    shuffle=True, 
                                    verbose=s.VERBOSE,
                                    validation_split=s.VALIDATION_SPLIT)
        else:
            print(f"Starting training the model with the last {int(s.VALIDATION_SPLIT * 100)}% of all the data as the validation set.")
            self.model.fit(self.data_processor.X, 
                                    self.data_processor.y, 
                                    batch_size=s.BATCH_SIZE, 
                                    epochs=s.N_EPOCHS, 
                                    shuffle=True, 
                                    verbose=s.VERBOSE,
                                    validation_split=s.VALIDATION_SPLIT)


    def validate_model_k_fold(self, is_train_test_split=True):
        if is_train_test_split == True:
            X = self.data_processor.X_train
            Y = self.data_processor.y_train
            info_text = "training data"
        else:
            X = self.data_processor.X
            Y = self.data_processor.y
            info_text = "all data"
        X = X.to_numpy()
        if self.data_processor.model_type == "regression":
            Y = Y.to_numpy()
        kfold = KFold(n_splits=s.N_FOLD, shuffle=True, random_state=7)
        cvscores = []
        print(f"Starting {s.N_FOLD}-fold cross-validation for a model built on {info_text}:")
        fold_counter = 1
        for train, test in kfold.split(X, Y):
            self.model.fit(X[train], 
                                    Y[train], 
                                    batch_size=s.BATCH_SIZE, 
                                    epochs=s.N_EPOCHS, 
                                    shuffle=True, 
                                    verbose=s.VERBOSE)

            score = self.model.evaluate(X[test], Y[test], verbose=0)
            print(f"The validation {self.data_processor.evaluation_metric} for fold {fold_counter} is {round(score[1], 4)}.")
            fold_counter += 1
            cvscores.append(score[1])
        print(f"The average validation {self.data_processor.evaluation_metric} for all {s.N_FOLD} folds is {round(np.mean(cvscores), 4)}.")


    def test_model(self):
        self.model.fit(self.data_processor.X_train, 
                                    self.data_processor.y_train, 
                                    batch_size=s.BATCH_SIZE, 
                                    epochs=s.N_EPOCHS, 
                                    shuffle=True, 
                                    verbose=s.VERBOSE)
        print("The final evaluation of the model on the basis of the testing set:")
        self.model.evaluate(self.data_processor.X_test, self.data_processor.y_test, verbose=s.VERBOSE)
        # print(f"The testing mean absolute error is {score[1]}.")
        


# Final model processing
########################
class IModelProcessor(ABC):

    def __init__(self, data_processor: IDataProcessor, model):
        self.data_processor = data_processor
        self.model = model

    @abstractclassmethod
    def train_model_final(self):
        pass

    @abstractclassmethod
    def save_model(self):
        pass


class DeepModelProcessor(IModelProcessor):

    def __init__(self, data_processor: IDataProcessor, model):
        super().__init__(data_processor, model)
    
    def train_model_final(self):
        self.model.fit(self.data_processor.X, 
                    self.data_processor.y, 
                    batch_size=s.batch_size, 
                    epochs=s.epochs, 
                    shuffle=True, 
                    verbose=s.VERBOSE)

    def save_model(self):
        self.model.save(s.model_name)