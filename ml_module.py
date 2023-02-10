import pandas as pd
import numpy as np
from abc import ABC, abstractclassmethod

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

import settings as s



# Data processing
#################
class IDataProcessor(ABC):

    def __init__(self):
        self.X = None
        self.y = None
        self.number_of_features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    @abstractclassmethod
    def process_data(self, data):
        pass

class DataProcessor(IDataProcessor):

    def __init__(self):
        super().__init__()

    def process_data(self, data):
        data = pd.read_csv(data)
        # seed specification could be added here...
        data = shuffle(data)
        self.number_of_features = len(s.X)
        self.X = data[s.X]
        self.y = data[s.y]
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
                                    batch_size=s.batch_size, 
                                    epochs=s.epochs, 
                                    shuffle=True, 
                                    verbose=s.VERBOSE,
                                    validation_split=s.VALIDATION_SPLIT)
        else:
            print(f"Starting training the model with the last {int(s.VALIDATION_SPLIT * 100)}% of all the data as the validation set.")
            self.model.fit(self.data_processor.X, 
                                    self.data_processor.y, 
                                    batch_size=s.batch_size, 
                                    epochs=s.epochs, 
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
        Y = Y.to_numpy()
        kfold = KFold(n_splits=5, shuffle=True, random_state=7)
        cvscores = []
        print(f"Starting {s.NFOLD}-fold cross-validation for a model built on {info_text}:")
        fold_counter = 1
        for train, test in kfold.split(X, Y):
            self.model.fit(X[train], 
                                    Y[train], 
                                    batch_size=s.batch_size, 
                                    epochs=s.epochs, 
                                    shuffle=True, 
                                    verbose=s.VERBOSE)

            score = self.model.evaluate(X[test], Y[test], verbose=0)
            print(f"The validation mean absolute error for fold {fold_counter} is {score[1]}.")
            fold_counter += 1
            cvscores.append(score[1])
        print(f"The average validation mean absolute error for all {s.NFOLD} folds is {np.mean(cvscores)}.")


    def test_model(self):
        self.model.fit(self.data_processor.X_train, 
                                    self.data_processor.y_train, 
                                    batch_size=s.batch_size, 
                                    epochs=s.epochs, 
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