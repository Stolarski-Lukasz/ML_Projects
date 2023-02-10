from ml_module import DataProcessor, DeepModelProcessor, DeepModelEvaluator
import settings as s

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam


# In the code below, comment out the lines that you currently do not need
if __name__ == "__main__":
    # data processing
    #################
    data_processor = DataProcessor()
    data_processor.process_data(s.DATA)

    # model building
    ################
    model = Sequential()
    model.add(Dense(s.number_of_neurons, input_dim=data_processor.number_of_features, activation='relu'))
    model.add(Dense(s.number_of_neurons, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(Adam(lr=s.lr), loss='mean_squared_error', metrics=['mae'])

    # model evaluation
    ##################
    model_evaluator = DeepModelEvaluator(data_processor, model)
    # model_evaluator.validate_model_single_fold()
    model_evaluator.validate_model_k_fold()
    # model_evaluator.test_model()

    # final model training (on all data)
    ######################
    # model_processor = DeepModelProcessor(data_processor, model)
    # model_processor.train_model_final()
    # model_processor.save_model()