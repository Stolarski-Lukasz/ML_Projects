from ml_module import DataProcessor, DeepModelProcessor, DeepModelEvaluator
import settings as s

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam


# In the code below, comment out the lines you currently do not need
if __name__ == "__main__":
    # data processing
    #################
    data_processor = DataProcessor(s.DATA)
    data_processor.prepare_data(model_type=s.MODEL_TYPE)

    # model building
    ################
    model = Sequential()
    model.add(Dense(s.N_NEURONS, input_dim=data_processor.n_features, activation='relu'))
    model.add(Dense(s.N_NEURONS, activation='relu'))
    model.add(Dense(data_processor.n_label_levels, activation='softmax'))
    model.compile(Adam(lr=s.LR), loss='categorical_crossentropy', metrics=['accuracy'])

    # model evaluation
    ##################
    model_evaluator = DeepModelEvaluator(data_processor, model)
    # model_evaluator.validate_model_single_fold(is_train_test_split=False)
    model_evaluator.validate_model_k_fold()
    # model_evaluator.test_model()

    # final model training (on all data)
    ######################
    # model_processor = DeepModelProcessor(data_processor, model)
    # model_processor.train_model_final()
    # model_processor.save_model()
