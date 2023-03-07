# data settings
DATA = "data/cardinal_vowels_data.csv"
MODEL_TYPE = "classification"
X = ['log_f1_sr', 'log_f2_f1', 'log_f3_f2']
y = 'vowel'
TEST_SIZE = 0.2

# evaluation settings
# for "kfold validation"
N_FOLD = 5
# for "single fold validation"
VALIDATION_SPLIT = 0.1

# other settings
VERBOSE = 2

# model building settings
N_NEURONS = 8
LR = .0001
BATCH_SIZE = 50
N_EPOCHS = 10
MODEL_NAME = "vowel_model.model"
