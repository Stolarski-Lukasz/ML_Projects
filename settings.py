# data settings
DATA = "data/vowel_data.csv"
X = ['log_f1_sr', 'log_f2_f1', 'log_f3_f2']
y = 'y_coordinates'
TEST_SIZE = 0.2

# evaluation settings
NFOLD = 5
VALIDATION_SPLIT = 0.1

# other settings
VERBOSE = 2

# model building settings
number_of_neurons = 8
lr = .0001
batch_size = 50
epochs = 3
model_name = "vowel_model.model"
