from network import Network
from data_tools import DataPreparation
import numpy as np
from sklearn.model_selection import train_test_split

# # DATA IMPORT AND BASIC MANIPULATIONS

# Importing dataset of interest
dataset = DataPreparation('iris_onehot.csv', 4)
# Get data in the format: [ (x_1, y_1), (x_2, y_2), .... , (x_N, y_N) ]
input_data = dataset.input_data
# Training / Testing split of the dataset
training_data, test_data = train_test_split(input_data, test_size=0.2, random_state=42)
# Create a constant equal to the shape of the output
y_shape = (len(training_data), len(training_data[0][1]))

# # PARAMETERS AND VARIABLES

number_of_networks = 3
gammas = [1.0]  # these are the coefficients for the weak learners
sizes = [4, 2, 3]  # architecture of the neural net: input - hidden layer 1 - ... - output
eta = 0.01  # learning rate
reg_eta = 0.1  # regularization parameter
epochs = 201
mini_batch_size = 25

# Initialization of variables
prediction_vector, error_vector = np.zeros(y_shape), np.zeros(y_shape)
neural_bundle = []

# # BOOSTING PROCEDURE

# Algorithm 1: Initialization
print('Training Network 1 out of {}'.format(number_of_networks))
print()
nn = Network(sizes)  # create instance of a neural network object
nn.sgd(training_data, epochs, mini_batch_size, eta=eta, reg_eta=reg_eta, verbose=True)  # fit nn w/ original dataset
neural_bundle.append(nn)  # append new trained weak learner to list

training_error = []  # create new list of training errors tuple for every loop (for each new weak learner)
# gamma_numerator, gamma_denominator = 0.0, 0.0
for idx, (x, y) in enumerate(training_data):
    prediction_vector = Network.point_predictor(neural_bundle, gammas, x)  # gammas = [1.0] for this initialization
    error_vector = y - prediction_vector  # calculate error vector
    training_error.append((x, y - prediction_vector))  # create new training error vector
    # gamma_numerator += np.dot(np.transpose(error_vector), prediction_vector)[0][0]
    # gamma_denominator += np.dot(np.transpose(prediction_vector), prediction_vector)[0][0]
# gamma = gamma_numerator / gamma_denominator
# gammas.append(gamma)

# Algorithm 2: We loop once for each neural network belonging to the weak learners
for iteration in range(number_of_networks-1):
    print()
    print('Training Network {} out of {}'.format(iteration + 2, number_of_networks))
    print()
    nn = Network(sizes)  # create instance of a neural network object
    nn.sgd(training_error, epochs, mini_batch_size, eta=eta, reg_eta=reg_eta, verbose=True)  # fit nn w/ training errors
    neural_bundle.append(nn)

    # Calculate error vectors
    training_error_copy = training_error.copy()
    training_error = []  # create new list of training errors tuple for every loop (for each new weak learner)
    gamma_numerator, gamma_denominator = 0.0, 0.0
    for idx, (x, y) in enumerate(training_error_copy):
        prediction_vector = Network.point_predictor(neural_bundle, gammas + [1.0], x)
        error_vector = y - prediction_vector
        training_error.append((x, y - prediction_vector))  # create new training error vector
        gamma_numerator += np.dot(np.transpose(error_vector), prediction_vector)[0][0]
        gamma_denominator += np.dot(np.transpose(prediction_vector), prediction_vector)[0][0]

    # Calculate weak learner coefficient
    gamma = gamma_numerator / gamma_denominator
    gammas.append(gamma)

########################################################################################################################

# # TESTING SOME RANDOM SAMPLES
# x_test = np.array([[0.11], [0.51], [0.05], [0.04]])  # output should be 0
# x_test = np.array([[0.94], [0.26], [0.98], [0.92]])  # output should be 1
x_test = np.array([[0.39], [0.33], [0.59], [0.51]])  # output should be 2
print()
print(Network.point_predictor(neural_bundle, gammas, x_test))
print()
print(np.argmax(Network.point_predictor(neural_bundle, gammas, x_test)))
