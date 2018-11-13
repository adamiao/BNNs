"""
Under construction!
"""

from network import Network, DataPreparation
import numpy as np

# # importing and manipulating dataset of interest
dataset = DataPreparation('iris_onehot.csv', 4)
training_data = dataset.training_data
training_x, training_y = dataset.training_x, dataset.training_y

x_shape = (len(training_data), len(training_data[0][0]))
y_shape = (len(training_data), len(training_data[0][1]))

# # boosting parameters
number_of_networks = 2
gammas = [1.0]
sizes = [4, 10, 3]
epochs = 1001
mini_batch_size = 10

# # initialization of variables and creation of multiple instances of "Network" object
training_error = training_data.copy()
prediction_vector, error_vector = np.zeros(y_shape), np.zeros(y_shape)
neural_bundle = []

# # boosting procedure
for _ in range(number_of_networks):
    nn = Network(sizes)
    nn.sgd(training_error, epochs, mini_batch_size, eta=0.01, reg_eta=0.1)
    neural_bundle.append(nn)

    # # calculate vector of errors
    training_error = []
    for idx, (x, y) in enumerate(training_data):
        prediction_vector = Network.point_predictor(neural_bundle, gammas, x)
        error_vector = y - prediction_vector
        training_error.append((x, y - prediction_vector))

    # # calculate gamma
    gamma = np.dot(np.transpose(error_vector), prediction_vector) / np.dot(np.transpose(prediction_vector),
                                                                           prediction_vector)
    gammas.append(gamma[0][0])
    # gammas.append(1.0)

########################################################################################################################

# x_test = np.array([[0.11], [0.51], [0.05], [0.04]])  # output should be 0
# x_test = np.array([[0.94], [0.26], [0.98], [0.92]])  # output should be 1
x_test = np.array([[0.39], [0.33], [0.59], [0.51]])  # output should be 2
print(Network.point_predictor(neural_bundle, gammas, x_test))
for idx in range(number_of_networks):
    print(neural_bundle[idx].feedforward(x_test))
print()
print('gammas: ', gammas)
