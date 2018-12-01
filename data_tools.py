import pickle
import pandas as pd
import numpy as np
import pylab as plt
import gzip
from struct import unpack


class DataPreparation:

    def __init__(self, filepath, x_columns):
        self.filepath = filepath  # filepath leading to the csv
        self.x_columns = x_columns  # in the csv file, it's the number of columns which make up the x_input
        self.input_data = DataPreparation.training_data_creation(self)
        self.input_x, self.input_y = DataPreparation.training_xy(self)

    def training_xy(self):
        input_x, input_y = [], []
        for x, y in self.input_data:
            input_x.append(x)
            input_y.append(y)
        return input_x, input_y

    def training_data_creation(self):

        # Load dataset
        dataset = pd.read_csv(self.filepath)
        dataset = dataset.values.tolist()

        # Split the columns of the file into X and Y data
        x, y = [idx[:self.x_columns] for idx in dataset], [idy[self.x_columns:] for idy in dataset]

        # Necessary manipulations to make it readable by the "Network" class
        x, y = [[[idx] for idx in idy] for idy in x], [[[idw] for idw in idy] for idy in y]
        x, y = np.array(x), np.array(y)

        # Creation of the training data
        return [(x, y) for x, y in zip(x, y)]


class MNIST:

    def __init__(self, imagefile, labelfile, pickle_filepath):

        self.imagefile = imagefile
        self.labelfile = labelfile
        self.pickle_filepath = pickle_filepath

    # Code obtained from: https://martin-thoma.com/classify-mnist-with-pybrain/
    def get_labeled_data(self):
        """Read input-vector (image) and target class (label, 0-9) and return
           it as list of tuples.
        """
        # Open the images with gzip in read binary mode
        images = gzip.open(self.imagefile, 'rb')
        labels = gzip.open(self.labelfile, 'rb')

        # Read the binary data

        # We have to get big endian unsigned int. So we need '>I'

        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]
        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        cols = images.read(4)
        cols = unpack('>I', cols)[0]

        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        n = labels.read(4)
        n = unpack('>I', n)[0]

        if number_of_images != n:
            raise Exception('Number of labels did not match the number of images')

        # Get the data
        image_array_bundle = np.zeros((n, rows, cols), dtype=np.float32)  # Initialize numpy array for all images
        image_label_bundle = np.zeros((n, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(n):
            if i % 5000 == 0:
                print('Iteration: {} out of {}'.format(i, n))
            for row in range(rows):
                for col in range(cols):
                    tmp_pixel = images.read(1)  # Just a single byte
                    tmp_pixel = unpack('>B', tmp_pixel)[0]
                    image_array_bundle[i][row][col] = tmp_pixel
            tmp_label = labels.read(1)
            image_label_bundle[i] = unpack('>B', tmp_label)[0]
        return image_array_bundle, image_label_bundle

    def training_set_creation(self):
        image_array_bundle, image_label_bundle = self.get_labeled_data()
        x, y, z = np.shape(image_array_bundle)  # determine the shape of image_array_bundle
        training_set = []  # each item of this list will represent a training point: (image array, associated label)
        for idx, image in enumerate(image_array_bundle):
            image_label_array = self.onehot_encoder(image_label_bundle[idx][0])
            training_set.append((np.reshape(image, (y*z, 1))/255.0, image_label_array))
        return training_set

    @staticmethod
    def onehot_encoder(input_value):
        array = np.zeros((10, 1))
        array[input_value] = 1
        return array

    @staticmethod
    def onehot_decoder(onehot):
        decoded_value = np.where(onehot == 1)
        return decoded_value

    @staticmethod
    def pickle_data(dataset, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        return None

    @staticmethod
    def unpickle_data(filepath):
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    @staticmethod
    def image2vector(image_matrix):  # image_array_bundle is the complete set of images (60000, 28, 28)
        x, y = np.shape(image_matrix)  # determine the shape of image_array_bundle
        vector = np.reshape(image_matrix, (x*y, 1))
        return vector

    @staticmethod
    def vector2image(vector, output_shape=(28, 28)):  # image_array_bundle is the complete set of images (60000, 28, 28)
        image = np.reshape(vector, output_shape)  # determine the shape of image_array_bundle
        return image

    @staticmethod
    def view_image(image_matrix, label=''):
        """View a single image"""
        print('Label: {}'.format(label))
        plt.imshow(image_matrix, cmap='gray')
        plt.title('Label: {}'.format(label))
        plt.show()
        return None
