import itertools

import cv2
import numpy as np
import pandas as pd
# from neupy.algorithms import DiscreteBAM
from neupy.algorithms.memory.base import DiscreteMemory
from neupy.utils import format_data
from numpy.core.umath_tests import inner1d
from scipy.spatial import distance
from tqdm import tqdm

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

PATH_PREFIX_1 = r'C:\Users\Skufler\PycharmProjects\comnist\data\letters'
PATH_PREFIX_2 = r'C:\Users\Skufler\PycharmProjects\comnist\data\letters2'
PATH_PREFIX_3 = r'C:\Users\Skufler\PycharmProjects\comnist\data\letters3'

# абвгдеёжзийклмнопрстуфхцчшщъыьэюя
russian_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'


def bin2sign(matrix):
    return np.where(matrix == 0, -1, 1)


def sign2bin(matrix):
    return np.where(matrix > 0, 1, 0).astype(int)


def hopfield_energy(weight, X, y):
    return -0.5 * inner1d(X.dot(weight), y)


class DiscreteBAM(DiscreteMemory):

    def predict_input(self, y_bin):
        if self.weight is None:
            raise Exception('Network not trained')

        self.discrete_validation(y_bin)

        y_bin = format_data(y_bin, is_feature1d=False)
        y_sign = bin2sign(y_bin)
        X_sign = np.sign(y_sign.dot(self.weight.T))

        if self.mode == 'sync':
            return sign2bin(X_sign), y_bin

    def predict_output(self, X_bin):
        if self.weight is None:
            raise Exception("Network hasn't been trained yet")

        self.discrete_validation(X_bin)

        X_bin = format_data(X_bin, is_feature1d=False)
        X_sign = bin2sign(X_bin)
        y_sign = np.sign(X_sign.dot(self.weight))

        if self.mode == 'sync':
            return X_bin, sign2bin(y_sign)

    def predict(self, X_bin):
        return self.predict_output(X_bin)

    def train(self, X_bin, y_bin):
        self.discrete_validation(X_bin)
        self.discrete_validation(y_bin)

        X_sign = bin2sign(format_data(X_bin, is_feature1d=False))
        y_sign = bin2sign(format_data(y_bin, is_feature1d=False))

        _, weight_nrows = X_sign.shape
        _, weight_ncols = y_sign.shape
        weight_shape = (weight_nrows, weight_ncols)

        if self.weight is None:
            self.weight = np.zeros(weight_shape)

        if self.weight.shape != weight_shape:
            raise ValueError(
                "Invalid input shapes. Number of input "
                "features must be equal to {} and {} output "
                "features".format(weight_nrows, weight_ncols))

        self.weight += X_sign.T.dot(y_sign)

    def energy(self, X_bin, y_bin):
        self.discrete_validation(X_bin)
        self.discrete_validation(y_bin)

        X_sign, y_sign = bin2sign(X_bin), bin2sign(y_bin)
        X_sign = format_data(X_sign, is_feature1d=False)
        y_sign = format_data(y_sign, is_feature1d=False)
        nrows, n_features = X_sign.shape

        if nrows == 1:
            return hopfield_energy(self.weight, X_sign, y_sign)

        output = np.zeros(nrows)
        for i, rows in enumerate(zip(X_sign, y_sign)):
            output[i] = hopfield_energy(self.weight, *rows)

        return output


def open_image(image_path) -> np.array:
    _image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _image = cv2.resize(_image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    (thresh, im_bw) = cv2.threshold(_image, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return np.array(im_bw, dtype='int16')


def convert_to_bipolar(_image) -> np.array:
    return np.array([-1 if pixel == 0 else 1 for pixel in _image], dtype='int16')


def compute_distances(target, associations):
    distances = distance.cdist(
        target,
        np.array(
            [[1 if x == 1 else 0 for x in y] for y in associations], dtype='int16'
        ),
        "cosine"
    )[0]
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    max_similarity = 1 - min_distance

    return max_similarity, russian_alphabet[min_index]


def make_pair(input_matrix):
    return convert_to_bipolar(input_matrix)


dataset = pd.read_csv(r'C:\Users\Skufler\PycharmProjects\comnist\data\letters3.csv')
dataset.head()

images_path = dataset['file']

a_i = np.zeros(shape=(0, 1024), dtype='int16')
b_i = np.zeros(shape=(0, 1024), dtype='int16')
COUNT = 2

for index, path in tqdm(enumerate(images_path), total=len(images_path)):
    if path == '24_34.png':
        continue
    if int(path[3:6]) > 230 + COUNT:
        continue
    image = open_image(PATH_PREFIX_3 + '\\' + path)

    if path[3:6:] == '231':
        for i in range(COUNT - 1):
            b_i = np.vstack((
                b_i,
                np.array(image.ravel())
            ))
    else:
        a_i = np.vstack((
            a_i,
            image.ravel()
        ))
a_i = np.delete(a_i, 0, axis=0)
b_i = np.delete(b_i, 0, axis=0)

bam = DiscreteBAM(mode='sync')
bam.train(a_i, b_i)


dataset = pd.read_csv(r'C:\Users\Skufler\PycharmProjects\comnist\data\letters.csv')
images_path = dataset['file']

x = 0
for path in tqdm(images_path, total=len(images_path)):
    if russian_alphabet[int(path[:2]) - 1] == compute_distances(bam.predict(open_image(PATH_PREFIX_1 + '\\' + path).ravel())[0], np.unique(b_i, axis=0))[1]:
        x += 1

print(x / len(images_path))
