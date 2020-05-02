import cv2

import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

PATH_PREFIX_1 = r'C:\Users\Skufler\PycharmProjects\comnist\data\letters'
PATH_PREFIX_2 = r'C:\Users\Skufler\PycharmProjects\comnist\data\letters2'
PATH_PREFIX_3 = r'C:\Users\Skufler\PycharmProjects\comnist\data\letters3'

# абвгдеёжзийклмнопрстуфхцчшщъыьэюя
russian_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'


class BAM:
    def __init__(self, train_df):
        self.pairs = train_df

        self.len_x = 1024
        self.len_y = 1024

        assert self.len_x > 0
        assert self.len_y > 0

        self.kernel = np.zeros(shape=(self.len_y, self.len_x), dtype='int16')
        self.make_association_matrix()

    def make_association_matrix(self):
        for vector in tqdm(self.pairs, desc='Building kernel', total=len(self.pairs)):
            x = vector[0]
            y = vector[1]

            self.kernel = np.add(self.kernel, np.dot(x.T[:, None], y[None]))
        print('MAX: ', np.max(self.kernel))

    def solve(self, input_vector):
        input_vector = input_vector.ravel()
        result = input_vector.dot(self.kernel.T)
        # input_vector = input_vector.dot(self.kernel)
        # result = self.compute_threshold(input_vector)
        return result

    def get_kernel(self):
        return self.kernel

    @staticmethod
    def compute_threshold(vector) -> np.array:
        return np.array([0 if number < 0 else 1 for number in vector], dtype='int16')

    @staticmethod
    def convert_to_bipolar(_image) -> np.array:
        return np.array([-1 if pixel == 0 else 1 for pixel in _image], dtype='int16')


print(np.__config__.show())


def compute_distances(target, associations):
    distances = distance.cdist(
        [target],
        np.array(
            [[1 if x == 255 else 0 for x in y] for y in associations], dtype='int16'
        ),
        "cosine"
    )[0]
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    max_similarity = 1 - min_distance

    return max_similarity, russian_alphabet[min_index]


def open_image(image_path) -> np.array:
    _image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _image = cv2.resize(_image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    (thresh, im_bw) = cv2.threshold(_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return np.array(im_bw, dtype='int16')


def make_pair(input_matrix):
    return (
        BAM.convert_to_bipolar(input_matrix[0]),
        BAM.convert_to_bipolar(input_matrix[1])
    )


dataset = pd.read_csv(r'C:\Users\Skufler\PycharmProjects\comnist\data\letters3.csv')
dataset.head()

images_path = dataset['file']

a_i = np.zeros(shape=(0, 2, 1024), dtype='int16')
b_i = np.zeros(shape=(0, 1024), dtype='int16')

for index, path in tqdm(enumerate(images_path), total=len(images_path)):
    if path == '24_34.png':
        continue
    if int(path[3:6]) > 230 + 2:
        continue
    image = open_image(PATH_PREFIX_3 + '\\' + path)

    if path[3:6:] == '231':
        b_i = np.vstack((
            b_i,
            np.array(image.ravel())
        ))
    else:
        a_i = np.insert(
            a_i,
            a_i.shape[0],
            values=make_pair([image.ravel(), b_i[int(path[:2:]) - 1]]),
            axis=0
        )
a_i = np.delete(a_i, 0, axis=0)
b_i = np.delete(b_i, 0, axis=0)

bam = BAM(a_i)

print(compute_distances(bam.solve(open_image(PATH_PREFIX_1 + '\\' + '24_34.png')), b_i))
print(russian_alphabet[25])
