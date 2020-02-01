import cv2
import glob
from multiprocessing import Pool
from multiprocessing import cpu_count

import numpy as np
from scipy.spatial import distance
from PIL import Image
from tqdm import tqdm

IMAGE_WIDTH = 278
IMAGE_HEIGHT = 278

#  АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ
russian_alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'


class BAM:
    def __init__(self, train_df):
        self.pairs = train_df

        self.len_x = len(self.pairs[0][1])
        self.len_y = len(self.pairs[0][0])

        assert self.len_x > 0
        assert self.len_y > 0

        self.kernel = np.zeros(shape=(self.len_y, self.len_x), dtype='uint8')
        self.make_association_matrix()

    def make_association_matrix(self):
        for vector in tqdm(self.pairs, desc='Building kernel'):
            x = np.array(vector[0])
            y = np.array(vector[1])

            self.kernel = np.add(self.kernel, np.dot(x.T[:, None], y[None]))
        print('MAX: ', np.max(self.kernel))

    def solve(self, input_vector):
        input_vector = input_vector.dot(self.kernel)
        return self.compute_threshold(input_vector)

    def get_kernel(self):
        return self.kernel

    @staticmethod
    def compute_threshold(vector):
        return list([0 if x < 0 else 1 for x in vector])

    @staticmethod
    def convert_to_bipolar(_image):
        return list([-1 if x == 0 else 1 for x in _image])


def fn(input_matrix):
    pairs = [
        BAM.convert_to_bipolar(input_matrix[0]),
        BAM.convert_to_bipolar(input_matrix[1])
    ]
    return pairs


def open_image(image_path):
    # load image with alpha channel.  use IMREAD_UNCHANGED to ensure loading of alpha channel
    _image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # make mask of where the transparent bits are
    trans_mask = _image[:, :, 3] == 0

    # replace areas of transparency with white and not transparent
    _image[trans_mask] = [255, 255, 255, 255]

    # new image without alpha channel...
    new_img = cv2.cvtColor(_image, cv2.COLOR_BGRA2BGR)
    new_img = cv2.resize(new_img, (16, 16))
    # cv2.imshow('image', th3)
    return np.array(new_img, dtype='uint8')


def compute_distances(target, associations):
    distances = distance.cdist(
        [target],
        np.array(
            [[1 if x == 255 else 0 for x in y] for y in associations], dtype='uint8'
        ),
        "cosine"
    )[0]
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    max_similarity = 1 - min_distance

    return max_similarity

# def solve(bam, path, ):
#     file = open_image(path)
#     _image = bam.solve(image.ravel())


if __name__ == "__main__":
    # data = np.empty((1, 154568), int)
    data = []
    np.show_config()

    Bi = []
    count = 0
    # test_image_1 = open_image('/home/skufler/comnist/images/Cyrillic/Cyrillic/А/2.png')
    # test_image_2 = open_image('/home/skufler/comnist/images/Cyrillic/Cyrillic/А/3.png')
    # test_image_3 = open_image('/home/skufler/comnist/images/Cyrillic/Cyrillic/А/4.png')
    #
    # b_image = open_image('/home/skufler/comnist/images/Cyrillic/Cyrillic/А/1.png')

    for number, letter in tqdm(enumerate(russian_alphabet), desc='Loading files'):
        for name, path in enumerate(
                sorted(glob.glob('/home/skufler/comnist/images/Cyrillic/Cyrillic/{}/*.png'.format(letter)))):
            if name < 200:
                if path[-6::] == '/1.png':
                    image = open_image(path)
                    Bi.append(image.ravel())
                    count += 1
                image = open_image(path)
                data.append(
                    fn([image.ravel(), Bi[count - 1]])
                )
            else:
                break

    # data.append(fn([test_image_1.ravel(), b_image.ravel()]))
    # data.append(fn([test_image_2.ravel(), b_image.ravel()]))
    # data.append(fn([test_image_3.ravel(), b_image.ravel()]))

    # with Pool(cpu_count()) as pool:
    #     for i in tqdm(pool.imap_unordered(fn, data), total=len(data), desc='Making vector pairs'):
    #         with open('/home/skufler/PycharmProjects/bam/vectors.txt', 'w') as file:
    #             file.write("{}\n".format(i))

    # del Bi
    bam = BAM(data)

    print('Kernel: ')
    print(bam.get_kernel())
    print('\n')

    bam.get_kernel().tofile('kernel.txt')

    tests = []

    for number, letter in tqdm(enumerate(russian_alphabet), total=33, desc='Testing'):
        # files = len(glob.glob('/home/skufler/comnist/images/Cyrillic/Cyrillic/{}/*'.format(letter)))
        x = 0
        for i in range(200, 300):
            try:
                image = open_image('/home/skufler/comnist/images/Cyrillic/Cyrillic/{}/{}.png'.format(letter, i))
                sln = bam.solve(image.ravel())
                x += compute_distances(sln, Bi)
            except TypeError:
                print(i)
        print('Letter {}, score {}'.format(letter, x / 100))

    # print(tests)
    solution = np.array(
        bam.solve(
            open_image('/home/skufler/comnist/images/Cyrillic/Cyrillic/А/5.png').ravel()
        ),
        dtype='uint8'
    )
    # solution = solution.reshape((16, 16, 3))
    # solution = np.logical_not(solution).astype('uint8')
    #
    # for x in solution:
    #     for y in x:
    #         print(y[0], end='')
    #     print('')
