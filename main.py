import cv2
import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf2
import tensorflow.compat.v1 as tf

from tqdm import tqdm
from scipy.spatial import distance

tf.disable_eager_execution()
tf.disable_v2_behavior()

np.random.seed(1000)
tf.compat.v1.set_random_seed(1000)

width = 32
height = 32
EPOCHS_COUNT = 300
vector_length = 1024
use_gpu = True

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
PATH_PREFIX_1 = r'data/letters'
PATH_PREFIX_2 = r'data/letters2'
PATH_PREFIX_3 = r'data/letters3'
BATCH_SIZE = 33
IMAGES_COUNT = 5

russian_alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'


def open_image(image_path) -> np.array:
    _image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    _image = cv2.resize(_image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    return np.array(_image, dtype='uint8')


def load_data():
    dataset = pd.read_csv(r'data/letters3.csv')
    dataset.head()

    images_path = dataset['file']

    _a_i = np.zeros(shape=(0, 32, 32), dtype='uint8')
    _b_i = np.zeros(shape=(0, 32, 32), dtype='uint8')

    for index, path in tqdm(enumerate(images_path), total=len(images_path)):
        if int(path[3:6]) > 230 + IMAGES_COUNT:
            continue

        image = open_image(PATH_PREFIX_3 + '/' + path)

        if path[3:6:] == '231':
            for count in range(IMAGES_COUNT - 1):
                _b_i = np.insert(
                    _b_i,
                    _b_i.shape[0],
                    np.array(image),
                    axis=0
                )
        else:
            _a_i = np.insert(
                _a_i,
                _a_i.shape[0],
                image,
                axis=0
            )

    np.delete(_a_i, 0, axis=0)
    np.delete(_b_i, 0, axis=0)
    assert len(_b_i) == len(_a_i)

    return _a_i, _b_i


def encoder(encoder_input):
    conv = tf.layers.conv2d(
        inputs=encoder_input,
        filters=32,
        kernel_size=(3, 3),
        kernel_initializer=tf2.initializers.GlorotUniform(),
        activation=tf.nn.tanh
    )

    conv_output = tf.layers.flatten(conv)

    dense = tf.layers.dense(
        inputs=conv_output,
        units=1024,
        activation=tf.nn.tanh
    )

    inputs = tf.layers.dense(
        inputs=dense,
        units=vector_length,
        activation=tf.nn.tanh
    )

    return inputs


def decoder(code_sequence, batch_size):
    dense = tf.layers.dense(
        inputs=code_sequence,
        units=1024,
        activation=tf.nn.tanh
    )

    output = tf.layers.dense(
        inputs=dense,
        units=(height - 2) * (width - 2) * 3,
        activation=tf.nn.tanh
    )

    deconv_input = tf.reshape(
        output,
        (batch_size, height - 2, width - 2, 3)
    )

    deconv1 = tf.layers.conv2d_transpose(
        inputs=deconv_input,
        filters=3,
        kernel_size=(3, 3),
        kernel_initializer=tf2.initializers.GlorotUniform(),
        activation=tf.sigmoid
    )

    output = tf.cast(tf.reshape(deconv1, (batch_size, height, width, 3)) * 255.0, tf.uint8)

    return deconv1, output


a_i, b_i = load_data()

X_source = a_i[0:IMAGES_COUNT * 33]
X_source = X_source[:, :, :, np.newaxis]
b_i = b_i[:, :, :, np.newaxis]


def create_batch(index):
    x = np.zeros((BATCH_SIZE, height, width, 3), dtype=np.float32)
    y = np.zeros((BATCH_SIZE, height, width, 3), dtype=np.float32)

    if index < X_source.shape[0] - BATCH_SIZE:
        _batch_size = index + BATCH_SIZE
    else:
        _batch_size = X_source.shape[0]

    for k, image in enumerate(X_source[index:_batch_size]):
        x[k, :, :, :] = image / 255.0

    for k, image in enumerate(b_i[index:_batch_size]):
        y[k, :, :, :] = image / 255.0

    return x, y


def predict(X, batch_size=1):
    feed = {
        input_images: X.reshape((1, height, width, 3)) / 255.0,
        output_images: np.zeros((batch_size, height, width, 3), dtype=np.float32),
        t_batch_size: batch_size
    }

    return session.run([output_batch], feed_dict=feed)[0]


def compute_distances(target, associations):
    distances = distance.cdist(
        [target],
        np.array(
            [[1 if x == 255 else 0 for x in y] for y in associations], dtype='uint8'
        ),
        'cosine'
    )[0]
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    max_similarity = 1 - min_distance

    return max_similarity, russian_alphabet[min_index]


graph = tf.Graph()

with graph.as_default():
    with tf.device('/cpu:0'):
        global_step = tf.compat.v1.Variable(0, trainable=False)

    with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
        input_images = tf.compat.v1.placeholder(tf.float32, shape=(None, height, width, 3))
        output_images = tf.compat.v1.placeholder(tf.float32, shape=(None, height, width, 3))

        t_batch_size = tf.compat.v1.placeholder(tf.int32, shape=())

        code_layer = encoder(encoder_input=input_images)
        deconv_output, output_batch = decoder(
            code_sequence=code_layer,
            batch_size=t_batch_size
        )

        loss = tf.nn.l2_loss(output_images - deconv_output)

        learning_rate = tf.train.exponential_decay(
            learning_rate=0.00025,
            global_step=global_step,
            decay_steps=int(X_source.shape[0] / (2 * BATCH_SIZE)),
            decay_rate=0.9,
            staircase=True
        )

        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        training_step = trainer.minimize(loss)

if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=multiprocessing.cpu_count(),
        inter_op_parallelism_threads=multiprocessing.cpu_count(),
        log_device_placement=True,
        allow_soft_placement=True,
        device_count={
            'CPU': 1,
            'GPU': 1 if use_gpu else 0
        }
    )

    session = tf.compat.v1.InteractiveSession(graph=graph, config=config)
    tf.compat.v1.global_variables_initializer().run()

    for i in range(EPOCHS_COUNT):
        total_loss = 0.0

        for j in range(0, X_source.shape[0], BATCH_SIZE):
            X, Y = create_batch(j)

            feed_dict = {
                input_images: X,
                output_images: Y,
                t_batch_size: BATCH_SIZE
            }

            _, t_loss = session.run([training_step, loss], feed_dict=feed_dict)
            total_loss += t_loss

        print('Epoch: {} -> Loss: {}'.format(
            i + 1, total_loss / float(a_i.shape[0]))
        )

    cnt = 0
    for i in range(0, (IMAGES_COUNT - 1) * 33, IMAGES_COUNT - 1):
        restored_images = np.zeros(shape=(2, height, width, 3), dtype=np.uint8)
        restored_images[0, :, :, :] = X_source[i]

        predicted = predict(restored_images[0])[0]

        cv2.imwrite('output/{}.jpg'.format(cnt), predicted)
        # print(compute_distances(
        #     predicted.ravel(), np.unique([x.ravel() for x in b_i], axis=0))
        # )
