import cv2
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow as tf2
from keras.datasets import mnist

from sklearn.metrics import classification_report
from tqdm import tqdm

# Set random seed (for reproducibility)

a = mnist.load_data()

tf.disable_eager_execution()
tf.disable_v2_behavior()

sess = tf.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

np.random.seed(1000)
tf.compat.v1.set_random_seed(1000)

width = 32
height = 32
batch_size = 33
nb_epochs = 400
code_length = 1024
use_gpu = True

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
PATH_PREFIX_1 = r'data/letters'
PATH_PREFIX_2 = r'data/letters2'
PATH_PREFIX_3 = r'data/letters3'


def open_image(image_path) -> np.array:
    _image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    _image = cv2.resize(_image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    return np.array(_image, dtype='uint8')


dataset = pd.read_csv(r'data/letters3.csv')
dataset.head()

images_path = dataset['file']

a_i = np.zeros(shape=(0, 32, 32), dtype='uint8')
b_i = np.zeros(shape=(0, 32, 32), dtype='uint8')
COUNT = 20

for index, path in tqdm(enumerate(images_path), total=len(images_path)):
    if int(path[3:6]) > 230 + COUNT:
        continue

    image = open_image(PATH_PREFIX_3 + '/' + path)

    if path[3:6:] == '231':
        for i in range(COUNT - 1):
            b_i = np.insert(
                b_i,
                b_i.shape[0],
                np.array(image),
                axis=0
            )
    else:
        a_i = np.insert(
            a_i,
            a_i.shape[0],
            image,
            axis=0
        )

np.delete(a_i, 0, axis=0)
np.delete(b_i, 0, axis=0)
assert len(b_i) == len(a_i)

# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Select 50 samples
# X_source = X_train[0:50]
#
# X_dest = X_source.copy()
X_train = a_i
X_source = X_train[0:COUNT * 33]
X_source = X_source[:, :, :, np.newaxis]
b_i = b_i[:, :, :, np.newaxis]


def encoder(encoder_input):
    # Convolutional layer 1
    conv1 = tf.layers.conv2d(inputs=encoder_input,
                             filters=32,
                             kernel_size=(3, 3),
                             kernel_initializer=tf2.initializers.GlorotUniform(),
                             activation=tf.nn.tanh)

    # Convolutional output (flattened)
    conv_output = tf.layers.flatten(conv1)

    # Encoder Dense layer 1
    d_layer_1 = tf.layers.dense(inputs=conv_output,
                                units=1024,
                                activation=tf.nn.tanh)

    # Code layer
    code_layer = tf.layers.dense(inputs=d_layer_1,
                                 units=code_length,
                                 activation=tf.nn.tanh)

    return code_layer


def decoder(code_sequence, bs):
    # Decoder Dense layer 1
    d_layer_1 = tf.layers.dense(inputs=code_sequence,
                                units=1024,
                                activation=tf.nn.tanh)

    # Code output layer
    code_output = tf.layers.dense(inputs=d_layer_1,
                                  units=(height - 2) * (width - 2) * 3,
                                  activation=tf.nn.tanh)

    # Deconvolution input
    deconv_input = tf.reshape(code_output, (bs, height - 2, width - 2, 3))

    # Deconvolution layer 1
    deconv1 = tf.layers.conv2d_transpose(inputs=deconv_input,
                                         filters=3,
                                         kernel_size=(3, 3),
                                         kernel_initializer=tf2.initializers.GlorotUniform(),
                                         activation=tf.sigmoid)

    # Output batch
    output_batch = tf.cast(tf.reshape(deconv1, (bs, height, width, 3)) * 255.0, tf.uint8)

    return deconv1, output_batch


def create_batch(t):
    X = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    Y = np.zeros((batch_size, height, width, 3), dtype=np.float32)

    if t < X_source.shape[0] - batch_size:
        tmax = t + batch_size
    else:
        tmax = X_source.shape[0]

    for k, image in enumerate(X_source[t:tmax]):
        X[k, :, :, :] = image / 255.0

    for k, image in enumerate(b_i[t:tmax]):
        Y[k, :, :, :] = image / 255.0

    return X, Y


# Create a Tensorflow Graph
graph = tf.Graph()

with graph.as_default():
    with tf.device('/cpu:0'):
        # Global step
        global_step = tf.compat.v1.Variable(0, trainable=False)

    with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
        # Input batch
        input_images = tf.compat.v1.placeholder(tf.float32, shape=(None, height, width, 3))

        # Output batch
        output_images = tf.compat.v1.placeholder(tf.float32, shape=(None, height, width, 3))

        # Batch_size
        t_batch_size = tf.compat.v1.placeholder(tf.int32, shape=())

        # Encoder
        code_layer = encoder(encoder_input=input_images)

        # Decoder
        deconv_output, output_batch = decoder(code_sequence=code_layer,
                                              bs=t_batch_size)

        # Reconstruction L2 loss
        loss = tf.nn.l2_loss(output_images - deconv_output)

        # Training operations
        learning_rate = tf.train.exponential_decay(learning_rate=0.00025,
                                                   global_step=global_step,
                                                   decay_steps=int(X_source.shape[0] / (2 * batch_size)),
                                                   decay_rate=0.9,
                                                   staircase=True)

        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        training_step = trainer.minimize(loss)


def predict(X, bs=1):
    feed_dict = {
        input_images: X.reshape((1, height, width, 3)) / 255.0,
        output_images: np.zeros((bs, height, width, 3), dtype=np.float32),
        t_batch_size: bs
    }

    return session.run([output_batch], feed_dict=feed_dict)[0]


def story(t):
    oimages = np.zeros(shape=(20, height, width, 3), dtype=np.uint8)
    oimages[0, :, :, :] = X_source[t]

    for i in range(1, 20):
        oimages[i, :, :, :] = predict(oimages[i - 1])

    fig, ax = plt.subplots(2, 10, figsize=(18, 4))

    for i in range(2):
        for j in range(10):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            ax[i, j].imshow(oimages[(10 * i) + j])

    plt.show()


if __name__ == '__main__':
    # Create a Tensorflow Session
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(),
                                      inter_op_parallelism_threads=multiprocessing.cpu_count(),
                                      allow_soft_placement=True,
                                      device_count={'CPU': 1,
                                                    'GPU': 1 if use_gpu else 0})

    session = tf.compat.v1.InteractiveSession(graph=graph, config=config)

    # Initialize all variables
    tf.compat.v1.global_variables_initializer().run()

    # Train the model
    for e in range(nb_epochs):
        total_loss = 0.0

        for t in range(0, X_source.shape[0], batch_size):
            X, Y = create_batch(t)

            feed_dict = {
                input_images: X,
                output_images: Y,
                t_batch_size: batch_size
            }

            _, t_loss = session.run([training_step, loss], feed_dict=feed_dict)
            total_loss += t_loss

        print('Epoch {} - Loss: {}'.
              format(e + 1,
                     total_loss / float(X_train.shape[0])))

    story(0)
    story(1)
    story(5)
    story(9)
