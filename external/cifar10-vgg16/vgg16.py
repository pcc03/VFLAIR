from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib.util
import os

# Before `import tensorflow`: optional CPU-only, and cuDNN path fix for GPU.
# TF 2.12 is built against cuDNN 8.6+; many Linux images expose 8.2 in /lib first,
# which yields "DNN library is not found". PyTorch in the same conda env ships
# a compatible libcudnn (e.g. 8.7) — prepend its lib dir so the dynamic linker
# picks it up. Disable with VGG16_SKIP_TORCH_CUDNN=1.
if os.environ.get("VGG16_CPU_ONLY", "") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif os.environ.get("VGG16_SKIP_TORCH_CUDNN", "") != "1":
    try:
        spec = importlib.util.find_spec("torch")
        if spec and getattr(spec, "origin", None):
            torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
            cudnn_so = os.path.join(torch_lib, "libcudnn.so.8")
            if os.path.isfile(cudnn_so):
                ld = os.environ.get("LD_LIBRARY_PATH", "")
                parts = [p for p in ld.split(":") if p]
                if torch_lib not in parts:
                    os.environ["LD_LIBRARY_PATH"] = (
                        torch_lib + (":" + ld if ld else "")
                    )
    except Exception:
        pass

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images


def load_images():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    (train_images, test_images) = normalization(train_images, test_images)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
    #     buffer_size=10000).batch(batch_size)
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    return train_images, train_labels, test_images, test_labels


class ConvBNRelu(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1, padding='SAME', weight_decay=0.0005, rate=0.4, drop=True):
        super(ConvBNRelu, self).__init__()
        self.drop = drop
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding=padding, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dropOut = keras.layers.Dropout(rate=rate)

    def call(self, inputs, training=False):
        layer = self.conv(inputs)
        layer = tf.nn.relu(layer)
        layer = self.batchnorm(layer)
        if self.drop:
            layer = self.dropOut(layer)

        return layer


class VGG16Model(tf.keras.Model):
    def __init__(self):
        super(VGG16Model, self).__init__()
        self.conv1 = ConvBNRelu(filters=64, kernel_size=[3, 3], rate=0.3)
        self.conv2 = ConvBNRelu(filters=64, kernel_size=[3, 3], drop=False)
        self.maxPooling1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = ConvBNRelu(filters=128, kernel_size=[3, 3])
        self.conv4 = ConvBNRelu(filters=128, kernel_size=[3, 3], drop=False)
        self.maxPooling2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv5 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.conv6 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.conv7 = ConvBNRelu(filters=256, kernel_size=[3, 3], drop=False)
        self.maxPooling3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv11 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv12 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv13 = ConvBNRelu(filters=512, kernel_size=[3, 3], drop=False)
        self.maxPooling5 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv14 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv15 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv16 = ConvBNRelu(filters=512, kernel_size=[3, 3], drop=False)
        self.maxPooling6 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flat = keras.layers.Flatten()
        self.dropOut = keras.layers.Dropout(rate=0.5)
        self.dense1 = keras.layers.Dense(units=512,
                                         activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(units=10)
        self.softmax = keras.layers.Activation('softmax')

    def call(self, inputs, training=False):
        net = self.conv1(inputs)
        net = self.conv2(net)
        net = self.maxPooling1(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.maxPooling2(net)
        net = self.conv5(net)
        net = self.conv6(net)
        net = self.conv7(net)
        net = self.maxPooling3(net)
        net = self.conv11(net)
        net = self.conv12(net)
        net = self.conv13(net)
        net = self.maxPooling5(net)
        net = self.conv14(net)
        net = self.conv15(net)
        net = self.conv16(net)
        net = self.maxPooling6(net)
        net = self.dropOut(net)
        net = self.flat(net)
        net = self.dense1(net)
        net = self.batchnorm(net)
        net = self.dropOut(net)
        net = self.dense2(net)
        net = self.softmax(net)
        return net


if __name__ == '__main__':
    import atexit
    import sys
    from datetime import datetime

    class _Tee(object):
        """Copy stdout/stderr to a log file without changing TTY behavior (no per-chunk
        terminal flush — avoids breaking ``\\r``-based progress output)."""

        __slots__ = ("_s", "_f")

        def __init__(self, stream, log_file):
            object.__setattr__(self, "_s", stream)
            object.__setattr__(self, "_f", log_file)

        def write(self, data):
            if not data:
                return
            self._s.write(data)
            self._f.write(data)
            if "\n" in data:
                self._f.flush()

        def flush(self):
            self._s.flush()
            self._f.flush()

        def __getattr__(self, name):
            return getattr(self._s, name)

    _log_f = None
    if os.environ.get("VGG16_NO_LOG", "").strip().lower() not in ("1", "true", "yes"):
        raw = os.environ.get("VGG16_LOG", "").strip()
        _log_path = (
            raw
            if raw
            else os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "vgg16_run.log"
            )
        )
        _log_f = open(
            _log_path,
            "a",
            encoding="utf-8",
            errors="replace",
            buffering=1,
        )
        _log_f.write("\n===== run %s =====\n" % datetime.now().isoformat(timespec="seconds"))
        _log_f.flush()
        sys.stdout = _Tee(sys.__stdout__, _log_f)
        sys.stderr = _Tee(sys.__stderr__, _log_f)

        def _restore_streams():
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            if _log_f is not None:
                _log_f.close()

        atexit.register(_restore_streams)

    print(tf.__version__)
    print(keras.__version__)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            print("Using GPU:", gpus[0])
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    training_epochs = 250
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    lr_decay = 1e-6
    lr_drop = 20

    tf.random.set_seed(777)

    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    train_images, train_labels, test_images, test_labels = load_images()

    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(train_images)

    model = VGG16Model()

    # Keras 3 / new SGD rejects `decay`; legacy SGD matches the original script behavior.
    optimizer = tf.keras.optimizers.legacy.SGD(
        learning_rate=learning_rate,
        decay=1e-6,
        momentum=momentum,
        nesterov=True,
    )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit_generator(datagen.flow(train_images, train_labels,
                                     batch_size=batch_size), epochs=training_epochs, verbose=2, callbacks=[reduce_lr],
                        steps_per_epoch=train_images.shape[0] // batch_size, validation_data=(test_images, test_labels))

    model.save_weights('cifar10vgg_custom.h5')
