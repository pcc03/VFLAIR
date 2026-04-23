"""
Converted from: cifar-10-using-vgg16.ipynb
VGG16 (ImageNet weights) + custom head on CIFAR-10.

Run (from this directory, with VFLAIR env):
  export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:\${LD_LIBRARY_PATH}\"
  CUDA_VISIBLE_DEVICES=0 python cifar_10_using_vgg16.py

Optional env: OUT_DIR, EPOCHS, BS; KAGGLE=1 lists /kaggle/input when that path exists.
Default OUT_DIR is /home/newdrive2/peng326/.
Log (stdout/stderr, append): default ``cifar_10_using_vgg16_run.log`` here; override with
CIFAR_10_VGG16_LOG, or disable with CIFAR_10_VGG16_NO_LOG=1.
"""
from __future__ import annotations

import importlib.util
import os

# Match TF 2.12 + system cuDNN: see VFLAIR external/cifar10-vgg16/vgg16.py
if os.environ.get("VFL_SKIP_TORCH_CUDNN", "") != "1":
    try:
        spec = importlib.util.find_spec("torch")
        if spec and getattr(spec, "origin", None):
            torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
            if os.path.isfile(os.path.join(torch_lib, "libcudnn.so.8")):
                ld = os.environ.get("LD_LIBRARY_PATH", "")
                parts = [p for p in ld.split(":") if p]
                if torch_lib not in parts:
                    os.environ["LD_LIBRARY_PATH"] = torch_lib + (":" + ld if ld else "")
    except Exception:
        pass

import tensorflow.keras as k
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def _setup_tee_log() -> None:
    """Append stdout/stderr to a file; TTY progress lines are not broken (see vgg16.py)."""
    import atexit
    import sys
    from datetime import datetime

    class _Tee(object):
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

    if os.environ.get("CIFAR_10_VGG16_NO_LOG", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    raw = os.environ.get("CIFAR_10_VGG16_LOG", "").strip()
    log_path = (
        raw
        if raw
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cifar_10_using_vgg16_run.log",
        )
    )
    log_f = open(
        log_path,
        "a",
        encoding="utf-8",
        errors="replace",
        buffering=1,
    )
    log_f.write(
        "\n===== run %s =====\n" % datetime.now().isoformat(timespec="seconds")
    )
    log_f.flush()
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)

    def _restore() -> None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_f.close()

    atexit.register(_restore)


def _maybe_list_kaggle_input() -> None:
    if os.environ.get("KAGGLE", "").strip() in ("1", "true", "yes") and os.path.isdir(
        "/kaggle/input"
    ):
        for dirname, _, filenames in os.walk("/kaggle/input"):
            for filename in filenames:
                print(os.path.join(dirname, filename))


def main() -> None:
    _setup_tee_log()
    _maybe_list_kaggle_input()

    out_dir = os.environ.get("OUT_DIR", "/home/newdrive2/peng326/kaggle-cifar10-vgg16/")
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "weights.h5")

    # VGG16 backbone + head (notebook: transfer learning on 32x32)
    vgg16_model = VGG16(
        weights="imagenet",
        include_top=False,
        classes=10,
        input_shape=(32, 32, 3),
    )

    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    model.add(Flatten())
    model.add(Dense(512, activation="relu", name="hidden1"))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation="relu", name="hidden2"))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation="softmax", name="predictions"))

    model.summary()

    (X_train, y_train), (X_test, y_test) = k.datasets.cifar10.load_data()

    print("******************")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    y_train_ohe = to_categorical(y_train, num_classes=10)
    y_test_ohe = to_categorical(y_test, num_classes=10)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255.0
    X_test /= 255.0

    print("******************")
    print(X_train.shape, y_train_ohe.shape, X_test.shape, y_test_ohe.shape)

    X_val = X_train[40000:]
    y_val = y_train_ohe[40000:]
    X_train = X_train[:40000]
    y_train_ohe = y_train_ohe[:40000]
    print(X_val.shape, y_val.shape)
    print(X_train.shape, y_train_ohe.shape)

    sgd = optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=sgd,
        metrics=["accuracy"],
    )

    def lr_scheduler(epoch: int) -> float:
        return 0.001 * (0.5 ** (epoch // 20))

    reduce_lr = LearningRateScheduler(lr_scheduler)
    mc = ModelCheckpoint(
        weights_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )

    epochs = int(os.environ.get("EPOCHS", "100"))
    bs = int(os.environ.get("BS", "128"))

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    model.fit(
        aug.flow(X_train, y_train_ohe, batch_size=bs),
        validation_data=(X_val, y_val),
        steps_per_epoch=len(X_train) // bs,
        epochs=epochs,
        callbacks=[reduce_lr, mc],
        verbose=1,
    )

    model.load_weights(weights_path)

    train_steps = len(X_train) // bs
    train_loss, train_accuracy = model.evaluate(
        aug.flow(X_train, y_train_ohe, batch_size=bs),
        steps=train_steps,
        verbose=0,
    )
    print("Training loss: {}\nTraining accuracy: {}".format(train_loss, train_accuracy))

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print("Validation loss: {}\nValidation accuracy: {}".format(val_loss, val_accuracy))

    test_loss, test_accuracy = model.evaluate(X_test, y_test_ohe, verbose=0)
    print("Testing loss: {}\nTesting accuracy: {}".format(test_loss, test_accuracy))


if __name__ == "__main__":
    main()
