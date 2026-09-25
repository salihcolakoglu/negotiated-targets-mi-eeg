"""Model, seed and Walsh routines from the original project and REPRO_v1.0.

Original project: Nuri Korhan and coauthors. Independent reproduction: Claude
(Anthropic, Cowork). Portable packaging: GPT-6 Astra / Codex, 2026-09-25.
TensorFlow is loaded when a training function is called, not during CLI help.
"""
import math
import os
import random
import numpy as np

try:
    from .config import RANDOM_SEED, SIZE_OF_WALSH, KERNEL_SIZE
except ImportError:
    from config import RANDOM_SEED, SIZE_OF_WALSH, KERNEL_SIZE


def set_seeds(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_walsh(size_of_walsh=SIZE_OF_WALSH):
    n = int(math.log(size_of_walsh, 2) + 1)
    j = 1
    W_old = 1
    for i in range(n):
        j_new = 2 ** i
        W_new = np.zeros([j_new, j_new], dtype=np.float32)
        W_new[:j, :j] = W_old
        W_new[j:, :j] = W_old
        W_new[:j, j:] = W_old
        W_new[j:, j:] = abs(1 - W_old)
        W_old = W_new
        j = j_new
    return W_new


def build_cnn_model_original(input_shape, dropout_rate=0.3,
                             size_of_walsh=SIZE_OF_WALSH, kernel_size=KERNEL_SIZE):
    """Preserve the exact separate-BatchNorm architecture used in Experiment 1."""
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, SpatialDropout1D, BatchNormalization
    input_len = input_shape[0]
    ks = min(kernel_size, input_len)
    pad = 'same' if input_len < 13 else 'valid'
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv1D(128, kernel_size=ks, strides=1,
                     data_format='channels_last', padding=pad, activation='relu'))
    model.add(SpatialDropout1D(dropout_rate))
    model.add(BatchNormalization())
    model.add(Conv1D(96, kernel_size=ks, padding=pad, activation='relu'))
    model.add(SpatialDropout1D(dropout_rate))
    model.add(BatchNormalization())
    model.add(Conv1D(64, kernel_size=ks, padding=pad, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(96, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(size_of_walsh, activation='sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
    return model
