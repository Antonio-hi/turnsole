# -*- coding: utf-8 -*-
# @Author        : Lyu Kui
# @Email         : 9428.al@gmail.com
# @Created Date  : 2021-02-24 13:58:46
# @Last Modified : 2021-03-05 18:14:17
# @Description   :

import tensorflow as tf

from .nets.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from .nets.efficientnet import EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

def load_backbone(phi, input_tensor, weights='imagenet'):
    if phi == 0:
        model = EfficientNetB0(include_top=False,
                               weights=weights,
                               input_tensor=input_tensor)
        # 从这些层提取特征
        layer_names = [
            'block2b_add',   # 1/4
            'block3b_add',   # 1/8
            'block5c_add',   # 1/16
            'block7a_project_bn',  # 1/32
        ]
    elif phi == 1:
        model = EfficientNetB1(include_top=False,
                               weights=weights,
                               input_tensor=input_tensor)
        layer_names = [
            'block2c_add',   # 1/4
            'block3c_add',   # 1/8
            'block5d_add',   # 1/16
            'block7b_add',   # 1/32
        ]
    elif phi == 2:
        model = EfficientNetB2(include_top=False,
                               weights=weights,
                               input_tensor=input_tensor)
        layer_names = [
            'block2c_add',   # 1/4
            'block3c_add',   # 1/8
            'block5d_add',   # 1/16
            'block7b_add',   # 1/32
        ]
    elif phi == 3:
        model = EfficientNetB3(include_top=False,
                               weights=weights,
                               input_tensor=input_tensor)
        layer_names = [
            'block2c_add',   # 1/4
            'block3c_add',   # 1/8
            'block5e_add',   # 1/16
            'block7b_add',   # 1/32
        ]
    elif phi == 4:
        model = EfficientNetB4(include_top=False,
                               weights=weights,
                               input_tensor=input_tensor)
        layer_names = [
            'block2c_add',   # 1/4
            'block3d_add',   # 1/8
            'block5f_add',   # 1/16
            'block7b_add',   # 1/32
        ]
    elif phi == 5:
        model = EfficientNetB5(include_top=False,
                               weights=weights,
                               input_tensor=input_tensor)
        layer_names = [
            'block2e_add',   # 1/4
            'block3e_add',   # 1/8
            'block5g_add',   # 1/16
            'block7c_add',   # 1/32
        ]
    elif phi == 6:
        model = EfficientNetB6(include_top=False,
                               weights=weights,
                               input_tensor=input_tensor)
        layer_names = [
            'block2f_add',   # 1/4
            'block3f_add',   # 1/8
            'block5h_add',   # 1/16
            'block7c_add',   # 1/32
        ]
    elif phi == 7:
        model = EfficientNetB7(include_top=False,
                               weights=weights,
                               input_tensor=input_tensor)
        layer_names = [
            'block2g_add',   # 1/4
            'block3g_add',   # 1/8
            'block5j_add',   # 1/16
            'block7d_add',   # 1/32
        ]

    skips = [model.get_layer(name).output for name in layer_names]
    return model, skips

def EasyDet(phi=0, input_size=(None, None, 3), weights='imagenet'):
    image_input = tf.keras.layers.Input(shape=input_size)

    backbone, skips = load_backbone(phi=phi, input_tensor=image_input, weights=weights)
    C2, C3, C4, C5 = skips

    in2 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
    in3 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
    in4 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
    in5 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)

    # 1 / 32 * 8 = 1 / 4
    P5 = tf.keras.layers.UpSampling2D(size=(8, 8))(
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5))
    # 1 / 16 * 4 = 1 / 4
    out4 = tf.keras.layers.Add()([in4, tf.keras.layers.UpSampling2D(size=(2, 2))(in5)])
    P4 = tf.keras.layers.UpSampling2D(size=(4, 4))(
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4))
    # 1 / 8 * 2 = 1 / 4
    out3 = tf.keras.layers.Add()([in3, tf.keras.layers.UpSampling2D(size=(2, 2))(out4)])
    P3 = tf.keras.layers.UpSampling2D(size=(2, 2))(
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3))
    # 1 / 4
    P2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.Add()([in2, tf.keras.layers.UpSampling2D(size=(2, 2))(out3)]))
    # (b, 1/4, 1/4, 256)
    fuse = tf.keras.layers.Concatenate()([P2, P3, P4, P5])

    model = tf.keras.models.Model(inputs=image_input, outputs=fuse)
    return model


if __name__ == '__main__':
    model = EasyDet(phi=0)
    model.summary()

    import time
    import numpy as np

    x = np.random.random_sample((1, 640, 640, 3))
    # warm up
    output = model.predict(x)

    print('\n[INFO] Test start')
    time_start = time.time()
    for i in range(1000):
        output = model.predict(x)
        
        time_end = time.time()
        print('[INFO] Time used: {:.2f} ms'.format((time_end - time_start)*1000/(i+1)))
