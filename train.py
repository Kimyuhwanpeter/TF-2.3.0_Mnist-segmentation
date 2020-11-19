# -*- coding:utf-8 -*-
from absl import flags, app
from random import random, shuffle
from model import *

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os

flags.DEFINE_bool('train', True, "")

flags.DEFINE_string("txt_path", "D:/[1]DB/[4]etc_experiment/Mnist/train.txt", "Training images path")

flags.DEFINE_integer("img_size", 64, "Image size")

flags.DEFINE_float("lr", 0.001, "Training learning rate")

flags.DEFINE_integer("batch_size", 1, "Batch size")

flags.DEFINE_integer("num_classes", 10, "Number of classes")

flags.DEFINE_integer("epochs", 100, "Total epochs")

flags.DEFINE_bool("pre_checkpoint", False, "")

flags.DEFINE_string("pre_checkpoint_path", "", "")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.keras.optimizers.Adam(FLAGS.lr)

# Segmenation 분야에서도 얻을 수 있는 아이디어가 많기 때문에 한번 다시 세그맨테이션 모델을 코딩해보자
def func(img_list, lab_list):
    
    img = tf.io.read_file(img_list)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size]) / 255.0

    lab_img = tf.io.read_file(lab_list[0])
    lab_img = tf.image.decode_png(lab_img, 1)
    lab_img = tf.image.resize(lab_img, [FLAGS.img_size, FLAGS.img_size])
    lab_img = tf.image.convert_image_dtype(lab_img, tf.int32)
    
    if int(lab_list[1]) == 0:       # 1,2,3,4,5,6,7,8,9,10 으로 sparse 하게 만들어놓으면될것같음!!
        lab_img = tf.where(lab_img[:, :, :] != 0, 1, 0)
    if int(lab_list[1]) == 1:
        lab_img = tf.where(lab_img[:, :, :] != 0, 2, 0)
    if int(lab_list[1]) == 2:
        lab_img = tf.where(lab_img[:, :, :] != 0, 3, 0)
    if int(lab_list[1]) == 3:
        lab_img = tf.where(lab_img[:, :, :] != 0, 4, 0)
    if int(lab_list[1]) == 4:
        lab_img = tf.where(lab_img[:, :, :] != 0, 5, 0)
    if int(lab_list[1]) == 5:
        lab_img = tf.where(lab_img[:, :, :] != 0, 6, 0)
    if int(lab_list[1]) == 6:
        lab_img = tf.where(lab_img[:, :, :] != 0, 7, 0)
    if int(lab_list[1]) == 7:
        lab_img = tf.where(lab_img[:, :, :] != 0, 8, 0)
    if int(lab_list[1]) == 8:
        lab_img = tf.where(lab_img[:, :, :] != 0, 9, 0)
    if int(lab_list[1]) == 9:
        lab_img = tf.where(lab_img[:, :, :] != 0, 10, 0)
   
    labels = int(lab_list[1])

    return img, lab_img, labels

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def True_fn(logits, labels):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)

def cal_loss(model, batch_images, batch_label_images, batch_labels):

    with tf.GradientTape() as tape:
        fake_img = run_model(model, batch_images, True)
        #fake_img = tf.nn.softmax(fake_img, axis=3)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(batch_label_images, fake_img)

    gradients = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def main(argv=None):
    # Segmentation model
    model = seg_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model.summary()

    #tf.keras.utils.plot_model(model, show_shapes=True)

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("===============================================")
            print("* Succeed the restoring checkpoint files!!!!! *")
            print("===============================================")

    if FLAGS.train:
        count = 0;

        img_data = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
        lab_img = img_data
        img_data = ["D:/[1]DB/[4]etc_experiment/Mnist/train/" + img for img in img_data]

        lab_img = ["D:/[1]DB/[4]etc_experiment/Mnist/seg_train/" + img for img in lab_img]
        lab_img = np.array(lab_img)
        lab_data = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        lab_data = list(zip(lab_img, lab_data))
        lab_data = np.array(lab_data)
        
        for epoch in range(FLAGS.epochs):
            train_gener = tf.data.Dataset.from_tensor_slices((img_data, lab_data))
            train_gener = train_gener.shuffle(len(img_data))
            train_gener = train_gener.map(func)
            train_gener = train_gener.batch(FLAGS.batch_size)
            train_gener = train_gener.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(train_gener)
            idx = len(img_data) // FLAGS.batch_size
            for step in range(idx):
                batch_images, batch_label_images, batch_labels = next(it)
                #plt.imshow(batch_images[0], cmap='gray')
                #plt.imshow(batch_label_images[0, :, :, :])
                #print(batch_labels[0])
                #plt.show()
                loss = cal_loss(model, batch_images, batch_label_images, batch_labels)
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step, idx, loss))

                if count % 100 == 0:
                    fake_img = run_model(model, batch_images, False)
                    fake_img = tf.nn.softmax(fake_img, 3)
                    fake_img = tf.argmax(fake_img[0], 2)

                    b_img = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.float32)
            
                    #color = [[255, 255, 255],   # White
                    #         [255, 0, 0],       # Red
                    #         [255, 255, 0],     # Yellow
                    #         [128, 128, 0],     # Olive
                    #         [0, 255, 0],       # Lime
                    #         [0, 255, 255],     # Aqua
                    #         [0, 128, 128],     # Teal
                    #         [0, 0, 255],       # Blue
                    #         [255, 0, 255],     # Fuchsia
                    #         [128, 0, 128]]     # Purple
                    #tf.where(fake_img[:,:] == 1,  )
                    for i in range(FLAGS.img_size):
                        for j in range(FLAGS.img_size):
                            if fake_img[i, j] == 1:
                                b_img[i, j, 0] = 255
                                b_img[i, j, 1] = 255
                                b_img[i, j, 2] = 255
                            elif fake_img[i, j] == 2:
                                b_img[i, j, 0] = 255
                                b_img[i, j, 1] = 0
                                b_img[i, j, 2] = 0
                            elif fake_img[i, j] == 3:
                                b_img[i, j, 0] = 255
                                b_img[i, j, 1] = 255
                                b_img[i, j, 2] = 0
                            elif fake_img[i, j] == 4:
                                b_img[i, j, 0] = 128
                                b_img[i, j, 1] = 128
                                b_img[i, j, 2] = 0
                            elif fake_img[i, j] == 5:
                                b_img[i, j, 0] = 0
                                b_img[i, j, 1] = 255
                                b_img[i, j, 2] = 0
                            elif fake_img[i, j] == 6:
                                b_img[i, j, 0] = 0
                                b_img[i, j, 1] = 255
                                b_img[i, j, 2] = 255
                            elif fake_img[i, j] == 7:
                                b_img[i, j, 0] = 0
                                b_img[i, j, 1] = 128
                                b_img[i, j, 2] = 128
                            elif fake_img[i, j] == 8:
                                b_img[i, j, 0] = 0
                                b_img[i, j, 1] = 0
                                b_img[i, j, 2] = 255
                            elif fake_img[i, j] == 9:
                                b_img[i, j, 0] = 255
                                b_img[i, j, 1] = 0
                                b_img[i, j, 2] = 255
                            elif fake_img[i, j] == 0:
                                b_img[i, j, 0] = 0
                                b_img[i, j, 1] = 0
                                b_img[i, j, 2] = 0

                    plt.imsave("C:/Users/Yuhwan/Pictures/sample/real_{}.png".format(count), batch_images[0].numpy() * 255)
                    plt.imsave("C:/Users/Yuhwan/Pictures/sample/output_{}.png".format(count), b_img)
                        
                count += 1

if __name__ == "__main__":
    app.run(main)
