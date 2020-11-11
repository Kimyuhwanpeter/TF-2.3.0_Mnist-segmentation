# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

color = [[255, 255, 255],   # White
         [255, 0, 0],       # Red
         [255, 255, 0],     # Yellow
         [128, 128, 0],     # Olive
         [0, 255, 0],       # Lime
         [0, 255, 255],     # Aqua
         [0, 128, 128],     # Teal
         [0, 0, 255],       # Blue
         [255, 0, 255],     # Fuchsia
         [128, 0, 128]]     # Purple

def func(img_path, lab):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, 1)
    
    lab = int(lab)

    return img, lab

def main():
    read_txt = "D:/[1]DB/[4]etc_experiment/Mnist"
    img_path = np.loadtxt(read_txt + "/train.txt", dtype="<U100", skiprows=0, usecols=0)
    name = img_path
    img_path = [read_txt + "/train/" + img for img in img_path]
    lab_path = np.loadtxt(read_txt + "/train.txt", dtype=np.int32, skiprows=0, usecols=1)

    generation = tf.data.Dataset.from_tensor_slices((img_path, lab_path))
    generation = generation.map(func)
    generation = generation.batch(1)
    generation = generation.prefetch(tf.data.experimental.AUTOTUNE)

    it = iter(generation)
    for step in range(len(img_path)):
        img, lab = next(it)
        #plt.imshow(img[0, :, :, 0], cmap="gray")
        #plt.show()
        if lab == 0:
            numpy_img = img[0].numpy()
            slice = tf.where(numpy_img[:, :, 0] != 0, 255, 0)
            slice = tf.expand_dims(slice, 2)
            slice = tf.concat([slice,slice,slice], 2)
            plt.imsave(read_txt + "/seg_train/" + name[step], slice)
        if lab == 1:
            numpy_img = img[0].numpy()
            slice = tf.where(numpy_img[:, :, 0] != 0, 255, 0)
            slice = tf.expand_dims(slice, 2)
            slice = tf.concat([slice,tf.zeros([28, 28, 1], dtype=tf.int32),tf.zeros([28, 28, 1], dtype=tf.int32)], 2)
            plt.imsave(read_txt + "/seg_train/" + name[step], slice)
        if lab == 2:
            numpy_img = img[0].numpy()
            slice = tf.where(numpy_img[:, :, 0] != 0, 255, 0)
            slice = tf.expand_dims(slice, 2)
            slice = tf.concat([slice, slice,tf.zeros([28, 28, 1], dtype=tf.int32)], 2)
            plt.imsave(read_txt + "/seg_train/" + name[step], slice)
        if lab == 3:
            numpy_img = img[0].numpy()
            slice = tf.where(numpy_img[:, :, 0] != 0, 128, 0)
            slice = tf.expand_dims(slice, 2)
            slice = tf.concat([slice, slice,tf.zeros([28, 28, 1], dtype=tf.int32)], 2)
            plt.imsave(read_txt + "/seg_train/" + name[step], slice)
        if lab == 4:
            numpy_img = img[0].numpy()
            slice = tf.where(numpy_img[:, :, 0] != 0, 255, 0)
            slice = tf.expand_dims(slice, 2)
            slice = tf.concat([tf.zeros([28, 28, 1], dtype=tf.int32), slice, tf.zeros([28, 28, 1], dtype=tf.int32)], 2)
            plt.imsave(read_txt + "/seg_train/" + name[step], slice)
        if lab == 5:
            numpy_img = img[0].numpy()
            slice = tf.where(numpy_img[:, :, 0] != 0, 255, 0)
            slice = tf.expand_dims(slice, 2)
            slice = tf.concat([tf.zeros([28, 28, 1], dtype=tf.int32), slice, slice], 2)
            plt.imsave(read_txt + "/seg_train/" + name[step], slice)
        if lab == 6:
            numpy_img = img[0].numpy()
            slice = tf.where(numpy_img[:, :, 0] != 0, 128, 0)
            slice = tf.expand_dims(slice, 2)
            slice = tf.concat([tf.zeros([28, 28, 1], dtype=tf.int32), slice, slice], 2)
            plt.imsave(read_txt + "/seg_train/" + name[step], slice)
        if lab == 7:
            numpy_img = img[0].numpy()
            slice = tf.where(numpy_img[:, :, 0] != 0, 255, 0)
            slice = tf.expand_dims(slice, 2)
            slice = tf.concat([tf.zeros([28, 28, 1], dtype=tf.int32), tf.zeros([28, 28, 1], dtype=tf.int32), slice], 2)
            plt.imsave(read_txt + "/seg_train/" + name[step], slice)
        if lab == 8:
            numpy_img = img[0].numpy()
            slice = tf.where(numpy_img[:, :, 0] != 0, 255, 0)
            slice = tf.expand_dims(slice, 2)
            slice = tf.concat([slice, tf.zeros([28, 28, 1], dtype=tf.int32), slice], 2)
            plt.imsave(read_txt + "/seg_train/" + name[step], slice)
        if lab == 9:
            numpy_img = img[0].numpy()
            slice = tf.where(numpy_img[:, :, 0] != 0, 128, 0)
            slice = tf.expand_dims(slice, 2)
            slice = tf.concat([slice, tf.zeros([28, 28, 1], dtype=tf.int32), slice], 2)
            plt.imsave(read_txt + "/seg_train/" + name[step], slice)


        #plt.imsave(read_txt + "/seg_train/" + name[step], img[0])

        print(step)


if __name__ == "__main__":
    main()