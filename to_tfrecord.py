import shutil
import xml.etree.ElementTree as ET
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def organ_classify():
    '''
    distribute the data images into 7 different folders according to the 'Content' labels,
    the 7 folders should be constructed manually before the function is called

    notice:
        this function only need to be executed once
        replace all the "\\" with "/" for linux system
    '''

    cwd = os.getcwd()                   # working path
    path = cwd  + "\\xml\\"            # folder contains all the xml files
    for xml_name in os.listdir(path):
        xml_path = path+xml_name                         # path of every xml file
        tree = ET.ElementTree(file=xml_path)
        content = tree.find('Content').text               # the category
        jpg_name =  tree.find('MediaId').text + '.jpg'   # the path of the jpg file
        scr_path = cwd + "\\"+jpg_name                    # source path
        des_path = cwd + "\\"+ content + "\\" + jpg_name  # destination path
        shutil.move(scr_path,des_path)                     # cut and copy the file



def create_record():
    '''
    this function is not used, because it doesn't shuffle the data
    此处我加载的数据目录如下：
    Flower --   1.jpg
                2.jpg
                3.jpg
                ...
    Entire --   1.jpg
                2.jpg
                ...
    Fruit  --   ...
                ...
    '''
    cwd = os.getcwd()
    classes = ['Branch','Entire','Leaf','LeafScan','Flower','Fruit','Stem']
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd +"\\"+ name + "\\"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img_raw = img.tobytes() #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def get_file(file_dir):
    '''Get full image directory and corresponding labels
    Args:
        file_dir: file directory
    Returns:
        images: image directories, list, string
        labels: label, list, int
    '''

    images = []
    labels = []
    classes = ['Branch', 'Entire', 'Leaf', 'LeafScan', 'Flower', 'Fruit', 'Stem']
    for index, name in enumerate(classes):
        class_path = file_dir +"\\"+ name + "\\"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            images.append(img_path)
            labels.append(index)

    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    return image_list, label_list


# %%

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# %%
def convert_to_tfrecord(images, labels, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''

    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)

    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], n_samples))

    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            image = Image.open(images[i])
            image = image.resize((224, 224))
            image_raw = image.tobytes()  # 将图片转化为原生bytes
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': int64_feature(label),
                'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' % e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')



# def read_and_decode(filename):
#     filename_queue = tf.train.string_input_producer([filename])
#
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw' : tf.FixedLenFeature([], tf.string),
#                                        })
#
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [224, 224, 3])
#     #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
#     label = tf.cast(features['label'], tf.int32)
#     return img, label


def read_and_decode(tfrecords_file, batch_size):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.

    image = tf.reshape(image, [224, 224,3])
    image = tf.cast(image,tf.float32)
    image = tf.image.per_image_standardization(image)  # normalize


    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=3,
                                              capacity=25)
    return image_batch, tf.reshape(label_batch, [batch_size])


def one_hot(label_batch):
    ## ONE-HOT
    n_classes = 7
    label_batch = tf.one_hot(label_batch, depth=n_classes)
    return label_batch





#######################################################3
# TO test train.tfrecord file

def plot_images(images, labels):
    '''plot one batch size
    '''
    classes = ['Branch', 'Entire', 'Leaf', 'LeafScan', 'Flower', 'Fruit', 'Stem']
    for i in np.arange(0, BATCH_SIZE):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title(str(classes[labels[i]]), fontsize = 14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()

########################################################
########################################################
# Convert data to TFRecord

if __name__ == '__main__':
    test_dir = "E:\\vgg\\test"
    save_dir = "E:\\vgg\\test"
    BATCH_SIZE = 10

####################################################
#Convert test data: you just need to run it ONCE !
    # name_test = 'test'
    # images, labels = get_file(test_dir)
    # convert_to_tfrecord(images, labels, save_dir, name_test)

    tfrecords_file = 'E:\\vgg\\test\\test.tfrecords'
    image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)

    with tf.Session()  as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i < 1:
            # just plot one batch size
                image, label = sess.run([image_batch, label_batch])
                plot_images(image, label)
                i += 1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)

# img, label = read_and_decode("train.tfrecords")
# print(img)
# print(label)
# with tf.Session() as sess:  # 开始一个会话
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(14):
#         example, l = sess.run([img, label])  # 在会话中取出image和label
#         image = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
#         # print("hello"+str(i))
#         image.save(cwd + "\\" + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
#         #print(example.shape, l)
#     coord.request_stop()
#     coord.join(threads)

# if __name__ == '__main__':
#     cwd = os.getcwd()
#     img, label = read_and_decode("train.tfrecords")
#     img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                     batch_size=30, capacity=2000,
#                                                     min_after_dequeue=1000)
#     #初始化所有的op
#     init = tf.initialize_all_variables()
#
#     with tf.Session() as sess:
#         sess.run(init)
# 	#启动队列
#         threads = tf.train.start_queue_runners(sess=sess)
#         for i in range(3):
#             val, l= sess.run([img_batch, label_batch])
#             #l = to_categorical(l, 12)
#             print(val.shape, l)
