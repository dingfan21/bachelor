import shutil
import xml.etree.ElementTree as ET
import os
import tensorflow as tf
from PIL import Image


def organ_classify():
    '''
    distribute the data images into 7 different folders according to the 'Content' labels
    window system! (replace all the "\\" with "/" in linux system)
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

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

#organ_classify()
# create_record()

cwd = os.getcwd()
img, label = read_and_decode("train.tfrecords")
print(img)
print(label)
with tf.Session() as sess:  # 开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(14):
        example, l = sess.run([img, label])  # 在会话中取出image和label
        image = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
        # print("hello"+str(i))
        image.save(cwd + "\\" + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
        #print(example.shape, l)
    coord.request_stop()
    coord.join(threads)

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
