import to_tfrecord
import model
import numpy as np
import tensorflow as tf
import os

BATCH_SIZE = 25
learning_rate = 1e-4
MAX_STEP = 2000
weight_file = 'E:\\vgg\\vgg16_weights.npz'

def train():
    '''
    Train the model on the training data
    you need to change the training data directory below
    :return:
    '''
    tfrecords_file = 'E:\\vgg\\test\\test.tfrecords'
    log_dir = 'E:\\vgg\\test\\logs'

    images, labels = to_tfrecord.read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
    labels = to_tfrecord.one_hot(labels)

    train_logits, parameters, trainparameters = model.inference(images)
    train_loss = model.losses(train_logits,labels)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(train_loss, var_list=trainparameters)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

##################################
# load the pre-trained weights
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    for i, k in enumerate(keys[:-2]):
        print(i, k, np.shape(weights[k]))
        sess.run(parameters[i].assign(weights[k]))

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(vgg.probs),
    #                                                   reduction_indices=[1]))  # loss
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=vgg.trainparameters)

  #  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
   # train_op = optimizer.minimize(loss, global_step= my_global_step)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, loss_value = sess.run([train_op, train_loss])

            if step % 50 == 0:
                print ('Step: %d, loss: %.4f' % (step, loss_value))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


# %% To test the model on the test data
def evaluate():
    with tf.Graph().as_default():

        log_dir = '/home/kevin/tensorflow/CIFAR10/logs10000/'
        test_dir = '/home/kevin/tensorflow/CIFAR10/data/cifar-10-batches-bin/'
        n_test = 10000


        # reading test data
        images, labels = cifar10_input.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size= BATCH_SIZE,
                                                    shuffle=False)

        logits = inference(images)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)

            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.3f' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':
    train()