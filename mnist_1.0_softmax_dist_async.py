# encoding:utf-8
'''
python mnist*.py --job_name=ps --task_index=0
python mnist*.py --job_name=worker --task_index=0
python mnist*.py --job_name=worker --task_index=1
'''
import math
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

flags = tf.app.flags
IMAGE_PIXELS = 28
# 定义默认训练参数和数据路径
flags.DEFINE_string('data_dir', 'data', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 100, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '10.22.240.142:22220', 'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', '10.22.240.142:22221,10.22.240.142:22222',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = flags.FLAGS


def main(unused_argv):
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, reshape=False, validation_size=0)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print 'job_name : %s' % FLAGS.job_name
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print 'task_index : %d' % FLAGS.task_index

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
    with tf.device(tf.train.replica_device_setter(
            cluster=cluster
    )):
        global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量

        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        Y_ = tf.placeholder(tf.float32, [None, 10])

        W = tf.Variable(tf.zeros([28 * 28, 10]))
        b = tf.Variable(tf.zeros([10]))

        XX = tf.reshape(X, [-1, 784])

        # The model
        Y = tf.nn.softmax(tf.matmul(XX, W) + b)
        output = tf.matmul(XX, W) + b

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=output)
        cross_entropy = tf.reduce_mean(cross_entropy) * 100
        #tf.summary.scalar("ce", cross_entropy)

        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

        train_step = opt.minimize(cross_entropy, global_step=global_step)
        # 生成本地的参数初始化操作init_op
        init_op = tf.global_variables_initializer()

        train_dir = tempfile.mkdtemp()
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:
            print 'Worker %d: Initailizing session...' % FLAGS.task_index
        else:
            print 'Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index
        sess = sv.prepare_or_wait_for_session(server.target)
        print 'Worker %d: Session initialization  complete.' % FLAGS.task_index
        #writer = tf.summary.FileWriter('./log', sess.graph)
        #merge_op = tf.summary.merge_all()

        sv.start_queue_runners(sess)

        time_begin = time.time()
        print 'Traing begins @ %f' % time_begin

        local_step = 0
        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            train_feed = {X: batch_xs, Y_: batch_ys}

            #_,_, ce, step = sess.run([train_step,merge_op, cross_entropy,  global_step], feed_dict=train_feed)
            _,ce, step = sess.run([train_step, cross_entropy,  global_step], feed_dict=train_feed)
            local_step += 1

            now = time.time()
            print '%f: Worker %d: traing step %d dome (global step:%d), ce %f' % (now, FLAGS.task_index, local_step, step, ce)

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print 'Training ends @ %f' % time_end
        train_time = time_end - time_begin
        print 'Training elapsed time:%f s' % train_time

        val_feed = {X: mnist.test.images, Y_: mnist.test.labels}
        val_xent = sess.run(cross_entropy, feed_dict=val_feed)
        print 'After %d training step(s), validation cross entropy = %g' % (FLAGS.train_steps, val_xent)
    sess.close()

if __name__ == '__main__':
    tf.app.run()
