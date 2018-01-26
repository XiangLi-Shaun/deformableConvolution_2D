import os
import time
import argparse
import numpy as np
import tensorflow as tf
from u_net import UNet

def configure():
    
    # 关于训练的参数
    flags = tf.app.flags
    flags.DEFINE_integer('max_epoch', 10, '# of step in an epoch')
    flags.DEFINE_integer('test_step', 100, '# of step to test a model')
    flags.DEFINE_integer('save_step', 100, '# of step to save a model')
    flags.DEFINE_integer('summary_step', 100000, '# of step to save the summary')
    
    # 设置优化器参数
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('beta1', 0.8, 'beta1')
    flags.DEFINE_float('beta2', 0.9, 'beta2')
    flags.DEFINE_float('epsilon', 1e-8, 'epsilon')
    
    flags.DEFINE_float('keep_prob', 0.9, 'dropout probability')
    flags.DEFINE_boolean('use_gpu', True, 'use GPU or not')
    
    # 关于验证的参数
    flags.DEFINE_integer('valid_start_epoch',2001,'start step to test a model')
    flags.DEFINE_string('valid_end_epoch',7501,'end step to test a model')
    flags.DEFINE_string('valid_stride_of_epoch',100,'stride to test a model')
    
    # 数据的存储和信息
    flags.DEFINE_string('data_dir', '/home/mzhang/dcn/data/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'cell_train_classify.h5', 'Training data')
    flags.DEFINE_string('valid_data', 'cell_valid_classify.h5', 'Validation data')
    flags.DEFINE_string('test_data', 'cell_valid_classify.h5', 'Testing data')
    flags.DEFINE_integer('batch', 5, 'batch size')
    flags.DEFINE_integer('channel', 3, 'channel size')
    flags.DEFINE_integer('height', 256, 'height size')
    flags.DEFINE_integer('width', 256, 'width size')
    
    # 存储路径
    flags.DEFINE_string('logdir', '/home/mzhang/dcn_2/logdir', 'Log dir')
    flags.DEFINE_string('modeldir', '/home/mzhang/dcn_2/modeldir', 'Model dir')
    flags.DEFINE_string('sample_dir', '/home/mzhang/dcn_2/samples/', 'Sample directory')
    flags.DEFINE_string('record_dir', '/home/mzhang/dcn_2/record/', 'Experiment record directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_epoch', 0, 'Reload epoch')
    flags.DEFINE_integer('test_epoch', 4401, 'Test or predict epoch')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    
    # 网络参数
    flags.DEFINE_integer('network_depth', 4, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 3, 'output class number')
    flags.DEFINE_integer('start_channel_num', 8, 'start number of outputs')
    flags.DEFINE_string('down_conv_name', 'deform_conv2d', 'Use which conv op: conv2d or deform_conv2d')
    flags.DEFINE_string('bottom_conv_name', 'deform_conv2d', 'Use which conv op: conv2d or deform_conv2d')
    flags.DEFINE_string('up_conv_name', 'deform_conv2d', 'Use which conv op: conv2d or deform_conv2d')
    flags.DEFINE_string('deconv_name', 'deconv', 'Use which deconv op: deconv, dilated_conv, co_dilated_conv')
    
    # Deformable Convolution 
    flags.DEFINE_boolean('add_dcn', False, 'add Deformable Convolution or not')
    flags.DEFINE_integer('dcn_location', 2,'The Deformable Convolution location')
    flags.DEFINE_integer('dcn_location2', 3,'The Deformable Convolution location')
      
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS

def train():
    model = UNet(sess, configure())
    model.train()

def valid():
    valid_loss = []
    valid_accuracy = []
    valid_m_iou = []
    conf = configure()
    model = UNet(sess, conf)
    for i in range(conf.valid_start_epoch,conf.valid_end_epoch,conf.valid_stride_of_epoch):
        loss,acc,m_iou=model.test(i)
        valid_loss.append(loss)
        valid_accuracy.append(acc)
        valid_m_iou.append(m_iou)
        np.save(conf.record_dir+"validate_loss.npy",np.array(valid_loss))
        np.save(conf.record_dir+"validate_accuracy.npy",np.array(valid_accuracy))
        np.save(conf.record_dir+"validate_m_iou.npy",np.array(valid_m_iou))
        print('valid_loss',valid_loss)
        print('valid_accuracy',valid_accuracy)
        print('valid_m_iou',valid_m_iou)

def predict(): 
    predict_loss = []
    predict_accuracy = []
    predict_m_iou = []
    model = UNet(sess, configure())
    loss,acc,m_iou = model.predict()
    predict_loss.append(loss)
    predict_accuracy.append(acc)
    predict_m_iou.append(m_iou)
    print('predict_loss',predict_loss)
    print('predict_accuracy',predict_accuracy)
    print('predict_m_iou',predict_m_iou)

def main(argv):
    start = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.action not in ['train', 'test', 'predict']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test, or predict")
    # test
    elif args.action == 'test':
        valid()
    # predict
    elif args.action == 'predict':
        predict()
    # train
    else:
        train()
    end = time.clock()
    print("program total running time",(end-start)/60)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto(log_device_placement=False)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess = tf.Session()
    tf.app.run()
