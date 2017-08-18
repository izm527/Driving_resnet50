import tensorflow as tf
import numpy as np
import random
import cv2
from nets import resnet_v1 as resnet
slim = tf.contrib.slim

cifarFile = "/home/izm/tianchi/cifar-10/data_batch_1"
#checkpoint_file = 'chp/model.ckpt-1600.data-00000-of-00001'
checkpoint_file = 'chp/model.ckpt-1200'
#checkpoint_file = 'checkpoint/resnet_v1_50.ckpt'
save_chp = 'saves/'
drive_images = '/home/izm/work/mxnet/Ex4_DistractedDr/input/distracteddriver/imgs/train'
dt_prfx = 'data/'
split_num = 22435/10




def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
        return dict



def readpic(name_list):
    res = []
    '''
    for nm in name_list:
        imagecontent = tf.read_file(nm)
        image0 = tf.image.decode_jpeg(imagecontent, 3)
        print image0
        resized_image = tf.image.resize_images(image0, [224, 224], \
                    method=tf.image.ResizeMethod.AREA)
        mat = resized_image.eval()
        res.append(mat)
    res = array(res, 'float')
    np.save(res)
    '''
    #opencv for resizing the pics

    cnt = 0
    for nm in name_list:
        image = cv2.imread(nm)
        res_img = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
    #    res_img = res_img - mean
        res.append(res_img)
        cnt += 1
        print cnt, ' : processed ',nm

        #show pictures
        #cv2.imshow('iker',res)
        #cv2.imshow('image',image)
        #cv2.waitKey(0)

    return np.array(res)


def construct_network(sess):


    #Load the model
    pics = tf.placeholder("float32", shape=[None, 224,224,3])      #for storing label
    with slim.arg_scope(resnet.resnet_arg_scope()):
        logits, end_points = resnet.resnet_v1_50(pics, 1000, is_training=False)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

    #print str(end_points).replace('),' , '),\n')
    #print type(end_points)
    #---------------Tensorboard showing
    #tf.summary.FileWriter('log', sess.graph)
    #merged_summary_op = tf.merge_all_summaries()
    #summary_writer = tf.train.SummaryWriter('log', sess.graph)
    #raw_input('Check the tfb from the web\n')
    #tf.initialize_all_variables()


    pred = end_points['predictions']
    lg = end_points['resnet_v1_50/logits']

    return end_points




def next_batch(start, data, batch_size):
    m = len(data)
    if start + batch_size >= m:
        return data[start:m], 0
    else:
        return data[start : start+batch_size], start+batch_size


def run_model(network, sess, vec, label, test_data, test_lb):


    pred = network['predictions']
    lg = network['resnet_v1_50/logits']
    flt = tf.Variable(tf.random_normal([1,1,1000,10]))
    #sess.run(flt.initialized_value())

    cv = tf.nn.conv2d(lg, flt, [1,1,1,1], "SAME")
    sq = tf.squeeze(cv, squeeze_dims=[1,2])
    y = tf.placeholder('float32', shape=[None, 10])     #y is the label's placeholder

    #reconstruct the output
    #cross_entropy = tf.reduce_mean((sfy)*(sfy))
    #cross_entropy = tf.reduce_sum((y-b)*(y-b))   #another type of cross_entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=sq))


    # for testing the results
    pred_res = tf.argmax(tf.nn.softmax(sq), 1)
    lb_res = tf.argmax(y, 1)

    #-----------------------constuct the learning rate decaying
    init_lr = 0.00008
    global_step = tf.Variable(0, trainable=False)
    #sess.run(global_step.initialized_value())
    learning_rate = tf.train.exponential_decay(init_lr, global_step=global_step, decay_steps=10, decay_rate=0.94)
    add_global = global_step.assign_add(1)

    mx = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    #----------------print the graph----------------
    #graph = tf.get_default_graph()
    #ops = graph.get_operations()
    #print str(ops).replace(',' , '\n')
    tf.summary.FileWriter('log', sess.graph)
    raw_input('alibb')

    
    train_step = 2000
    batch_size = 20
    idx1 = 0
    idx2 = 0
    #sess.run(tf.initialize_all_variables())
    #saver = tf.train.Saver()
    #saver.restore(sess, checkpoint_file)
    print "restore the model from checkpoint!\n"
    print vec.shape, label.shape


    res = sess.run([pred_res, lb_res], feed_dict={'Placeholder:0':test_data, 'Placeholder_1:0': test_lb})
    print res
    prd = res[0]
    test_lb = res[1]
    crrt = 0
    for ii in range(len(res[0])):
        print prd[ii], test_lb[ii], crrt
        if prd[ii] == test_lb[ii]:
            crrt += 1
    print "correct rate: ", (crrt+0.001)/len(res[0])
    
    raw_input()


    for i in range(train_step):
        dtx, idx1 = next_batch(idx1, vec, batch_size)
        dty, idx2 = next_batch(idx2, label, batch_size)

        print 'step', i, ': res = '
        #res = sess.run([mx, add_global, cross_entropy, learning_rate], feed_dict={'Placeholder:0':dtx, 'Placeholder_1:0':dty})
        #print res

        #for testing
        if i%5 == 0:
            res = sess.run([pred_res, lb_res], feed_dict={'Placeholder:0':test_data})
            prd = res[0], test_lb = res[1]
            crrt = 0
            for ii in range(len(res[0])):
                print prd[ii], test_lb[ii], crrt
                if prd[ii] == test_lb[ii]:
                    crrt += 1
            print "correct rate: ", crrt/len(res[0])

        
        #for model saving
        if i%400 == 399:
            saver.save(sess, save_chp + 'model.ckpt', global_step=i+1)



train_prefix = '/home/izm/work/mxnet/Ex4_DistractedDr/input/distracteddriver/imgs/train/'


def save_array():
    #-----shuffle the file list---------------
    #lines = open(train_prefix+'label.txt').readlines()
    #random.shuffle(lines)
    #wf = open(train_prefix+'label1.txt', 'w')
    #for ll in lines:
    #    wf.write(ll)


    ff = open(train_prefix+'label.txt').readlines()
    lbs = []
    nms = []
    for line in ff:
        item = line.split(' ')
        nms.append(train_prefix+item[0])
        tmp = [0 for j in range(10)]
        tmp[int(item[1][0])] = 1
        lbs.append(tmp)


    #print names
    for i in range(2):
        data_mat= readpic(nms[i*split_num : (i+1)*split_num])
        lb_mat = np.array(lbs[i*split_num : (i+1)*split_num])
        np.save(dt_prfx + 'img_mat'+str(i), data_mat)
        np.save(dt_prfx + 'lb'+str(i), lb_mat)
        print '\n\n\nsplit ', i, "get the mats from pics! array shape: ", data_mat.shape


if __name__ == '__main__':
    #save_array()
    #data_mat = np.concatenate(np.load(dt_prfx+'img_mat0.npy'), np.load(dt_prfx+'img_mat1.npy'))
    #lb_mat = np.concatenate(np.load(dt_prfx+'lb0.npy'), np.load(dt_prfx+'lb1.npy'))

    data_mat = np.load(dt_prfx+'img_mat0.npy')
    lb_mat = np.load(dt_prfx+'lb0.npy')
    data_mat1 = np.load(dt_prfx+'img_mat1.npy')
    lb_mat1 = np.load(dt_prfx+'lb1.npy')
    data_mat = np.concatenate((data_mat, data_mat1))
    lb_mat = np.concatenate((lb_mat, lb_mat1))

    data_test = np.load(dt_prfx+'img_mat2.npy')
    lb_test = np.load(dt_prfx+'lb2.npy')
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            res = construct_network(sess)
            run_model(res, sess, data_mat, lb_mat, data_mat[0:200], lb_mat[0:200])
