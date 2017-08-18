import tensorflow as tf
import numpy as np
import random
from nets import resnet_v1 as resnet
slim = tf.contrib.slim

cifarFile = "/home/izm/tianchi/cifar-10/data_batch_1"
checkpoint_file = 'chp/model.ckpt-1200'
#checkpoint_file = 'checkpoint/resnet_v1_50.ckpt'
save_chp = 'chp/'
drive_images = '/home/izm/work/mxnet/Ex4_DistractedDr/input/distracteddriver/imgs/train'
dt_prfx = 'data/'
split_num = 22435/10




def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
        return dict



def construct_network(sess):


    #Load the model
    pics = tf.placeholder("float32", shape=[None, 224,224,3])      #for storing label
    with slim.arg_scope(resnet.resnet_arg_scope()):
        logits, end_points = resnet.resnet_v1_50(pics, 1000, is_training=False)
    #    saver = tf.train.Saver()
    #    saver.restore(sess, checkpoint_file)

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
    #res = sess.run(lg, feed_dict={'Placeholder:0':x})
    #mx = 0
    #for i in range(len(res[0])):
    #    if res[0][i] > mx:
    #        mx = i
    #print 'final label prob = ', mx

    return end_points

    #saver = tf.train.import_meta_graph('tensorflow-resnet-pretrained-20160509/ResNet-L50.meta')
    #network = rn.resnet_v1_50(pics)
    #vals = slim.get_model_variables()
    #slim.assign_from_checkpoint_fn(chpt_dir, vals, True)

    #graph = tf.get_default_graph()
    #op = graph.get_operation_by_name('resnet_v2_50/block2/unit_4/bottleneck_v2/conv1/biases')
    # To initialize values with saved data
    #ops = graph.get_operations()
    #print str(ops).replace(',' , '\n')


    #drpo = tf.nn.dropout(fc, 0.75)
    #res = tf.nn.softmax(drpo)



def next_batch(start, data, batch_size):
    m = len(data)
    if start + batch_size >= m:
        return data[start:m], 0
    else:
        return data[start : start+batch_size], start+batch_size


def run_model(network, sess, vec, label, test_data, test_lb):
    #print vec[0],vec[1],vec[2]

    
    # fix the wrong subtract op
    #graph = tf.get_default_graph()
    #print graph.get_operations()
    #y_ = tf.placeholder("float", shape=[None, 10])      #for storing label
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=ss[0]))
    #print sess.run(cross_entropy, feed_dict={'concat:0':sb.eval(), y_:label})
    ## the real y_ value
    #mx = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #res = sess.run(mx, feed_dict={ y_ : label})

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

    
    train_step = 1200
    batch_size = 2
    idx1 = 0
    idx2 = 0
    global_step = tf.Variable(0, trainable=False)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

    init_lr = 3.0
    #sess.run(global_step.initialized_value())
    learning_rate = tf.train.exponential_decay(init_lr, global_step=global_step, decay_steps=10, decay_rate=0.975)
    add_global = global_step.assign_add(1)

    mx = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    print "restore the model from checkpoint!\n"
    print vec.shape, label.shape


    #---------------------------------just for testing ---------------------------------
    #res = sess.run(flt)
    #print res
    #raw_input()

    #res = sess.run([pred_res, lb_res], feed_dict={'Placeholder:0':test_data, 'Placeholder_1:0': test_lb})
    #print res 
    #prd = res[0]
    #test_lb = res[1]
    #crrt = 0
    #for ii in range(len(res[0])):
    #    print prd[ii], test_lb[ii], crrt
    #    if prd[ii] == test_lb[ii]:
    #        crrt += 1
    #print "correct rate: ", (crrt+0.001)/len(res[0])
    


    for i in range(train_step):
        dtx, idx1 = next_batch(idx1, vec, batch_size)
        dty, idx2 = next_batch(idx2, label, batch_size)
        print 'step', i, ': res = '
        res = sess.run([add_global, cross_entropy, learning_rate], feed_dict={'Placeholder:0':dtx, 
                            'Placeholder_1:0':dty, 'ExponentialDecay/learning_rate:0':0.024})
        print res

        #for testing
        if i%30 == 29:
            res = sess.run([pred_res, lb_res, flt], feed_dict={'Placeholder:0':test_data, 'Placeholder_1:0': test_lb})
            prd = res[0]
            lb = res[1]
            cov = res[2]
            crrt = 0
            for ii in range(len(prd)):
                #print prd[ii], test_lb[ii], crrt
                if prd[ii] == lb[ii]:
                    crrt += 1
            print "correct rate: ", (crrt+0.01)/len(prd)
            print cov


        #for model saving
        if i%300 == 299:
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


def read_data():
	dit = unpickle('cifar-10/data_batch_1')
	data = dit['data']
	labels = dit['labels']
	
	dit = unpickle('cifar-10/data_batch_2')
	data = np.concatenate((data, dit['data']))
	labels = np.concatenate((labels, dit['labels']))

	dit = unpickle('cifar-10/data_batch_3')
	data = np.concatenate((data, dit['data']))
	labels = np.concatenate((labels, dit['labels']))

	dit = unpickle('cifar-10/data_batch_4')
	data = np.concatenate((data, dit['data']))
	labels = np.concatenate((labels, dit['labels']))

	dit = unpickle('cifar-10/data_batch_5')
	data = np.concatenate((data, dit['data']))
	labels = np.concatenate((labels, dit['labels']))


	train_data = []
	train_lb = np.zeros([len(labels), 10])	
	for i in range(len(data)):
		pp = np.reshape(data[i], [3,32,32])
		dd = np.zeros([3,224,224])
		for ii in range(3):
			for jj in range(32):
				for kk in range(32):
					dd[ii][jj][kk] = pp[ii][jj][kk]
		train_data.append(dd)
		train_lb[i][labels[i]] = 1
	train_data = np.array(train_data).transpose((0,2,3,1))


	dit = unpickle('cifar-10/test_batch')
	data = dit['data']
	labels = dit['labels']

	test_data = []
	test_lb = np.zeros([len(labels), 10])	
	for i in range(len(data)):
		pp = np.reshape(data[i], [3,32,32])
		dd = np.zeros([3,224,224])
		for ii in range(3):
			for jj in range(32):
				for kk in range(32):
					dd[ii][jj][kk] = pp[ii][jj][kk]
		test_data.append(dd)
		test_lb[i][labels[i]] = 1
	test_data = np.array(test_data).transpose((0,2,3,1))

	print train_data.shape, train_lb.shape, test_data.shape, test_lb.shape
	print train_lb
	return train_data, train_lb, test_data, test_lb


if __name__ == '__main__':
    #save_array()
    #data_mat = np.concatenate(np.load(dt_prfx+'img_mat0.npy'), np.load(dt_prfx+'img_mat1.npy'))
    #lb_mat = np.concatenate(np.load(dt_prfx+'lb0.npy'), np.load(dt_prfx+'lb1.npy'))
    train_data, train_lb, test_data, test_lb = read_data()
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            res = construct_network(sess)
            run_model(res, sess, train_data, train_lb, test_data[0:200], test_lb[0:200])
