#%%
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow  as tf
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data

import keras



#%%
tf.reset_default_graph()


def dataset():
  return mnist.load_data()


(X_train,y_train), (X_test, y_test) = dataset()
mnists = input_data.read_data_sets("/tmp/data/")

#%%
from functools import partial
tf.reset_default_graph()
n_inputs = 784
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_hidden1


learning_rate = 0.01
l2_reg = .0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

he_init = tf.keras.initializers.he_normal(seed=None)
l2_regularizer = tf.keras.regularizers.l2(l2_reg)


my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu,
                         kernel_initializer=he_init,
                         kernel_regularizerr=l2_regularizer)


hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3 = my_dense_layer(hidden2, n_hidden3)
n_outputs = my_dense_layer(hidden3, n_outputs, activation=None)


loss = tf.losses.mean_squared_error(X,n_outputs)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)



loss_func = tf.summary.scalar("loss_func",loss)
merge     = tf.summary.merge_all()
writer = tf.summary.FileWriter("./mylog")
saver = tf.train.Saver()
init = tf.global_variables_initializer()

#%%

n_epochs=5
batch_size=150

with tf.Session() as sess:
  init.run()
  for epoch in range(n_epochs):
     
    n_batch = X_train.size//batch_size
    for iteration in range(n_batch):
      X_batch = X_train[iteration*n_batch:iteration*n_batch+batch_size]
      _,summary=sess.run([train_op,merge],feed_dict={X:X_batch})

    loss_train=loss.eval(feed_dict={X:X_batch,})
    print("\r{}".format(epoch),"Train MSE:",loss_train)
    writer.add_summary(summary)
    saver.save(sess, "./my_model_all_layers.ckpt")



