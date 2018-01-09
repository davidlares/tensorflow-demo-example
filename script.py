from tensorflow.examples.tutorials.mnist import input_data # importamos data desde tensorflow (ejemplos)
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True) # definir que estamos trabajando con este dataset
x = tf.placeholder(tf.float32, [None, 784]) #creacion de una matriz de 0 vacia de 784 posiciones
W = tf.Variable(tf.zeros([784, 10])) #inicializacion de la matriz de Ceros
b = tf.Variable(tf.zeros([10])) #creacion de vector de 10 posiciones
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # reduccion y calculo de la media
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy) #script de entrenamiento
sess = tf.InteractiveSession() # repl de Tensorflow
tf.global_variables_initializer().run() # inicializacion de las variables globales

for _ in range(90000):
    batch_xs, batch_ys = mnist.train.next_batch(100) # genera dos batches en base a 100 partes o datos del dataset
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
result = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
print('El porcentaje de acierto es de: {}'.format(result))
