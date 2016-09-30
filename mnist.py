import tensorflow as tf
import tf_lib as tfl
from default import Default

input_size = 784
output_size = 10

class SimpleModel(Default):
  options = [
    ('hidden_sizes', [100]),
  ]
  
  def __init__(self, **kwargs):
    super(SimpleModel, self).__init__(**kwargs)
    
    self.net = tfl.Sequential()

    prev_size = input_size

    nl = tfl.leaky_softplus()
    for next_size in self.hidden_sizes:
      self.net.append(tfl.FCLayer(prev_size, next_size, nl))
      prev_size = next_size

    self.net.append(tfl.FCLayer(prev_size, output_size, tf.nn.log_softmax))

  def predict(self, x):
    return self.net(x)

class SimpleTrainer(Default):
  options = [
    ('learning_rate', 1e-2)
  ]
  
  def train_op(self, _, loss, global_step):
    return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss, global_step=global_step)

class NaturalGradientTrainer(Default):
  options = [
    ('learning_rate', 1e-1),
    ('damping', 1e-3),
    #('max_distance', 1e-3),
  ]
  
  def train_op(self, log_probs, loss, global_step):
    params = tf.trainable_variables()
    
    grad = tf.gradients(-self.learning_rate * loss, params)
    
    import natgrad
    nat_grad = natgrad.natural_gradients(params, grad, log_probs, tfl.kl, **dict(self.items()))
    
    ops = [tf.assign_add(global_step, 1)]
    for param, ng in zip(params, nat_grad):
      ops.append(tf.assign_add(param, ng))
    
    return tf.group(*ops)

def options_str(obj):
  return "_".join([name + '_' + str(value) for name, value in options])

def run(model = SimpleModel, trainer=SimpleTrainer, batch_size=100, steps=50, iters=100, **kwargs):
  graph = tf.Graph()
  
  with graph.as_default():
    x = tf.placeholder(tf.float32, [None, input_size])
    y_ = tf.placeholder(tf.float32, [None, output_size])
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    model_ = model(**kwargs)

    log_probs = model_.predict(x)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(log_probs * y_, -1))

    tf.scalar_summary("cross_entropy", cross_entropy)

    trainer_ = trainer(**kwargs)
    train_op = trainer_.train_op(log_probs, cross_entropy, global_step)

    merged = tf.merge_all_summaries()
    run_dict = dict(summary=merged, train=train_op, global_step=global_step)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
  
  label = "_".join([model_.label(), trainer_.label()])
  print(label)

  writer = tf.train.SummaryWriter('logs/%s/' % label, sess.graph)

  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  import time

  for i in range(iters):
    start = time.time()
    for _ in range(50):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
    
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    results = sess.run(run_dict, feed_dict={x: batch_xs, y_: batch_ys})
    
    t = time.time() - start
    print("step %d" % i, t)
    
    writer.add_summary(results['summary'], results['global_step'])

if __name__ == "__main__":
  #run()
  #run(trainer=NaturalGradientTrainer)
  run(trainer=NaturalGradientTrainer, learning_rate=1.0, damping=0.0)
  
