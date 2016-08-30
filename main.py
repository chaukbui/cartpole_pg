import os
import sys
import time

import tensorflow as tf
import numpy as np
import gym

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('render', False, '')

flags.DEFINE_integer('train_eps', 1000, '')
flags.DEFINE_integer('test_eps', 100, '')
flags.DEFINE_integer('eps_length', 200, '')

flags.DEFINE_integer('decay_steps', 200, '')
flags.DEFINE_float('lr', 0.1, '')
flags.DEFINE_float('lr_dec', 0.8, '')
flags.DEFINE_float('init_range', 0.1, '')

def build_graph():
  states = tf.placeholder(tf.float32, [None, 4])
  actions = tf.placeholder(tf.int64, [None])
  rewards = tf.placeholder(tf.float32, [None])

  initializer = tf.random_uniform_initializer(minval=-FLAGS.init_range, maxval=FLAGS.init_range)
  tf.get_variable_scope().set_initializer(initializer)

  w = tf.get_variable('w', [4, 2])
  logits = tf.matmul(states, w)
  probs = tf.nn.softmax(logits)
  probs = tf.reduce_sum(tf.one_hot(actions, 2) * probs, 1, keep_dims=False)

  action_op = {
      'sample': tf.multinomial(logits, 1),
      'greedy': tf.argmax(logits, 1),
  }

  loss = -tf.reduce_sum(probs * rewards)
  global_step = tf.Variable(0, dtype=tf.int64, name='global_step')
  learning_rate = tf.train.exponential_decay(FLAGS.lr,
                                             global_step,
                                             FLAGS.decay_steps,
                                             FLAGS.lr_dec,
                                             staircase=True) 
  train_op = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(loss, global_step)
  return states, actions, rewards, action_op, loss, \
      global_step, train_op, learning_rate, w


def run_episode(graph, sess, env, mode='sample'):
  states_ph, actions_ph, rewards_ph, action_op, loss, \
      global_step, train_op, learning_rate, w = graph

  all_states = []
  all_actions = []
  all_rewards = []

  state = env.reset()
  for step in xrange(FLAGS.eps_length):
    if FLAGS.render:
      env.render()
    state = np.reshape(state, [-1, 4])
    action = sess.run(action_op[mode], feed_dict={states_ph: state})
    action = np.reshape(action, [-1])[0]

    new_state, reward, done, info = env.step(action)
    all_states.append(state)
    all_actions.append(action)
    all_rewards.append(reward)

    state = new_state

    if done:
      break

  return all_states, all_actions, all_rewards


def train():
  graph  = build_graph()
  states_ph, actions_ph, rewards_ph, action_op, \
      loss, global_step, train_op, learning_rate, w = graph

  env = gym.make('CartPole-v0')

  sv = tf.train.Supervisor(logdir='output')

  with sv.managed_session() as sess:
    w_out = np.reshape(sess.run([w])[0], [-1])
    w_log = ''
    for w_out_i in w_out:
      w_log += '{:10f} '.format(w_out_i)
    print w_log

    print 'train'
    reward_baseline = 0
    for eps in xrange(FLAGS.train_eps):
      states, actions, rewards = run_episode(graph, sess, env)

      num_steps = len(states)
      if eps >= 100 and num_steps < 20:
        print 'fail'
        break

      if num_steps >= FLAGS.eps_length:
        print 'good'
        break

      states = np.concatenate(states, 0)
      actions = np.array(actions)
      rewards = np.arange(num_steps, 0, -1).astype(np.float32)

      outputs = sess.run([learning_rate, train_op], feed_dict={states_ph: states,
                                                               actions_ph: actions,
                                                               rewards_ph: rewards})

      reward_baseline = num_steps if eps == 0 else 0.99*reward_baseline + 0.01*num_steps
      log_string = ''
      log_string += 'eps {:4}:'.format(eps+1)
      log_string += ' lr={:.10f},'.format(outputs[0])
      log_string += ' reward={:3},'.format(num_steps)
      log_string += ' avg={:.3f},'.format(reward_baseline)
      print log_string

    print 'test'
    for eps in xrange(FLAGS.test_eps):
      states, _, _= run_episode(graph, sess, env, mode='greedy')
      print 'eps {:3}: {}'.format(eps+1, len(states))


def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
