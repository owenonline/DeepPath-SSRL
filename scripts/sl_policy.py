from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from itertools import count
import sys

from networks import policy_nn
from utils import *
from env import Env
from BFS.KB import KB
from BFS.BFS import BFS
import time

tf.compat.v1.disable_eager_execution()

relation = sys.argv[1]
dataset = sys.argv[2]
graphpath = relation + 'graph.txt'#+ '/' + 'graph.txt'#dataPath + 'tasks/' + 
relationPath = relation + 'train_pos'#+ '/' + 'train_pos'#dataPath + 'tasks/' + 

class SupervisedPolicy(object):
	"""docstring for SupervisedPolicy"""
	def __init__(self, learning_rate = 0.001):
		self.initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
		with tf.compat.v1.variable_scope('supervised_policy'):
			self.state = tf.compat.v1.placeholder(tf.float32, [None, state_dim], name = 'state')
			self.action = tf.compat.v1.placeholder(tf.int32, [None], name = 'action')
			self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)

			action_mask = tf.cast(tf.one_hot(self.action, depth = action_space), tf.bool)
			self.picked_action_prob = tf.boolean_mask(tensor=self.action_prob, mask=action_mask)

			self.loss = tf.reduce_sum(input_tensor=-tf.math.log(self.picked_action_prob)) + sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, scope = 'supervised_policy'))
			self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate)
			self.train_op = self.optimizer.minimize(self.loss)

	def predict(self, state, sess = None):
		sess = sess or tf.compat.v1.get_default_session()
		return sess.run(self.action_prob, {self.state: state})

	def update(self, state, action, sess = None):
		sess = sess or tf.compat.v1.get_default_session()
		_, loss = sess.run([self.train_op, self.loss], {self.state: state, self.action: action})
		return loss

def train():
	# tf.reset_default_graph()
	policy_nn = SupervisedPolicy()

	print("relation path: {}".format(relationPath))
	print("graph path: {}".format(graphpath))

	f = open(relationPath)
	train_data = f.readlines()
	f.close()

	num_samples = len(train_data)

	saver = tf.compat.v1.train.Saver()
	with tf.compat.v1.Session() as sess:
		sess.run(tf.compat.v1.global_variables_initializer())
		if num_samples > 500:
			num_samples = 500
		else:
			num_episodes = num_samples

		for episode in range(num_samples):
			print("Episode %d" % episode)
			print('Training Sample:', train_data[episode%num_samples]) # [:-1])

			env = Env(dataset + "/", train_data[episode%num_samples])
			sample = train_data[episode%num_samples].split()

			try:
				good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
			except Exception as e:
				print('Cannot find a path')
				continue

			for item in good_episodes:
				state_batch = []
				action_batch = []
				for t, transition in enumerate(item):
					state_batch.append(transition.state)
					action_batch.append(transition.action)
				state_batch = np.squeeze(state_batch)
				state_batch = np.reshape(state_batch, [-1, state_dim])
				policy_nn.update(state_batch, action_batch)

		saver.save(sess, 'models/policy_supervised_rl_' + relation.split("/")[-2].replace(".","_") + "_" + dataset)
		print("model saved at models/policy_supervised_rl_" + relation.split("/")[-2].replace(".","_") + "_" + dataset)

if __name__ == "__main__":
	train()
	# test(50)

