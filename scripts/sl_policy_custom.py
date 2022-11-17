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

relation = sys.argv[1]
# episodes = int(sys.argv[2])
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

class SupervisedPolicy(object):
	"""docstring for SupervisedPolicy"""
	def __init__(self, learning_rate = 0.001):
		self.initializer = tf.contrib.layers.xavier_initializer()
		with tf.variable_scope('supervised_policy'):
			self.state = tf.placeholder(tf.float32, [None, state_dim], name = 'state')
			self.action = tf.placeholder(tf.int32, [None], name = 'action')
			self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)

			action_mask = tf.cast(tf.one_hot(self.action, depth = action_space), tf.bool)
			self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)

			self.correct = tf.placeholder(tf.float32, [None], 'correct')
			self.predicted = tf.placeholder(tf.float32, [None], 'predicted')
			self.loss = loss(self.correct, self.predicted)
			self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
			self.train_op = self.optimizer.minimize(self.loss)

	def predict(self, state, sess = None):
		sess = sess or tf.get_default_session()
		return sess.run(self.action_prob, {self.state: state})

	def update(self, correct, predicted, sess = None):
		sess = sess or tf.get_default_session()
		_, loss = sess.run([self.train_op, self.loss], {self.correct: correct, self.predicted: predicted})
		return loss

def label_gen(e1, e2, num_paths, env, path = None):
    raise NotImplementedError

def normalize_probs(probs):
	probs = tf.cast(probs, dtype=tf.float32)
	probs = tf.divide(tf.subtract(probs, tf.reduce_min(probs)), tf.subtract(tf.reduce_max(probs), tf.reduce_min(probs)))

	# if a row in scores is all 0s, change it to a very small nonzero value instead so cce can't produce nans
	if tf.math.reduce_min(tf.math.count_nonzero(probs, axis=1)).numpy() == 0:
		probs = probs + tf.fill(tf.shape(probs), 0.0001)

	return probs

def train():
	tf.reset_default_graph()
	policy_nn = SupervisedPolicy()

	print("relation path: {}".format(relationPath))
	print("graph path: {}".format(graphpath))

	# get relations
	f = open(relationPath)
	train_data = f.readlines()
	f.close()

	# get knowledge graph
	f = open(graphpath)
	content = f.readlines()
	f.close()
	kb = KB()
	for line in content:
		ent1, rel, ent2 = line.rsplit()
		kb.addRelation(ent1, rel, ent2)

	num_samples = len(train_data)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if num_samples > 500:
			num_samples = 500
		else:
			num_episodes = num_samples

		for episode in range(num_samples):
			print("Episode %d" % episode)
			print('Training Sample:', train_data[episode%num_samples]) # [:-1])

			env = Env(dataPath, train_data[episode%num_samples])
			sample = train_data[episode%num_samples].split()

			try:
				correct_path = label_gen(sample[0], sample[1], kb, env)
			except Exception as e:
				print('Cannot find a path')
				continue

			state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
			for t in count():
				state_vec = env.idx_state(state_idx)
				action_probs = policy_nn.predict(state_vec)
				action_chosen = np.random.choice(np.arange(action_space), p = np.squeeze(action_probs))

				# supervised learning magic
				normalized_action_probs = normalize_probs(action_probs)
				active_length = normalized_action_probs.shape[0]
				choices = normalized_action_probs.shape[1]

				correct = np.full((active_length,choices),0)

				for batch_num in range(len(correct_path[t])):
					try:
						valid = correct_path[t][batch_num][last_step[batch_num]]
					except:
						valid = env.backtrack(sample[0], kb)

					# if no paths were found, set the label equal to the score so nothing gets changed
					if len(valid) == 1 and valid[0] == -1:
						correct[np.array([batch_num]*len(valid), int), :] = normalized_action_probs[batch_num]
					else:
						correct[np.array([batch_num]*len(valid), int),np.array(valid, int)] = np.ones(len(valid))

				current_actions = action_chosen.numpy()
				last_step = [tuple(list(x) + [y]) for (x, y) in zip(last_step, current_actions)]

				# update agent weights
				correct = tf.convert_to_tensor(correct)
				policy_nn.update(correct, action_probs)
				
				_, new_state, done = env.interact(state_idx, action_chosen)
				if done or t == 3:
					if done:
						print('Success')
						success += 1
					print('Episode ends\n')
					break
				state_idx = new_state