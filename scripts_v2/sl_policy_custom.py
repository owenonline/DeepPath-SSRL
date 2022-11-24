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
# episodes = int(sys.argv[2])
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'

class SupervisedPolicy(object):
	"""docstring for SupervisedPolicy"""
	def __init__(self, learning_rate = 0.001):
		self.initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
		with tf.compat.v1.variable_scope('supervised_policy'):
			self.state = tf.compat.v1.placeholder(tf.float32, [None, state_dim], name = 'state')
			self.action = tf.compat.v1.placeholder(tf.int32, [None], name = 'action')
			self.correct = tf.compat.v1.placeholder(tf.float32, [None, action_space], 'correct')
			self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)

			action_mask = tf.cast(tf.one_hot(self.action, depth = action_space), tf.bool)
			self.picked_action_prob = tf.boolean_mask(tensor=self.action_prob, mask=action_mask)

			self.loss = tf.compat.v1.keras.losses.categorical_crossentropy(self.correct, self.action_prob)
			self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate)
			self.train_op = self.optimizer.minimize(self.loss)

	def predict(self, state, sess = None):
		sess = sess or tf.compat.v1.get_default_session()
		return sess.run(self.action_prob, {self.state: state})

	def update(self, state, correct, sess = None):
		sess = sess or tf.compat.v1.get_default_session()
		_, loss = sess.run([self.train_op, self.loss], {self.state: state, self.correct: correct})
		return loss

def normalize_probs(probs):
	probs = tf.cast(probs, dtype=tf.float32)
	probs = tf.divide(tf.subtract(probs, tf.reduce_min(input_tensor=probs)), tf.subtract(tf.reduce_max(input_tensor=probs), tf.reduce_min(input_tensor=probs)))

	# if a row in scores is all 0s, change it to a very small nonzero value instead so cce can't produce nans
	if tf.math.reduce_min(input_tensor=tf.math.count_nonzero(probs, axis=1)) == 0:
		probs = probs + tf.fill(tf.shape(input=probs), 0.0001)

	return probs

def train():
	tf.compat.v1.reset_default_graph()
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

			env = Env(dataPath, train_data[episode%num_samples])
			sample = train_data[episode%num_samples].split()

			correct_path = label_gen(sample[0], sample[1], kb, env)

			last_step = [("N/A",)]
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
						valid = correct_path[t][last_step[batch_num]]
					except:
						valid = env.backtrack(sample[0], kb)

					# if no paths were found, set the label equal to the score so nothing gets changed
					if len(valid) == 1 and valid[0] == -1:
						correct[np.array([batch_num]*len(valid), int), :] = normalized_action_probs[batch_num]
					else:
						correct[np.array([batch_num]*len(valid), int), np.array(valid, int)] = np.ones(len(valid))

				current_actions = action_chosen
				last_step = [tuple(list(x) + [y]) for (x, y) in zip(last_step, [current_actions])]

				# update agent weights
				# correct = tf.convert_to_tensor(value=correct)
				policy_nn.update(state_vec, correct)
				
				_, new_state, done = env.interact(state_idx, action_chosen)
				if done or t == 2:
					if done:
						print('Success')
						success += 1
					print('Episode ends\n')
					break
				state_idx = new_state

if __name__ == "__main__":
	train()