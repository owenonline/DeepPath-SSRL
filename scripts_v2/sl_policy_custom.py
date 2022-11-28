from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np
from sklearn.preprocessing import normalize
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
			self.correct = tf.compat.v1.placeholder(tf.float32, [None, action_space], 'correct')
			self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)

			self.action_prob = normalize_probs(self.action_prob)

			self.loss = tf.compat.v1.keras.losses.categorical_crossentropy(self.correct, self.action_prob)
			self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate)
			self.train_op = self.optimizer.minimize(self.loss)

	def update(self, state, correct, sess = None):
		sess = sess or tf.compat.v1.get_default_session()
		_, _, action_prob = sess.run([self.train_op, self.loss, self.action_prob], {self.state: state, self.correct: correct})
		return action_prob

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
	success = 0

	saver = tf.compat.v1.train.Saver()
	with tf.compat.v1.Session() as sess:
		sess.run(tf.compat.v1.global_variables_initializer())
		
		if num_samples > 500:
			num_samples = 500

		for episode in range(num_samples):
			print("Episode %d" % episode)
			print('Training Sample:', train_data[episode%num_samples]) # [:-1])

			env = Env(dataPath, train_data[episode%num_samples])
			sample = train_data[episode%num_samples].split()

			correct_path = label_gen(sample[0], sample[1], kb, env)

			last_step = ("N/A",)
			state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
			last_state_idx = None
			for t in count():
				state_vec = env.idx_state(state_idx)

				correct = np.full((1, action_space), 0)

				try:
					valid = np.histogram(correct_path[t][last_step], bins=action_space, range=(0, action_space), density=True)[0]

					if len(valid) == 1 and valid[0] == -1:
						print("no correct paths found, breaking")
						break
					else:
						print("working")
						correct[0, :] = valid
				except:
					# state_idx = last_state_idx
					# last_step = last_step[:-1]
					# t -= 2
					# continue
					print("agent has entered unrecoverable state, breaking")
					break

				# if training results are undesirable, try putting a loop here that runs the update through action choosing lines until an action in the label is picked

				# update agent weights
				action_prob = policy_nn.update(state_vec, correct)
				normalized_probs = normalize(action_prob, norm="l1")[0]

				# select action based on agent output
				action_chosen = int(np.random.choice(np.arange(action_space), 1, p=normalized_probs))
				last_step = tuple(list(last_step) + [action_chosen])
				
				_, new_state, done = env.interact(state_idx, action_chosen)

				if done or t == 2:
					print("path: {}".format(env.path))
					if done:
						print('Success')
						success += 1
					print('Episode ends\n')
					break
				last_state_idx = state_idx
				state_idx = new_state

if __name__ == "__main__":
	train()