import numpy as np
import tensorflow as tf

# list of the bandits. Bandit #4 wins currently most often
bandits = [0.3, 0.4, 0.9, 0.4]  # bandit probabilities of winning
num_bandits = len(bandits)


def pullBandit(bandit):
    rand = np.random.random()
    if bandit > rand:
        # return a positive reward
        return 1
    else:
        # return a negative reward
        return -1


tf.reset_default_graph()

# these two lines establish the feed-forward part of the network. Not the actual choosing
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights)

# the following paragraph establishes the training procedure. Reward and chosen action are fed into ... to compute the loss and use that to update
# the NN
reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)
action_holder = tf.placeholder(shape = [1], dtype = tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
update = optimizer.minimize(loss)

total_episodes = 20  # amount of episodes to train on
total_reward = np.zeros(num_bandits)
print("Total Reward at begin: ", total_reward)

# Execute Computation Graph
sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)
random_action_prob = 0.1  # CHange that a random action occurs
i = 0
while i < total_episodes:
    # choose either a random action or one from our network
    if np.random.rand(1) > random_action_prob:
        action = sess.run(chosen_action)
    else:
        action = np.random.randint(num_bandits)

    reward = pullBandit(bandits[action])  # get the reward for picking one of the bandits

    # update the network
    _, resp, ww = sess.run([update, responsible_weight, weights], feed_dict = {reward_holder: [reward], action_holder: [action]})
    # update our running tally of scores
    total_reward[action] += reward

    i += 1
    print("Total reward after ", i, " episodes: ", total_reward)

print("\nThe agent guessed the best bandit is: ", np.argmax(ww) + 1)
print("The known best bandit is: ", np.argmax(bandits) + 1)
