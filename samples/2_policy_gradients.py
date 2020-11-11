############################################################
############################################################
############################################################
############################################################
############################################################
###################  WORK IN PROGRESS  #####################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################

import argparse
import imp
import os
import sys
import numpy as np
import random
import tensorflow as tf
import time
import cv2
from scipy.interpolate import interp1d
from scipy.misc import derivative

# import print_util as P

import vista

# Parse Arguments
parser = argparse.ArgumentParser(
    description='Train a new policy in VISTA using policy gradients')
parser.add_argument(
    '--trace-path',
    type=str,
    nargs='+',
    help='Path to the traces to use for simulation')
args = parser.parse_args()


def preprocess(image, camera):
    if image is not None:
        (i1, j1, i2, j2) = camera.get_roi()
        return cv2.resize(image[i1:i2, j1:j2], (192, 74))
    return image


class Model:
    def __init__(self, sess=tf.Session(), trainable=False):

        self.sess = sess

        self.tf_xs = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='camera')
        self.tf_ys = tf.placeholder(
            tf.float32, shape=[None, 1], name='control')
        self.tf_rs = tf.placeholder(
            tf.float32, shape=[None, 1], name='rewards')
        self.rewards_ep = tf.placeholder(
            tf.float32, shape=[None, 1], name='rewards_ep')
        self.distance_ep = tf.placeholder(
            tf.float32, shape=[None, 1], name='distance_ep')
        self.keep_prob = tf.placeholder(tf.float32)

        # Convenience functions with default parameters
        f_conv = functools.partial(
            tf.layers.conv2d, padding="valid", activation="relu")
        f_dense = functools.partial(tf.layers.dense, activation="relu")

        # batch color normalization
        x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),
            self.tf_xs)

        # convolutional layers
        c_params = [(24, 5, 2), (36, 5, 2), (48, 3, 2), (64, 3, 1), (64, 3, 1)]
        for filters, kernel, stride in c_params:
            x = f_conv(x, filters, kernel, stride)

        x = tf.layers.flatten(x)  # flatten

        # fully connected layers
        d_params = [1024, 512]
        for units in d_params:
            x = f_dense(x, units=units)
            x = tf.nn.dropout(x, self.keep_prob)

        # Output action distribution
        x = f_dense(x, units=2, activation=None)
        mu, log_sigma = tf.split(x, 2, axis=1)

        # Some constraints and numerical stability
        mu = 1 / 5. * tf.tanh(mu)
        sigma = 0.05 * tf.sigmoid(log_sigma) + 0.001

        # Sample and save relevant object attributes
        distribution = tf.distributions.Normal(mu, sigma)
        self.y_sample = distribution.sample()
        self.y_ = tf.identity(mu, name='prediction')

        if trainable:
            neg_log_prob = -1 * self.dist.log_prob(self.tf_ys)
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_rs)

            optimizer = tf.train.AdamOptimizer(5e-5)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.train_step = optimizer.apply_gradients(
                zip(gradients, variables))


    def init_summaries(self, logdir):
        # create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("neglogprob", tf.reduce_mean(self.neg_log_prob))
        tf.summary.scalar("rewards", tf.reduce_sum(self.tf_rs))
        tf.summary.scalar("ys", tf.reduce_mean(self.tf_ys))
        tf.summary.image("cropped image", self.x_image, max_outputs=1)
        self.rewards_ep_summary = tf.summary.scalar("rewards_ep",
                                                    tf.reduce_mean(
                                                        self.rewards_ep))
        self.merged_summary_op = tf.summary.merge_all()

        self.distances_ep_summary = tf.summary.scalar("distance_ep",
                                                      tf.reduce_mean(
                                                          self.distance_ep))

        self.summary_writer = tf.summary.FileWriter(
            logdir, graph=tf.get_default_graph())
        self.summary_iter = 0

    def init_saver(self):
        self.saver = tf.train.Saver()

    def inference(self, feed):
        y = self.sess.run(self.y_sample, feed_dict=feed)
        return y

    def write_summary(self, feed):
        summary = self.sess.run(self.merged_summary_op, feed_dict=feed)
        self.summary_writer.add_summary(summary, self.summary_iter)
        self.summary_iter += 1

    def write_checkpoint(self, path, name='model'):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, name)
        filename = self.saver.save(self.sess, checkpoint_path)

    def restore(self, path):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, path)


class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def discount_and_norm_rewards(self, gamma=0.95):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * gamma + self.rewards[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs.reshape((-1, 1))


memory = Memory()

sim = vista.Simulator(args.trace_path)

model = imp.load_source("model", 'model.py').Model(
    sess=sim.sess, trainable=True)

init_op = tf.compat.v1.variables_initializer([
    v for v in tf.compat.v1.global_variables()
    if not tf.compat.v1.is_variable_initialized(v).eval(session=sim.sess)
])
model.sess.run(init_op)

# Create save and log dirs
t = time.strftime("%y%m%d_%H%M%S")
save_dir = os.path.join('save', str(t) + '_' + "pg_delta_save")
log_dir = os.path.join('log', str(t) + '_' + "pg_delta_log")
if not os.path.exists(save_dir):  #create if not existing
    os.makedirs(save_dir)

model.init_saver()
model.init_summaries(log_dir)

step = 0.0
dist = 0.0
summary_steps = 0.0
best_overall = 0

print("\n\nStarting Training\n\n")

full_observation = sim.reset()
cropped_observation = preprocess(full_observation, sim.camera)
max_ep_reward = 0
reward_mean = None
alpha = 0.95
MAX_EPISODE_STEPS = 10000
MAX_EPISODE_DISTANCE = 10000  #m

crash, done = False, False
EVAL, FINAL = False, False

i_episode = 0
while i_episode < 1500:

    if crash and not done:
        # crashed before reaching the end of the trace
        i_episode += 1
        memory.clear()
        step = 0.0
        dist = 0.0
    else:
        # successfully completed the previous trace without crashing
        # just continue on a new part of the trace to continue the episode
        pass

    # current_env_index = reset_current_env(env_list)
    full_observation = sim.reset()
    cropped_observation = preprocess(full_observation, sim.camera)

    EVAL = True if i_episode % 10 == 0 or FINAL else False
    if EVAL: print("==== STARTING EVALUATION EPISODE ====")

    while True:

        feed = {
            model.tf_xs: [cropped_observation],
            model.keep_prob: 1.0,
        }
        if EVAL:
            action = model.sess.run(model.mu, feed)[0][0]
        else:
            action = model.inference(feed)[0][
                0]  #this is an action based on the previous observation

        next_full_observation, reward, crash, info = sim.step(
            action)  #step the previous frame with this action
        next_cropped_observation = preprocess(next_full_observation,
                                              sim.camera)

        memory.add_to_memory(cropped_observation, [action], reward)
        step += 1

        if crash:
            dist += info['distance']
            ep_rs_sum = sum(memory.rewards)  # np.sum(rs)

            reward_mean = alpha * reward_mean + (1 - alpha) * (
                ep_rs_sum) if reward_mean is not None else ep_rs_sum
            print(
                "episode: {:4.0f} \t step: {:4.0f} \t dist: {:5.1f} \t reward: {:7.2f} \t reward/step: {:5.2f} \t running {:5.2f}".
                format(i_episode, step, dist, ep_rs_sum, ep_rs_sum / step,
                       reward_mean))
            if EVAL:
                print("======================================")
                summary = model.sess.run(model.rewards_ep_summary, {
                    model.rewards_ep: [[ep_rs_sum]]
                })
                model.summary_writer.add_summary(summary, model.summary_iter)
                model.summary_iter += 1

                summary_steps += step
                model.summary_writer.add_summary(
                    model.sess.run(model.distances_ep_summary, {
                        model.distance_ep: [[dist]]
                    }), summary_steps)

                if ep_rs_sum > max_ep_reward:  # new best model obtained! save it!!
                    print("BEAT THE PREVIOUS BEST MODEL! SAVING A CHECKPOINT")
                    model.write_checkpoint(
                        save_dir, name="model-{}".format(int(ep_rs_sum)))

                    max_ep_reward = ep_rs_sum  # update the best ep reward for future
                    if FINAL:
                        print("Done!")
                        FINAL = False
                        break
                else:
                    FINAL = False

                break

            discounted_ep_rs_norm = memory.discount_and_norm_rewards()

            # train on episode
            feed = {
                model.tf_xs:
                np.array(memory.observations),  # shape=[None, 74, 192, 3]
                model.tf_ys: np.array(memory.actions),  # shape=[None, 1]
                model.tf_rs: discounted_ep_rs_norm,  # shape=[None, 1]
                model.rewards_ep: [[ep_rs_sum]],
                model.keep_prob: 1.0,
            }

            model.sess.run(model.train_step, feed)
            model.write_summary(feed)
            summary_steps += step
            model.summary_writer.add_summary(
                model.sess.run(model.distances_ep_summary, {
                    model.distance_ep: [[dist]]
                }), summary_steps)

            memory.clear()
            break

        if dist > MAX_EPISODE_DISTANCE:
            print("Beat max episode steps!")
            i_episode += 1
            summary = model.sess.run(model.rewards_ep_summary, {
                model.rewards_ep: [[ep_rs_sum]]
            })
            model.summary_writer.add_summary(summary, model.summary_iter)
            model.summary_iter += 1
            summary_steps += step
            model.summary_writer.add_summary(
                model.sess.run(model.distances_ep_summary, {
                    model.distance_ep: [[dist]]
                }), summary_steps)

            if FINAL:
                print("Writing checkpoint")
                model.write_checkpoint(save_dir, name="model-FINAL")
                FINAL = False
                break

            memory.clear()
            step = 0.0
            dist = 0.0
            FINAL = True
            break

        full_observation = next_full_observation
        cropped_observation = next_cropped_observation
