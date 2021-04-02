import argparse
import functools
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

import vista

# This script trains an agent using vanilla policy gradients from scratch within
# the VISTA environment. The script is split up into three main parts:
#
# 1. Model: The model contains the brain of the agent. It is the neural network
#   that predicts steering given a raw visual input.
#
# 2. Memory: The memory buffer contains a playback of observations, actions
#   from the previous episode so they can be recalled for training.
#
# 3. Trainer: The main() method of the script which instantiates the agent,
#   simulator, and memory. These are then all used within the policy gradient
#   training loop to optimize the agent.
#


def main(args):

    # Initialize VISTA, the model, and a memory for training
    world = vista.World(args.trace_path)
    agent = world.spawn_agent()
    sensor = agent.spawn_camera()
    display = vista.Display(world)

    model = Model(trainable=True)
    memory = Memory()

    # Some tracking variables for training status info
    step = 0.0
    dist = 0.0
    summary_steps = 0.0
    best_overall = 0
    max_ep_reward = 0.0
    reward_mean = None
    crash, done = False, False
    EVAL, FINAL = False, False

    print("\n\nStarting Training\n\n")
    full_observation = agent.reset()
    cropped_observation = preprocess(full_observation, sensor)
    i_episode = 0

    # Reset and start training
    while i_episode < args.max_episodes:

        if crash and not done:
            # Crashed before reaching the end of the trace
            i_episode += 1
            memory.clear()
            step = 0.0
            dist = 0.0
        else:
            # Successfully completed the previous trace without crashing
            # Just continue on a new part of the trace to continue the episode
            pass

        full_observation = agent.reset()
        cropped_observation = preprocess(full_observation, sensor)

        EVAL = True if i_episode % 10 == 0 or FINAL else False
        if EVAL: print("==== STARTING EVALUATION EPISODE ====")

        # Run the agent for a single episode (until crashing)
        while True:
            # Compute an action from the policy
            feed = {
                model.tf_xs: [cropped_observation],
                model.keep_prob: 1.0,
            }
            action = model.compute_action(feed, EVAL)[0][0]
            next_full_observation, crash = agent.step(action)
            display.render()

            next_cropped_observation = preprocess(next_full_observation,
                                                  sensor)

            # Add to memory
            reward = 1.0
            memory.add_to_memory(cropped_observation, [action], reward)
            step += 1

            # If there was a crash, episode is over. Train with the resulting
            # data if it came from a explorative run (i.e. EVAL==False)
            if crash:
                dist += agent.distance
                ep_rs_sum = sum(memory.rewards)

                reward_mean = args.alpha * reward_mean + (1 - args.alpha) * (
                    ep_rs_sum) if reward_mean is not None else ep_rs_sum

                print(f"episode: {i_episode:4.0f}\t " +
                      f"dist: {dist:5.1f}\t " +
                      f"reward: {ep_rs_sum:7.2f}\t " +
                      f"running: {reward_mean:5.2f}")

                if EVAL:
                    print("=====================================")
                    summary = model.sess.run(model.rewards_ep_summary,
                                             {model.rewards_ep: [[ep_rs_sum]]})
                    model.summary_writer.add_summary(summary,
                                                     model.summary_iter)
                    model.summary_iter += 1

                    summary_steps += step
                    model.summary_writer.add_summary(
                        model.sess.run(model.distances_ep_summary,
                                       {model.distance_ep: [[dist]]}),
                        summary_steps)

                    # New best model obtained! Save it!
                    if ep_rs_sum > max_ep_reward:
                        print("BEAT THE PREVIOUS BEST MODEL! SAVING!")
                        model.write_checkpoint(name=f"model-{int(ep_rs_sum)}")

                        # update the best ep reward for future
                        max_ep_reward = ep_rs_sum
                        if FINAL:
                            print("Done!")
                            FINAL = False
                            break
                    else:
                        FINAL = False

                    break

                # Discount the rewards, and train on the episode
                discounted_ep_rs_norm = memory.discount_and_norm_rewards()
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
                    model.sess.run(model.distances_ep_summary,
                                   {model.distance_ep: [[dist]]}),
                    summary_steps)

                memory.clear()
                break

            if dist > args.max_distance:
                print("Beat max episode steps!")
                i_episode += 1
                summary = model.sess.run(model.rewards_ep_summary,
                                         {model.rewards_ep: [[ep_rs_sum]]})
                model.summary_writer.add_summary(summary, model.summary_iter)
                model.summary_iter += 1
                summary_steps += step
                model.summary_writer.add_summary(
                    model.sess.run(model.distances_ep_summary,
                                   {model.distance_ep: [[dist]]}),
                    summary_steps)

                if FINAL:
                    print("Writing checkpoint")
                    model.write_checkpoint(name="model-FINAL")
                    FINAL = False
                    break

                memory.clear()
                step = 0.0
                dist = 0.0
                FINAL = True
                break

            full_observation = next_full_observation
            cropped_observation = next_cropped_observation


# The model contains the brain of the agent. It takes as input an observation,
# and outputs the action that the agent should execute at that instant.
class Model:
    def __init__(self, trainable=False):

        self.sess = tf.compat.v1.Session()
        self.tf_xs = tf.placeholder(tf.float32,
                                    shape=[None, 74, 192, 3],
                                    name='camera')
        self.tf_ys = tf.placeholder(tf.float32,
                                    shape=[None, 1],
                                    name='control')
        self.tf_rs = tf.placeholder(tf.float32,
                                    shape=[None, 1],
                                    name='rewards')
        self.rewards_ep = tf.placeholder(tf.float32,
                                         shape=[None, 1],
                                         name='rewards_ep')
        self.distance_ep = tf.placeholder(tf.float32,
                                          shape=[None, 1],
                                          name='distance_ep')
        self.keep_prob = tf.placeholder(tf.float32)

        # Convenience functions with default parameters
        f_conv = functools.partial(tf.layers.conv2d,
                                   padding="valid",
                                   activation="relu")
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
            self.neg_log_prob = -1 * distribution.log_prob(self.tf_ys)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_rs)

            optimizer = tf.train.AdamOptimizer(5e-5)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.train_step = optimizer.apply_gradients(
                zip(gradients, variables))

        # Initialize all unintialized variables
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self._init_summaries()

    def compute_action(self, feed, greedy):
        tensor = self.y_ if greedy else self.y_sample
        return self.sess.run(tensor, feed)

    def _init_summaries(self):
        self.saver = tf.train.Saver()

        # Create save and log dirs
        t = time.strftime("%y%m%d_%H%M%S")
        self.save_dir = os.path.join('save', str(t) + '_' + "pg_delta_save")
        self.log_dir = os.path.join('log', str(t) + '_' + "pg_delta_log")
        if not os.path.exists(self.save_dir):  #create if not existing
            os.makedirs(self.save_dir)

        # create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("neglogprob", tf.reduce_mean(self.neg_log_prob))
        tf.summary.scalar("rewards", tf.reduce_sum(self.tf_rs))
        tf.summary.scalar("ys", tf.reduce_mean(self.tf_ys))
        self.rewards_ep_summary = tf.summary.scalar(
            "rewards_ep", tf.reduce_mean(self.rewards_ep))
        self.merged_summary_op = tf.summary.merge_all()

        self.distances_ep_summary = tf.summary.scalar(
            "distance_ep", tf.reduce_mean(self.distance_ep))

        self.summary_writer = tf.summary.FileWriter(
            self.log_dir, graph=tf.get_default_graph())
        self.summary_iter = 0

    def write_summary(self, feed):
        summary = self.sess.run(self.merged_summary_op, feed_dict=feed)
        self.summary_writer.add_summary(summary, self.summary_iter)
        self.summary_iter += 1

    def write_checkpoint(self, name='model'):
        path = self.save_dir
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, name)
        filename = self.saver.save(self.sess, checkpoint_path)


# The memory is a helper class that keeps track of some important variables
# during training, such as the episodes observations, actions, and rewards.
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


def preprocess(observation, camera):
    image = observation[camera.id]
    if image is not None:
        (i1, j1, i2, j2) = camera.camera.get_roi()
        return image[i1:i2, j1:j2]
    return image


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Train a new policy in VISTA using policy gradients')
    parser.add_argument('--trace-path',
                        type=str,
                        nargs='+',
                        help='Path to the traces to use for simulation')
    parser.add_argument(
        '--max-distance',
        type=int,
        default=10000,
        help='Distance in [m] to drive without crashing to "master" the scene')
    parser.add_argument(
        '--max_episodes',
        type=int,
        default=1500,
        help='Number of episodes to train with (stop earlier if agents masters)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.95,
        help='Smoothing alpha for the exponential filter when reporting reward'
    )
    args = parser.parse_args()

    main(args)
