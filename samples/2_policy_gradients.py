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
parser = argparse.ArgumentParser(description='Train a new policy in VISTA using policy gradients')
parser.add_argument('--trace-path', type=str, nargs='+', help='Path to the traces to use for simulation')
args = parser.parse_args()


def preprocess(image, camera):
    if image is not None:
        (i1,j1,i2,j2) = camera.get_roi()
        return cv2.resize(image[i1:i2, j1:j2], (192, 74))
    return image

def discount_rewards(r):
    gamma = .9
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] == 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    normalized_r = (discounted_r)/discounted_r.var()
    return normalized_r

def _discount_and_norm_rewards(rs, gamma=0.95):
    # discount episode rewards
    discounted_ep_rs = np.zeros_like(rs)
    running_add = 0
    for t in reversed(range(0, len(rs))):
        running_add = running_add * gamma + rs[t]
        discounted_ep_rs[t] = running_add

    # normalize episode rewards
    discounted_ep_rs -= np.mean(discounted_ep_rs)
    discounted_ep_rs /= np.std(discounted_ep_rs)
    return discounted_ep_rs.reshape((-1,1))


sim = vista.Simulator(args.trace_path)

model = imp.load_source("model", 'model.py').Model(sess=sim.sess, trainable=True)

camera_obj = sim.camera
# init_op = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in set(model.sess.run(tf.report_uninitialized_variables()))])
init_op = tf.compat.v1.variables_initializer([v for v in tf.compat.v1.global_variables() if not tf.compat.v1.is_variable_initialized(v).eval(session=sim.sess)])
model.sess.run(init_op)

# Create save and log dirs
t = time.strftime("%y%m%d_%H%M%S")
save_dir = os.path.join('save', str(t) +'_'+ "pg_delta_save")
log_dir = os.path.join('log', str(t) +'_'+ "pg_delta_log")
if not os.path.exists(save_dir): #create if not existing
    os.makedirs(save_dir)
# if not os.path.exists(CONFIG["train"].save_dir): #create if not existing
#     os.makedirs(CONFIG["train"].save_dir)

model.init_saver()
model.init_summaries(log_dir)

xs = [] # state_t
ys = [] # action_t
ys_prev = [] # action_{t-1}
rs = [] # reward_t
action_prev = 0.0 # previous action
step = 0.0
dist = 0.0
summary_steps = 0.0
best_overall = 0

batch_size = 24 # CONFIG['train'].mem_buffer_batch_size

print("\n\nStarting Training\n\n")

full_observation = sim.reset()
cropped_observation = preprocess(full_observation, camera_obj)
max_ep_reward = 0
reward_mean = None
alpha = 0.95
MAX_EPISODE_STEPS = 10000
MAX_EPISODE_DISTANCE = 10000 #m

crash, done = False, False
EVAL, FINAL = False, False

i_episode = 0
while i_episode < 1500 :

    if crash and not done:
        # crashed before reaching the end of the trace
        i_episode += 1
        xs, ys, ys_prev, rs = [], [], [], []    # empty episode data
        action_prev = 0.0
        step = 0.0
        dist = 0.0
    else:
        # successfully completed the previous trace without crashing
        # just continue on a new part of the trace to continue the episode
        pass

    # current_env_index = reset_current_env(env_list)
    full_observation = sim.reset()
    cropped_observation = preprocess(full_observation, camera_obj)

    EVAL = True if i_episode % 10 == 0 or FINAL else False
    if EVAL: print("==== STARTING EVALUATION EPISODE ====")

    while True:

        feed = {
            model.tf_xs: [cropped_observation],
            model.tf_ys_prev: [[action_prev]],
            model.keep_prob: 1.0,
        }
        if EVAL:
            action = model.sess.run(model.mu, feed)[0][0]
        else:
            action = model.inference(feed)[0][0]  #this is an action based on the previous observation


        next_full_observation, reward, crash, info = sim.step(action) #step the previous frame with this action
        next_cropped_observation = preprocess(next_full_observation, camera_obj)

        xs.append(cropped_observation)
        ys.append([action])
        ys_prev.append([action_prev])
        rs.append(reward)
        step += 1

        if crash:
            dist += info['distance']
            rs = np.array(rs)
            ep_rs_sum = np.sum(rs)

            reward_mean = alpha*reward_mean+(1-alpha)*(ep_rs_sum) if reward_mean is not None else ep_rs_sum
            print("episode: {:4.0f} \t step: {:4.0f} \t dist: {:5.1f} \t reward: {:7.2f} \t reward/step: {:5.2f} \t running {:5.2f}".format(i_episode, step, dist, ep_rs_sum, ep_rs_sum/step, reward_mean))
            if EVAL:
                print("======================================")
                summary = model.sess.run(model.rewards_ep_summary, {model.rewards_ep: [[ep_rs_sum]]})
                model.summary_writer.add_summary(summary, model.summary_iter)
                model.summary_iter += 1

                summary_steps += step
                model.summary_writer.add_summary( model.sess.run(model.distances_ep_summary, {model.distance_ep: [[dist]]}), summary_steps)

                if ep_rs_sum > max_ep_reward: # new best model obtained! save it!!
                    print("BEAT THE PREVIOUS BEST MODEL! SAVING A CHECKPOINT")
                    model.write_checkpoint(save_dir, name="model-{}".format(int(ep_rs_sum)))

                    max_ep_reward = ep_rs_sum # update the best ep reward for future
                    if FINAL:
                        print("Done!")
                        FINAL = False
                        break
                else:
                    FINAL = False

                break

            discounted_ep_rs_norm = _discount_and_norm_rewards(rs)

            xs = np.stack(xs)
            ys = np.array(ys)
            ys_prev = np.array(ys_prev)
            if len(xs) > 5000:
                idx = np.random.choice(xs.shape[0], 5000)
                xs = xs[idx]
                ys = ys[idx]
                ys_prev = ys_prev[idx]
                discounted_ep_rs_norm = discounted_ep_rs_norm[idx]

            # train on episode
            feed = {
                model.tf_xs: xs,  # shape=[None, 74, 192, 3]
                model.tf_ys: ys,  # shape=[None, 1]
                model.tf_ys_prev: ys_prev,
                model.tf_rs: discounted_ep_rs_norm,  # shape=[None, 1]
                model.rewards_ep: [[ep_rs_sum]],
                model.keep_prob: 1.0,
            }

            model.sess.run(model.train_step, feed)
            model.write_summary(feed)
            summary_steps += step
            model.summary_writer.add_summary( model.sess.run(model.distances_ep_summary, {model.distance_ep: [[dist]]}), summary_steps)

            xs, ys, ys_prev, rs = [], [], [], []    # empty episode data
            break

        if dist > MAX_EPISODE_DISTANCE:
            print("Beat max episode steps!")
            i_episode += 1
            summary = model.sess.run(model.rewards_ep_summary, {model.rewards_ep: [[ep_rs_sum]]})
            model.summary_writer.add_summary(summary, model.summary_iter)
            model.summary_iter += 1
            summary_steps += step
            model.summary_writer.add_summary( model.sess.run(model.distances_ep_summary, {model.distance_ep: [[dist]]}), summary_steps)

            if FINAL:
                print("Writing checkpoint")
                model.write_checkpoint(save_dir, name="model-FINAL")
                FINAL = False
                break
                # exit()

            # solved the enviornment (?)
            xs, ys, ys_prev, rs = [], [], [], []    # empty episode data
            step = 0.0
            dist = 0.0
            FINAL = True; break

        action_prev = action
        full_observation = next_full_observation
        cropped_observation = next_cropped_observation
