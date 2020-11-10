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
#######################
root = os.environ.get('DEEPKNIGHT_ROOT')

sys.path.insert(0,os.path.join(root,'simulator'))
from gym_deepknight.envs import DeepknightEnv

sys.path.insert(0,os.path.join(root,'util'))
from Camera import Camera
import Image
import print_util as P

parser = argparse.ArgumentParser(description='train sim')
parser.add_argument('--trace', default=[], nargs='+')

args = parser.parse_args()

INPUT_IMG_SHAPE = (250, 400, 3)

def preprocess(image, camera):
    if image is not None:
        roi = camera.get_roi()
        image = Image.crop(image, roi)
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

def reset_current_env(env_list):
    steps_per_env = np.array([env.num_steps for env in env_list]).astype(np.float)
    return np.random.choice(len(env_list), p=steps_per_env/np.sum(steps_per_env))



print P.INFO("Initializing configuration permutations")
# ALL_TRAIN_CONFIGS = imp.load_source("train",os.path.join(root,'config/train/multi_run.py')).Config().startup()
ALL_TRAIN_CONFIGS = imp.load_source("train",os.path.join(root,'config/train/sim_test_aa.py')).Config().startup()
total_permutations = len(ALL_TRAIN_CONFIGS)

# t_start = time.strftime("%y%m%d_%H%M%S")
# file = open(os.path.join(root,'simulator/rl_stuff/next/meta_logs',t_start + '_meta_log.txt'), 'w+')
# t = time.strftime("%y-%m-%d at %H:%M.%S")
# file.write("Meta trainer for sim_delta_train_memory on "+t)
# file.write("\n")
# file.close()

print P.INFO("There are {} config permutations".format(total_permutations))
best_overall = 0

for a in range(0, total_permutations):
    # with open(os.path.join(root,'simulator/rl_stuff/next/meta_logs', t_start + '_meta_log.txt'), 'a') as f:
    #     f.write("with config {}".format(ALL_TRAIN_CONFIGS[a].__dict__))
    #     # f.write("with config {}".format())
    #     f.close()
    print("\n")
    print P.INFO("Running permutation {} of {} ".format(a+1,total_permutations))

    CONFIG = {
        'sim':      imp.load_source("sim_config", os.path.join(root,'config/simulator/default.py')).Config(),
        'camera':   imp.load_source("cam_config", os.path.join(root,'config/camera/default.py')).Config(),
        'train':   ALL_TRAIN_CONFIGS[a] #imp.load_source("train_config", os.path.join(root,'config/train/default.py')).Config()
    }


    trace_list = args.trace

    tf.reset_default_graph()
    env = DeepknightEnv(trace_list, obs_size=INPUT_IMG_SHAPE[:2])
    model = imp.load_source("model", os.path.join(root, 'models/rl', 'single_camera_gaussian_delta_steering_aa.py')).Model(sess=env.sess, trainable=True, CONFIG=CONFIG['train'])
    camera_obj = env.camera

    init_op = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in set(model.sess.run(tf.report_uninitialized_variables()))])
    model.sess.run(init_op)

    # Create save and log dirs
    t = time.strftime("%y%m%d_%H%M%S")
    save_dir = os.path.join(CONFIG["train"].save_dir, str(t) +'_'+ "pg_delta_save")
    log_dir = os.path.join(CONFIG["train"].log_dir, str(t) +'_'+ "pg_delta_log")
    if not os.path.exists(save_dir): #create if not existing
        os.makedirs(save_dir)
    if not os.path.exists(CONFIG["train"].save_dir): #create if not existing
        os.makedirs(CONFIG["train"].save_dir)

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

    batch_size = CONFIG['train'].mem_buffer_batch_size

    print ("\n\n")
    P.INFO("Starting Training")
    print("\n\n")

    full_observation = env.reset()
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
        full_observation = env.reset()
        cropped_observation = preprocess(full_observation, camera_obj)

        EVAL = True if i_episode % 10 == 0 or FINAL else False
        if EVAL: print "==== STARTING EVALUATION EPISODE ===="

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


            # observation_, reward, done, info = env.step(action)
            next_full_observation, reward, crash, info = env.step(action) #step the previous frame with this action
            next_cropped_observation = preprocess(next_full_observation, camera_obj)
            # done = env.trace_done
            # crash = info['done']

            # reward = 1.0 #model.get_step_reward(info['translation'], CONFIG['sim'].car_width, CONFIG['sim'].road_width)

            # if done: #to deal w looping problem -- need to reset to other envs if you finish entire env without crashing
            #     dist += info['distance']
            #     break

            # RL.store_transition(observation, action, reward)
            xs.append(cropped_observation)
            ys.append([action])
            ys_prev.append([action_prev])
            rs.append(reward)
            step += 1

            if crash:
                dist += info['distance']
                if step == 1:
                    print "Crashing on first step... Possibly stuck in local min"
                    import pdb; pdb.set_trace()
                    break

                f_curv = interp1d(np.arange(len(ys)), np.array(ys), axis=0, fill_value='extrapolate')
                jerk = np.abs(derivative(f_curv, np.arange(len(ys)), dx=1./700, n=3, order=5))
                jerk = np.abs(ys)
                rs = np.array(rs)
                # rs -= 2*np.squeeze(jerk)
                # rs -= 1*np.squeeze(np.abs(ys))
                ep_rs_sum = np.sum(rs)

                reward_mean = alpha*reward_mean+(1-alpha)*(ep_rs_sum) if reward_mean is not None else ep_rs_sum
                print "episode: {:4.0f} \t step: {:4.0f} \t dist: {:5.1f} \t reward: {:7.2f} \t reward/step: {:5.2f} \t jerk/step: {:5.2f} \t running {:5.2f}".format(i_episode, step, dist, ep_rs_sum, ep_rs_sum/step, jerk.sum()/step, reward_mean)
                if EVAL:
                    print "======================================"
                    summary = model.sess.run(model.rewards_ep_summary, {model.rewards_ep: [[ep_rs_sum]]})
                    model.summary_writer.add_summary(summary, model.summary_iter)
                    model.summary_iter += 1

                    summary_steps += step
                    model.summary_writer.add_summary( model.sess.run(model.distances_ep_summary, {model.distance_ep: [[dist]]}), summary_steps)

                    if ep_rs_sum > max_ep_reward: # new best model obtained! save it!!
                        print "BEAT THE PREVIOUS BEST MODEL! SAVING A CHECKPOINT"
                        model.write_checkpoint(save_dir, name="model-{}".format(int(ep_rs_sum)))

                        max_ep_reward = ep_rs_sum # update the best ep reward for future
                        if FINAL:
                            print "Done!"
                            FINAL = False; break
                            # exit()
                    else:
                        FINAL = False

                    break

                # vt = RL.learn()
                # discount and normalize episode reward
                discounted_ep_rs_norm = _discount_and_norm_rewards(rs)

                xs = np.stack(xs)
                ys = np.array(ys)
                ys_prev = np.array(ys_prev)
                if len(xs)>5000:
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
                print "Beat max episode steps!"
                i_episode += 1
                summary = model.sess.run(model.rewards_ep_summary, {model.rewards_ep: [[ep_rs_sum]]})
                model.summary_writer.add_summary(summary, model.summary_iter)
                model.summary_iter += 1
                summary_steps += step
                model.summary_writer.add_summary( model.sess.run(model.distances_ep_summary, {model.distance_ep: [[dist]]}), summary_steps)

                if FINAL:
                    print "Writing checkpoint"
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



    for trace_obj in env_list:
        trace_obj.terminate()

    sess.close()
