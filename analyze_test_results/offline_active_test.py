import os
import numpy as np
import pickle
import argparse

# import utils


def main():
    # Parse arguments and config
    parser = argparse.ArgumentParser(description='Analyze offline active test')
    parser.add_argument('--result-path',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to result file')
    args = parser.parse_args()

    # Load data
    # args.result_path = utils.validate_path(args.result_path)
    with open(args.result_path, 'rb') as f:
        results = pickle.load(f)

    # Compute stats
    dev_mean, dev_std = compute_deviation(results)
    dist_mean, dist_std = compute_distance(results)
    crash_likelihood = compute_crash_likelihood(results, 25)
    
    print('Result from: {}'.format(args.result_path))
    print('Deviation: {} ({})'.format(dev_mean, dev_std))
    print('Distance: {} ({})'.format(dist_mean, dist_std))
    print('Crash likelihood: {}'.format(crash_likelihood))
    print('')


def compute_crash_likelihood(results, max_dist=500):
    crashes = []
    for ep_i in range(len(results['out_of_lane'])):
        idx = np.argmin(np.abs(np.array(results['distance'][ep_i]) - max_dist))
        out_of_lane = results['out_of_lane'][ep_i][idx]
        exceed_rot = results['exceed_rot'][ep_i][idx]
        crash = out_of_lane or exceed_rot
        crashes.append(crash)
    return np.mean(crashes)


def compute_deviation(results):
    dev = []
    for ep_result in results['relative_state']:
        ep_result = np.array(ep_result)
        dev.append(np.abs(ep_result[:,0]))
    dev = np.concatenate(dev)
    return dev.mean(), dev.std()


def compute_distance(results):
    dist = []
    ep_dist = 0
    for ep_i, ep_result in enumerate(results['ego_dynamics']):
        prev_xy = np.array([0., 0.])
        for step_i, step_result in enumerate(ep_result):
            current_xy = step_result[:2]
            ep_dist += np.linalg.norm(current_xy - prev_xy)
            prev_xy = current_xy

            out_of_lane = results['out_of_lane'][ep_i][step_i]
            exceed_rot = results['exceed_rot'][ep_i][step_i]
        dist.append(ep_dist)
        if out_of_lane or exceed_rot:
            ep_dist = 0
    dist = np.array(dist)
    return dist.mean(), dist.std()


if __name__ == '__main__':
    main()
