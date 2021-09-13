import os
import glob
from re import L
import numpy as np
import pickle
import argparse
from collections import OrderedDict
from pprint import pprint
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

from utils import *


EXPs = {
    'rgb': {
        'il': '*lexus_il_rgb_gn_iter400k*',
        'gpl': '*lexus_gpl_rgb_pid_gn_iter400k*',
        'gpl_il': '*lexus_gpl_il_rgb_gn_iter400k*',
    },
    'lidar': {
        'il': '*lexus_lidar_il*',
        'gpl': '*lexus_lidar_gpl-0.2-150k*',
        'gpl_il': '*lexus_lidar_gpl-il-60k*',
    },
    'event': {
        'il': '*lexus_il_event_iter120k*',
        'gpl': '*lexus_gpl_event_pid_gn_iter400k*',
        'gpl_il': '*lexus_gpl_il_event_gn_no_branching_iter400k*',
        'gpl_no_br': '*lexus_gpl_event_pid_gn_no_branching_iter400k*'
    },
}
EXPs = OrderedDict(EXPs)
for k, v in EXPs.items():
    EXPs[k] = OrderedDict(v)

HUMAN_TRACEs = [
    '$DATA_DIR/traces/20210609-122400_lexus_devens_outerloop_reverse',
    '$DATA_DIR/traces/20210609-123703_lexus_devens_outerloop',
    '$DATA_DIR/traces/20210609-154745_lexus_devens_outerloop_reverse',
    '$DATA_DIR/traces/20210609-155238_lexus_devens_outerloop',
    '$DATA_DIR/traces/20210609-175037_lexus_devens_outerloop_reverse',
    '$DATA_DIR/traces/20210609-175503_lexus_devens_outerloop',
]


def main():
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bag-root-dir',
        type=str,
        required=True,
        help='Root directory of all .')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Directory to save results')
    parser.add_argument(
        '--recompute-intervention',
        action='store_true',
        default=False)
    parser.add_argument(
        '--recompute-deviation',
        action='store_true',
        default=False)
    parser.add_argument(
        '--recompute-jitteriness',
        action='store_true',
        default=False)
    parser.add_argument(
        '--devens-road-path',
        type=str,
        default=None,
        help='Path to the pickle file that stores Devens road')
    args = parser.parse_args()

    args.bag_root_dir = validate_path(args.bag_root_dir)
    args.results_dir = validate_path(args.results_dir)

    # Get devens road and lane center
    all_human_xys = OrderedDict()
    for trace_path in HUMAN_TRACEs:
        trace_path = validate_path(trace_path)
        gps = np.genfromtxt(os.path.join(trace_path, 'gps.csv'), delimiter=',')
        x, y, _, _ = utm.from_latlon(gps[:,1], gps[:,2])
        all_human_xys[trace_path] = np.stack([x,y], axis=1)

    # Extract data from bag
    total_trials = 0
    trial_paths = OrderedDict()
    for modal in EXPs.keys():
        trial_paths[modal] = OrderedDict()
        for alg, partial_name in EXPs[modal].items():
            search_pattern = os.path.join(args.bag_root_dir, partial_name)
            paths = glob.glob(search_pattern)
            if len(paths) > 3:
                new_paths = []
                for trial_i in range(1, 4):
                    new_path = sorted([v for v in paths if f'trial{trial_i}' in v \
                                       and not f'trial{trial_i}_' in v])[-1]
                    new_paths.append(new_path)
                paths = new_paths
            trial_paths[modal][alg] = paths
            total_trials += 1
    pprint(trial_paths)

    # Compute stats
    all_results_path = os.path.join(args.results_dir, 'all_results.pkl')
    recompute = {
        'intervention': args.recompute_intervention, 
        'deviation': args.recompute_deviation, 
        'jitteriness': args.recompute_jitteriness
    }
    if not os.path.exists(all_results_path) or np.any(list(recompute.values())):
        pbar = tqdm(total=total_trials)
        if np.any(list(recompute.values())):
            with open(all_results_path, 'rb') as f:
                all_results = pickle.load(f)
        else:
            all_results = OrderedDict()
        for modal in trial_paths.keys():
            if not np.any(list(recompute.values())):
                all_results[modal] = OrderedDict()
            for alg, dirs in trial_paths[modal].items():
                if not np.any(list(recompute.values())):
                    all_results[modal][alg] = OrderedDict()
                for dd in dirs:
                    bag_paths = [v for v in glob.glob(os.path.join(dd,'sensor*.bag'))]
                    if len(bag_paths) > 1:
                        bag_paths = [v for v in glob.glob(os.path.join(dd,'sensor_part*.bag'))]
                    for bag_path in bag_paths:
                        if np.any(list(recompute.values())):
                            results = all_results[modal][alg][bag_path]
                        else:
                            results = dict()
                        results = analyze_bag(bag_path, all_human_xys, recompute, results)
                        all_results[modal][alg][bag_path] = results
                pbar.update(1)
        with open(all_results_path, 'wb') as f:
            pickle.dump(all_results, f)
    else:
        with open(all_results_path, 'rb') as f:
            all_results = pickle.load(f)

    # Get quantitative results
    for modal in all_results.keys():
        for alg in all_results[modal].keys():
            alg_n_intervention = []
            alg_mean_dev = []
            alg_jitteriness = []
            for bag_path, results in all_results[modal][alg].items():
                n_intervention = len(results['intervention'])
                if bag_path == '/home/tsunw/data/icra2022/multisensor/20210911-103200_lexus_il_rgb_gn_iter400k_trial1/sensor.bag':
                    n_intervention = 4
                mean_dev = results['deviation'].mean()
                mean_jitteriness = results['jitteriness'].mean()
                alg_n_intervention.append(n_intervention)
                alg_mean_dev.append(mean_dev)
                alg_jitteriness.append(mean_jitteriness)
            alg_mean_n_intervention = np.mean(alg_n_intervention)
            alg_std_n_intervention = np.std(alg_n_intervention)
            alg_mean_mean_dev = np.mean(alg_mean_dev)
            alg_std_mean_dev = np.std(alg_mean_dev)
            alg_mean_jitteriness = np.mean(alg_jitteriness)
            alg_std_jitteriness = np.std(alg_jitteriness)
            print(f'{modal} {alg}: ({alg_mean_n_intervention}, {alg_std_n_intervention}),' + \
                 f'({alg_mean_mean_dev}, {alg_std_mean_dev}), ({alg_mean_jitteriness}, {alg_std_jitteriness})')

    # Get plot
    fig, ax = plt.subplots(1, 1)
    for modal in EXPs.keys():
        for alg in EXPs[modal].keys():
            if alg not in ['il', 'gpl']:
                continue

            plot_loop = False
            if plot_loop:
                fig2, ax2 = plt.subplots(1, 1)
                if args.devens_road_path is not None:
                    args.devens_road_path = validate_path(args.devens_road_path)
                    devens_road = load_devens_road(args.devens_road_path)
                    plot_devens_road(devens_road, [fig2, ax2])
                
            alg_deviation = []
            alg_dev = []
            for bag_path, results in all_results[modal][alg].items():
                alg_deviation.append(results['deviation'])
                if plot_loop:
                    intervention = results['intervention']  
                    deviation_txy = results['deviation_txy']
                    intervention_xy = np.array([deviation_txy[np.argmin(np.abs(v-deviation_txy[:,0])),1:] for v in intervention])
                    
                    ax2.plot(deviation_txy[:,1], deviation_txy[:,2], c=(60/255.,179/255.,113/255., 0.5), linewidth=1, zorder=1)
                    if intervention_xy.shape[0] > 0:
                        ax2.scatter(intervention_xy[:,0], intervention_xy[:,1], s=35, c=(1., 0., 0., 0.7), marker='o', zorder=2) # (171/255., 78/255., 82/255., 0.5)
            if plot_loop:
                fig2.savefig(os.path.join(args.results_dir, f'loop-{modal}-{alg}.png'))
            alg_deviation = np.concatenate(alg_deviation)

            deviation_percent = []
            lambda_range = np.arange(alg_deviation.min(), alg_deviation.max(), 0.01)
            for lamb in lambda_range:
                deviation_percent.append(np.mean(alg_deviation < lamb))
            ax.plot(lambda_range, deviation_percent, label=f'{modal}-{alg}')
    ax.legend()
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Percentage of Deviation < Lambda')
    fig.savefig(os.path.join(args.results_dir, 'deviation_lambda.png'))
    import pdb; pdb.set_trace()


def analyze_bag(bag_path, all_human_xys, recompute, results={}):
    data = read_rosbag(bag_path)

    if recompute['intervention']:
        intervention = fetch_intervention(data, min_t_diff=15)
        results['intervention'] = intervention

    if recompute['deviation']:
        gps = fetch_gps(data)
        gps = split_to_segments(gps, intervention=results['intervention'])
        deviation = []
        deviation_txy = []
        for gps_seg in gps:
            all_dev = []
            for human_xys in all_human_xys.values():
                dist_mat = distance_matrix(gps_seg[:,1:], human_xys)
                all_dev.append(dist_mat.min(1))
            all_dev = np.array(all_dev)
            dev = np.mean(all_dev, axis=0)
            deviation.append(dev)
            deviation_txy.append(gps_seg)
        deviation = np.concatenate(deviation)
        deviation_txy = np.concatenate(deviation_txy)
        results['deviation'] = deviation
        results['deviation_txy'] = deviation_txy

    # compute jitteriness
    if recompute['jitteriness']:
        ego_curvature = fetch_curvature(data, '/lexus/deepknight/goal_steer')
        ego_curvature = split_to_segments(ego_curvature, intervention=results['intervention'])
        jitteriness = []
        for seg in ego_curvature:
            dcurv = seg[1:,1] - seg[:-1,1]
            ddcurv = dcurv[1:] - dcurv[:-1]
            jitteriness.append(ddcurv**2)
        jitteriness = np.concatenate(jitteriness)
        results['jitteriness'] = jitteriness

    return results


if __name__ == '__main__':
    main()