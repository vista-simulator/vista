import sys
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt


with open(sys.argv[1], 'rb') as f:
    total_stats = pickle.load(f)

print('=================================')
print('Intervention: {} / {} ({:.3f})'.format(np.sum(total_stats['has_intervention']), 
    len(total_stats['has_intervention']), np.mean(total_stats['has_intervention'])))
print('Min Dist: {} ({})'.format(np.mean(total_stats['min_dist']), np.std(total_stats['min_dist'])))
print('Max Dev: {} ({})'.format(np.mean(total_stats['max_dev']), np.std(total_stats['max_dev'])))
print('Max Rot: {} ({})'.format(np.mean(total_stats['max_rot']), np.std(total_stats['max_rot'])))

n_bins = 15
cutoff_dist = 2.0
n_trials = len(total_stats['poly_dist'])

# plot histogram of clearance for all steps
all_poly_dist = np.array([vv for v in total_stats['poly_dist'] for vv in v])
roi_mask = all_poly_dist < cutoff_dist
all_poly_dist_roi = all_poly_dist[roi_mask]
hist, bin_edge = np.histogram(all_poly_dist_roi, n_bins)
bin_interval = bin_edge[-1] - bin_edge[-2]
fig, ax = plt.subplots(1, 1)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,3))
ax.set_xticks(bin_edge)
bar_x = [(bin_edge[i] + bin_edge[i+1])/2. for i in range(bin_edge.shape[0]-1)]
ax.bar(bar_x, hist, width=bin_interval)
ax.set_xticks(bin_edge)
ax.xaxis.set_tick_params(rotation=45)
ax.set_xlabel('Clearance (m)')
ax.set_ylabel('Number Of Steps')
ax.grid(color='grey', alpha=0.5, linestyle='dashed', linewidth=0.5)
fig.tight_layout()

# plot cumulative distribution of trials smaller than some clearance
trial_ids = np.array([i for i, v in enumerate(total_stats['poly_dist']) for vv in v])
acc_trial_ids = []
cum_trial_dist = []
for i in range(bin_edge.shape[0]):
    if i < bin_edge.shape[0] - 1:
        mask = np.logical_and(all_poly_dist > bin_edge[i], all_poly_dist <= bin_edge[i+1])
    else:
        mask = all_poly_dist > bin_edge[i]
    acc_trial_ids.extend(trial_ids[mask])
    acc_trial_ids = list(np.unique(acc_trial_ids))
    cum_trial_dist.append(len(acc_trial_ids) / n_trials)
fig, ax = plt.subplots(1, 1)
ax.xaxis.set_tick_params(rotation=45)
ax.set_xticks(bin_edge)
cum_trial_dist_x = [(bin_edge[i] + bin_edge[i+1])/2. for i in range(bin_edge.shape[0]-1)]
cum_trial_dist_x.append(bin_edge[-1] + bin_interval/2.)
ax.plot(cum_trial_dist_x, cum_trial_dist)
ax.set_xlabel('Clearance (m)')
ax.set_ylabel('Cumulative Distribution Of Trials')
ax.grid(color='grey', alpha=0.5, linestyle='dashed', linewidth=0.5)
fig.tight_layout()
plt.show()