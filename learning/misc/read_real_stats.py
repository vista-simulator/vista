import sys
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')


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
label_size = 22
number_ticklabel_size = 16
n_trials = len(total_stats['poly_dist'])

fig, axes = plt.subplots(1, 2, figsize=(20,6))
fig.subplots_adjust(left=0.048, bottom=0.174, right=0.991, top=0.97, wspace=0.24, hspace=0.2)

# plot histogram of clearance for all steps
all_poly_dist = np.array([vv for v in total_stats['poly_dist'] for vv in v])
roi_mask = all_poly_dist < cutoff_dist
all_poly_dist_roi = all_poly_dist[roi_mask]
hist, bin_edge = np.histogram(all_poly_dist_roi, n_bins)
bin_interval = bin_edge[-1] - bin_edge[-2]
# fig, ax = plt.subplots(1, 1)
ax = axes[0]
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,3))
ax.set_xticks(bin_edge)
ax.set_xticklabels(['{:.2f}'.format(v) for v in bin_edge])
ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
ax.xaxis.set_tick_params(labelsize=number_ticklabel_size)
ax.yaxis.set_tick_params(labelsize=number_ticklabel_size)
bar_x = [(bin_edge[i] + bin_edge[i+1])/2. for i in range(bin_edge.shape[0]-1)]
ax.bar(bar_x, hist, width=bin_interval)
ax.set_xticks(bin_edge)
ax.xaxis.set_tick_params(rotation=60)
ax.set_xlabel('Clearance (m)', fontsize=label_size)
ax.set_ylabel('Number Of Steps', fontsize=label_size)
ax.grid(color='grey', alpha=0.5, linestyle='dashed', linewidth=0.5)
# fig.tight_layout()

# fig.savefig('clearance_hist.pdf')

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
# fig, ax = plt.subplots(1, 1)
ax = axes[1]
ax.xaxis.set_tick_params(rotation=60)
ax.set_xticks(bin_edge)
ax.set_xticklabels(['{:.2f}'.format(v) for v in bin_edge])
ax.xaxis.set_tick_params(labelsize=number_ticklabel_size)
ax.yaxis.set_tick_params(labelsize=number_ticklabel_size)
cum_trial_dist_x = [(bin_edge[i] + bin_edge[i+1])/2. for i in range(bin_edge.shape[0]-1)]
cum_trial_dist_x.append(bin_edge[-1] + bin_interval/2.)
ax.plot(cum_trial_dist_x, cum_trial_dist, linewidth=4)
ax.set_xlabel('Clearance (m)', fontsize=label_size)
ax.set_ylabel('Cumulated Trials (%)', fontsize=label_size)
ax.grid(color='grey', alpha=0.5, linestyle='dashed', linewidth=0.5)
# fig.tight_layout()
# plt.show()
fig.savefig('clearance_analysis.pdf')
plt.show()