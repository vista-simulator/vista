import os
import sys


num_sgd_iter_list = [8, 16, 32]
sgd_minibatch_size_list = [512, 1024, 2048, 4096]

new_config = dict()
i = 1 # NOTE: task id starts from 1
for num_sgd_iter in num_sgd_iter_list:
    for sgd_minibatch_size in sgd_minibatch_size_list:
        new_config[i] = {
            'num_sgd_iter': num_sgd_iter,
            'sgd_minibatch_size': sgd_minibatch_size
        }
        i += 1


def update_exp_by_task_id(exp, task_id):
    exp['config']['num_sgd_iter'] = new_config[task_id]['num_sgd_iter']
    exp['config']['sgd_minibatch_size'] = new_config[task_id]['sgd_minibatch_size']
    exp_name = 'iter{}-mbsize{}'.format(new_config[task_id]['num_sgd_iter'], new_config[task_id]['sgd_minibatch_size'])

    return exp, exp_name


if __name__ == '__main__':
    task_id = int(sys.argv[1])
    print('\n[Task Dependent Config]')
    for k, v in new_config[task_id].items():
        print('{}: {}'.format(k, v))
    print('')