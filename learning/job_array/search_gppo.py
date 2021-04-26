import os
import sys


guidance_criterion_list = ['simple_kl']
guidance_coef_list = [0.1, 1.0, 5.0] #[0.1, 0.5, 1.0, 5.0, 10.0]
kl_coeff_list = [0.0, 0.2]
vf_loss_coeff_list = [0.0, 1.0]
entropy_coeff_list = [0.0, 0.01]


new_config = dict()
i = 1 # NOTE: task id starts from 1
for guidance_criterion in guidance_criterion_list:
    for guidance_coef in guidance_coef_list:
        for kl_coeff in kl_coeff_list:
            for vf_loss_coeff in vf_loss_coeff_list:
                for entropy_coeff in entropy_coeff_list:
                    new_config[i] = {
                        'guidance_criterion': guidance_criterion,
                        'guidance_coef': guidance_coef,
                        'kl_coeff': kl_coeff,
                        'vf_loss_coeff': vf_loss_coeff,
                        'entropy_coeff': entropy_coeff
                    }
                    i += 1


def update_exp_by_task_id(exp, task_id):
    exp['config']['guidance_criterion'] = new_config[task_id]['guidance_criterion']
    exp['config']['guidance_coef'] = new_config[task_id]['guidance_coef']
    exp['config']['kl_coeff'] = new_config[task_id]['kl_coeff']
    exp['config']['vf_loss_coeff'] = new_config[task_id]['vf_loss_coeff']
    exp['config']['entropy_coeff'] = new_config[task_id]['entropy_coeff']
    exp_name = 'gcrit-{}-gcoef{}-klcoef{}-vfcoef{}-entcoef{}'.format(
        new_config[task_id]['guidance_criterion'],
        new_config[task_id]['guidance_coef'],
        new_config[task_id]['kl_coeff'],
        new_config[task_id]['vf_loss_coeff'],
        new_config[task_id]['entropy_coeff']
    )

    return exp, exp_name


if __name__ == '__main__':
    task_id = int(sys.argv[1])
    print('\n[Task Dependent Config]')
    for k, v in new_config[task_id].items():
        print('{}: {}'.format(k, v))
    print('')