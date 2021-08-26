""" Minimal example of using guided policy learning (GPL) for lane following. """
import os
import sys
import time
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from collections import deque
from tqdm import tqdm

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging, transform
from vista.entities.agents.Dynamics import tireangle2curvature
from vista.core.Display import events2frame

logging.setLevel(logging.ERROR)


class VistaDataset(IterableDataset):
    def __init__(self, trace_paths, transform, train=False,
                 buffer_size=1, snippet_size=100):
        # NOTE: do NOT define anything that will be updated here since multiprocessing in pytorch
        #       is doing something like memory copy instead of instantiating new copies (unless we
        #       are using 'spawn' instead of 'fork' for start method)
        trace_config = dict(
            road_width=4,
            reset_mode='default',
            master_sensor='lidar_3d',
        )
        car_config = dict(
            length=5.,
            width=2.,
            wheel_base=2.78,
            steering_ratio=14.7,
            lookahead_road=True, # NOTE: for optimal control
            road_buffer_size=50, # NOTE: too large make
        )
        lidar_config = dict(
            name='lidar_3d',
        )

        self.trace_paths = trace_paths
        self.trace_config = trace_config
        self.car_config = car_config
        self.lidar_config = lidar_config

        self.transform = transform
        self.train = train

        self.buffer_size = max(1, buffer_size)
        self.snippet_size = max(1, snippet_size)

        self.reset_config = dict(
            x_perturbation=[-1.0, 1.0],
            yaw_perturbation=[-0.04, 0.04],
            yaw_tolerance=0.01,
            distance_tolerance=0.3,
            maneuvor_cnt_ratio=0.5,
            max_horizon=400,
        )

        self.optimal_control_config = dict(
            lookahead_dist=10,
            dt=1 / 30.,
            Kp=0.5,
        )

        # NOTE: just to make len() still work in the master process; will be instantiated
        # independently for each worker in worker_init_fn
        self.world = vista.World(trace_paths, trace_config)

    def __len__(self):
        return np.sum([tr.num_of_frames for tr in self.world.traces])

    def __iter__(self):
        # buffered shuffle dataset
        self.rng = random.Random(torch.utils.data.get_worker_info().id)
        buf = []
        print("Filling buffer...")
        pbar = tqdm(total=self.buffer_size)
        for x in self._simulate():
            if len(buf) == self.buffer_size:
                pbar.close()
                idx = self.rng.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
                pbar.update(1)
        self.rng.shuffle(buf)
        while buf:
            yield buf.pop()

    def _simulate(self):
        # initialization for num_worker = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.agent = self.world.spawn_agent(self.car_config)
            self.camera = self.agent.spawn_camera(self.camera_config)
            self.world.reset()

        # data generator from simulation
        self.snippet_i = 0
        while True:
            # reset when trace is done or doing lane following for a while
            if self.agent.done or self.do_reset or self.snippet_i >= self.snippet_size:
                if worker_info is not None:
                    self.world.set_seed(worker_info.id)
                self.world.reset({self.agent.id: self.initial_dynamics_fn})
                self.snippet_i = 0

                self.do_reset = False
                self.maneuvor_cnt['recovery'] = 0
                self.maneuvor_cnt['lane_following'] = 0

            # optimal control
            tic = time.time()

            speed = self.agent.human_speed

            road = self.agent.road
            ego_pose = self.agent.ego_dynamics.numpy()[:3]
            road_in_ego = np.array([ # TODO: vectorize this: slow if road buffer size too large
                transform.compute_relative_latlongyaw(_v, ego_pose)
                for _v in road
            ])
            road_in_ego = road_in_ego[road_in_ego[:,1] > 0] # drop road in the back

            lookahead_dist = self.optimal_control_config['lookahead_dist']
            dt = self.optimal_control_config['dt']
            Kp = self.optimal_control_config['Kp']

            dist = np.linalg.norm(road_in_ego[:,:2], axis=1)
            tgt_idx = np.argmin(np.abs(dist - lookahead_dist))
            dx, dy, dyaw = road_in_ego[tgt_idx]

            lat_shift = -self.agent.relative_state.x
            dx += lat_shift * np.cos(dyaw)
            dy += lat_shift * np.sin(dyaw)

            arc_len = speed * dt
            curvature = (Kp * np.arctan2(-dx, dy) * dt) / arc_len
            curvature_bound = [
                tireangle2curvature(_v, self.agent.wheel_base)
                for _v in self.agent.ego_dynamics.steering_bound]
            curvature = np.clip(curvature, *curvature_bound)

            label = np.array([curvature]).astype(np.float32)

            toc = time.time()
            time_optimal_control = toc - tic

            # actively reset to balance between lane following and recovery maneuvor
            x_diff, y_diff, yaw_diff = self.agent.relative_state.numpy()
            yaw_diff_mag = np.abs(yaw_diff)
            distance = np.linalg.norm([x_diff, y_diff])
            if yaw_diff_mag <= self.reset_config['yaw_tolerance'] and \
                distance <= self.reset_config['distance_tolerance']:
                self.maneuvor_cnt['lane_following'] += 1
            else:
                self.maneuvor_cnt['recovery'] += 1
            is_balanced = self.maneuvor_cnt['lane_following'] >= \
                self.maneuvor_cnt['recovery'] * self.reset_config['maneuvor_cnt_ratio']
            exceed_max_horizon = (self.maneuvor_cnt['lane_following'] + \
                self.maneuvor_cnt['recovery']) >= self.reset_config['max_horizon']
            if is_balanced or exceed_max_horizon:
                self.do_reset = True

            # step simulation
            action = np.array([curvature, speed])

            tic = time.time()
            self.agent.step_dynamics(action)
            toc = time.time()
            time_step_dynamics = toc - tic

            tic = time.time()
            self.agent.step_sensors()
            toc = time.time()
            time_step_sensors = toc - tic

            # fetch observation
            sensor_name = self.agent.sensors[0].name
            pcd = self.agent.observations[sensor_name]

            xyz = pcd.xyz / 100.
            intensity = np.log(pcd.intensity)
            intensity = intensity - intensity.mean()
            data = np.concatenate((xyz, intensity[:, np.newaxis]), axis=1)
            data = data.astype(np.float32)

            # # preprocess observation
            # event_cam_param = self.agent.sensors[0].camera_param
            # img = events2frame(events, event_cam_param.get_height(), event_cam_param.get_width())
            # img = standardize(self.transform(img))

            self.snippet_i += 1

            yield data, label

    def initial_dynamics_fn(self, x, y, yaw, steering, speed):
        return [
            x + np.random.uniform(*self.reset_config['x_perturbation']),
            y,
            yaw + np.random.uniform(*self.reset_config['yaw_perturbation']),
            steering,
            speed,
        ]


def standardize(x):
    # follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    mean, stddev = x.mean(), x.std()
    adjusted_stddev = max(stddev, 1.0/np.sqrt(np.prod(x.shape)))
    return (x - mean) / adjusted_stddev


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 64, (1, 4), 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.policy = nn.Sequential(
            nn.Linear(1024, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        z = torch.unsqueeze(x, 1) # (B, 1, N, 3)
        z = self.extractor(z)
        z = torch.mean(z, dim=2)
        z = z.flatten(start_dim=1)
        z = self.policy(z)
        return z



def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    tic = time.time()
    sample_i = 0
    total_samples = len(train_loader.dataset)
    loss_buf = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = 10e4 * criterion(output, target) / len(data)
        loss.backward()
        optimizer.step()
        loss_buf.append(loss.item())
        if batch_idx % 100 == 0:
            toc = time.time()
            elapsed_time = toc - tic
            curr_n_samples = batch_idx * len(data)
            progress = 100. * batch_idx / len(train_loader)
            avg_loss = np.mean(loss_buf)
            print(f'Training epoch {epoch} [{curr_n_samples}/{total_samples} ({progress:.2f}%)]' \
                  + f'\t Average Loss: {avg_loss:.6f}\t Elapsed time: {elapsed_time:.2f}')
            sys.stdout.flush()
            tic = toc
            loss_buf = []

        if batch_idx % 500 == 0:
            save_dir = os.environ.get('RESULT_DIR', './ckpt/gpl_lidar')
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep_{:03d}.ckpt'.format(epoch)))


        sample_i += len(data)
        if sample_i >= total_samples:
            break


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    sample_i = 0
    total_samples = len(test_loader.dataset)
    tic = time.time()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            test_loss += 10e4 * criterion(output, target).item()

        sample_i += len(data)
        if sample_i >= total_samples:
            break
    toc = time.time()

    elapsed_time = toc - tic
    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average Loss: {test_loss:.6f}\t Elapsed time: {elapsed_time:.2f}')
    sys.stdout.flush()
    return test_loss


# NOTE: must be a global function otherwise cannot be handled by spawn in multiprocessing
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.world = vista.World(dataset.trace_paths, dataset.trace_config)
    dataset.agent = dataset.world.spawn_agent(dataset.car_config)
    dataset.lidar = dataset.agent.spawn_lidar(dataset.lidar_config)
    dataset.do_reset = False
    dataset.maneuvor_cnt = dict(recovery=0, lane_following=0)
    dataset.world.set_seed(worker_id)
    dataset.rng = random.Random(worker_id)
    dataset.world.reset({dataset.agent.id: dataset.initial_dynamics_fn})


def main():
    # Parse arguments (NOTE: just a placeholder here; hardcoded argument for now)
    parser = argparse.ArgumentParser(description='Minimal example of Vista IL training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-path', type=str, default=None,
                        help='Path to log file; default None to simply print out logs')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.log_path is not None:
        log_path = os.path.abspath(os.path.expanduser(args.log_path))
        log_f = open(log_path, 'w')
        sys.stdout = log_f

    # Define data loader
    data_dir = os.environ.get('DATA_DIR', '/home/tsunw/data/')

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(),
    ])
    train_trace_paths = ['20210726-131322_lexus_devens_center',
                         '20210726-131912_lexus_devens_center_reverse',
                         '20210726-154641_lexus_devens_center',
                         '20210726-155941_lexus_devens_center_reverse',
                         '20210726-184624_lexus_devens_center',
                         '20210728-203206_lexus_devens_center_night']
    # train_trace_paths = ['20210527-131252_lexus_devens_center_outerloop',
    #                      '20210527-131709_lexus_devens_center_outerloop_reverse',
    #                      '20210609-122400_lexus_devens_outerloop_reverse',
    #                      '20210609-123703_lexus_devens_outerloop',
    #                      '20210609-133320_lexus_devens_outerloop',
    #                      '20210609-154525_lexus_devens_sideroad',
    #                      '20210609-154745_lexus_devens_outerloop_reverse',
    #                      '20210609-155238_lexus_devens_outerloop',
    #                      '20210609-155752_lexus_devens_subroad',
    #                      '20210609-175037_lexus_devens_outerloop_reverse',
    #                      '20210609-175503_lexus_devens_outerloop']
    train_trace_paths = [os.path.join(data_dir, 'traces', v) for v in train_trace_paths]
    train_dataset = VistaDataset(train_trace_paths, train_transform, buffer_size=500, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False, # NOTE: IterDataset doesn't allow shuffle
                              pin_memory=True,
                              num_workers=1,
                              drop_last=True,
                              worker_init_fn=worker_init_fn)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_trace_paths = ["20210726-184956_lexus_devens_center_reverse",
                        "20210728-204515_lexus_devens_center_night",]
    # test_trace_paths = ['20210613-171636_lexus_devens_outerloop',
    #                     '20210613-172102_lexus_devens_outerloop_reverse',
    #                     '20210613-194157_lexus_devens_subroad',
    #                     '20210613-194324_lexus_devens_subroad_reverse']
    test_trace_paths = [os.path.join(data_dir, 'traces', v) for v in test_trace_paths]
    test_dataset = VistaDataset(test_trace_paths, test_transform, train=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=64,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=4,
                             drop_last=True,
                             worker_init_fn=worker_init_fn)

    # Define model and optimizer
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss(reduction='sum')

    # Run training
    all_test_loss = []
    for epoch in range(1, 10 + 1):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test_loss = test(model, device, test_loader, criterion)
        all_test_loss.append(test_loss)

        if epoch % 1 == 0:
            save_dir = os.environ.get('RESULT_DIR', './ckpt/gpl_lidar')
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep_{:03d}.ckpt'.format(epoch)))


    if args.log_path is not None:
        log_f.close()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn') # NOTE: for CUDA in multi-process
    main()
