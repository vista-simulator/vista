""" Minimal example of using guided policy learning (GPL) for lane following. """
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import deque

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging, transform
from vista.entities.agents.Dynamics import tireangle2curvature

logging.setLevel(logging.ERROR)


class VistaDataset(Dataset):
    def __init__(self, trace_paths, transform, train=False):
        trace_config = dict(
            road_width=4,
            reset_mode='default',
            master_sensor='front_center',
        )
        car_config = dict(
            length=5.,
            width=2.,
            wheel_base=2.78,
            steering_ratio=14.7,
            lookahead_road=True, # NOTE: for optimal control
        )
        camera_config = dict(
            # camera params
            name='front_center',
            rig_path='/home/tsunw/data/traces/20200424-133758_blue_prius_cambridge_rain/RIG.xml',
            size=(200, 320),
            # rendering params
            depth_mode=DepthModes.FIXED_PLANE,
            use_lighting=False,
        )
        self.world = vista.World(trace_paths, trace_config)
        self.agent = self.world.spawn_agent(car_config)
        self.camera = self.agent.spawn_camera(camera_config)

        self.transform = transform
        self.train = train

        self.reset_config = dict(
            x_perturbation=[-1.0, 1.0],
            yaw_perturbation=[-0.04, 0.04],
            yaw_tolerance=0.01,
            distance_tolerance=0.3,
            maneuvor_cnt_ratio=0.5,
            max_horizon=400,
        )
        self.do_reset = False
        self.maneuvor_cnt = dict(recovery=0, lane_following=0)

        self.optimal_control_config = dict(
            lookahead_dist=10,
            dt=1 / 30.,
            Kp=0.5,
        )

        def _initial_dynamics_fn(x, y, yaw, steering, speed):
            return [
                x + np.random.uniform(*self.reset_config['x_perturbation']),
                y,
                yaw + np.random.uniform(*self.reset_config['yaw_perturbation']),
                steering,
                speed,
            ]
        self.initial_dynamics_fn = _initial_dynamics_fn
        self.world.reset({self.agent.id: self.initial_dynamics_fn})

    def __len__(self):
        return np.sum([tr.num_of_frames for tr in self.world.traces])

    def __getitem__(self, idx):
        # reset when trace is done or doing lane following for a while
        if self.agent.done or self.do_reset:
            self.world.reset({self.agent.id: self.initial_dynamics_fn})

            self.do_reset = False
            self.maneuvor_cnt['recovery'] = 0
            self.maneuvor_cnt['lane_following'] = 0

        # optimal control
        speed = self.agent.human_speed

        road = self.agent.road
        ego_pose = self.agent.ego_dynamics.numpy()[:3]
        road_in_ego = np.array([ # TODO: vectorize this
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
        self.agent.step_dynamics(action)
        self.agent.step_sensors()

        # fetch observation
        sensor_name = self.agent.sensors[0].name
        img = self.agent.observations[sensor_name]

        # preprocess observation
        (i1, j1, i2, j2) = self.agent.sensors[0].camera_param.get_roi()
        img = img[i1:i2, j1:j2].copy()
        img = self.transform(img)

        return img, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, 5, 2, 2),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, 3, 2, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.policy = nn.Sequential(
            nn.Linear(1280, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        z = self.extractor(x)
        z = torch.mean(z, dim=2)
        z = z.flatten(start_dim=1, end_dim=2)
        out = self.policy(z)
        return out


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) / len(data)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            curr_n_samples = batch_idx * len(data)
            total_samples = len(train_loader.dataset)
            progress = 100. * batch_idx / len(train_loader)
            print(f'Training epoch {epoch} [{curr_n_samples}/{total_samples} ({progress:.2f}%)]' \
                  + f'\t Loss: {loss.item():.6f}')


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average Loss: {test_loss:.6f}')
    return test_loss


def main():
    # Parse arguments (NOTE: just a placeholder here; hardcoded argument for now)
    parser = argparse.ArgumentParser(description='Minimal example of Vista IL training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Define data loader
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(),
    ])
    train_trace_paths = ['/home/tsunw/data/traces/20210527-131252_lexus_devens_center_outerloop',
                         '/home/tsunw/data/traces/20210527-131709_lexus_devens_center_outerloop_reverse',
                         '/home/tsunw/data/traces/20210609-122400_lexus_devens_outerloop_reverse',
                         '/home/tsunw/data/traces/20210609-123703_lexus_devens_outerloop',
                         '/home/tsunw/data/traces/20210609-133320_lexus_devens_outerloop',
                         '/home/tsunw/data/traces/20210609-154525_lexus_devens_sideroad',
                         '/home/tsunw/data/traces/20210609-154745_lexus_devens_outerloop_reverse',
                         '/home/tsunw/data/traces/20210609-155238_lexus_devens_outerloop',
                         '/home/tsunw/data/traces/20210609-155752_lexus_devens_subroad',
                         '/home/tsunw/data/traces/20210609-175037_lexus_devens_outerloop_reverse',
                         '/home/tsunw/data/traces/20210609-175503_lexus_devens_outerloop']
    train_dataset = VistaDataset(train_trace_paths, train_transform, train=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=0, # NOTE: multi-process data loader make FFReader fail
                              drop_last=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_trace_paths = ['/home/tsunw/data/traces/20210613-171636_lexus_devens_outerloop',
                        '/home/tsunw/data/traces/20210613-172102_lexus_devens_outerloop_reverse',
                        '/home/tsunw/data/traces/20210613-194157_lexus_devens_subroad',
                        '/home/tsunw/data/traces/20210613-194324_lexus_devens_subroad_reverse']
    test_dataset = VistaDataset(test_trace_paths, test_transform, train=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=64,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=0,
                             drop_last=True)

    # Define model and optimizer
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss(reduction='sum')

    # Run training
    all_test_loss = []
    for epoch in range(1, 100 + 1):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test_loss = test(model, device, test_loader, criterion)
        all_test_loss.append(test_loss)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), './ep_{:03d}.ckpt'.format(epoch))


if __name__ == '__main__':
    main()
