""" Minimal example of using imitation learning (IL) for lane following. """
import os
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

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging

logging.setLevel(logging.ERROR)


class VistaDataset(IterableDataset):
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
        )
        camera_config = dict(
            # camera params
            name='front_center',
            rig_path='../examples/RIG.xml',
            size=(200, 320),
            # rendering params
            depth_mode=DepthModes.FIXED_PLANE,
            use_lighting=False,
            use_synthesizer=False, # NOTE: don't do view synthesis
        )

        self.trace_paths = trace_paths
        self.trace_config = trace_config
        self.car_config = car_config
        self.camera_config = camera_config

        self.transform = transform
        self.train = train

        self.buffer_size = 1000 # DEBUG

        # NOTE: just to make len() still work in the master process; will be instantiated 
        # independently for each worker in worker_init_fn
        self.world = vista.World(trace_paths, trace_config)

    def __len__(self):
        return 6000 # DEBUG np.sum([tr.num_of_frames for tr in self.world.traces])

    def __iter__(self):
        buf = []
        for x in self._simulate():
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
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
        [self.world.reset() for _ in range(worker_info.id)] # hacky way to make different workers reset differently
        while True:
            if self.agent.done:
                # self.world.reset()
                [self.world.reset() for _ in range(worker_info.id)]

            print(worker_info.id, self.agent.trace_index, self.agent.segment_index, self.agent.frame_index) # DEBUG

            self.agent.step_dataset(step_dynamics=False)
            sensor_name = self.agent.sensors[0].name
            img = self.agent.observations[sensor_name]

            (i1, j1, i2, j2) = self.agent.sensors[0].camera_param.get_roi()
            img = img[i1:i2, j1:j2]

            img = standardize(self.transform(img))
            label = np.array([self.agent.human_curvature]).astype(np.float32)

            yield img, label


def standardize(x):
    # follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    mean, stddev = x.mean(), x.std()
    adjusted_stddev = max(stddev, 1.0/np.sqrt(np.prod(x.shape)))
    return (x - mean) / adjusted_stddev


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
            nn.Linear(1280, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        z = self.extractor(x)
        z = torch.mean(z, dim=2)
        z = z.flatten(start_dim=1, end_dim=2)
        out = self.policy(z)
        return out


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    tic = time.time()
    sample_i = 0
    total_samples = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = 10e4 * criterion(output, target) / len(data)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            toc = time.time()
            elapsed_time = toc - tic
            curr_n_samples = batch_idx * len(data)
            progress = 100. * batch_idx / len(train_loader)
            print(f'Training epoch {epoch} [{curr_n_samples}/{total_samples} ({progress:.2f}%)]' \
                  + f'\t Loss: {loss.item():.6f}\t Elapsed time: {elapsed_time:.2f}')
            tic = toc
        
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

        print(output.item(), target.item()) # DEBUG

        sample_i += len(data)
        if sample_i >= total_samples:
            break
    toc = time.time()

    elapsed_time = toc - tic
    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average Loss: {test_loss:.6f}\t Elapsed time: {elapsed_time:.2f}')
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
    data_dir = os.environ.get('DATA_DIR', '/home/tsunw/data/')

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.world = vista.World(dataset.trace_paths, dataset.trace_config)
        dataset.agent = dataset.world.spawn_agent(dataset.car_config)
        dataset.camera = dataset.agent.spawn_camera(dataset.camera_config)
        dataset.world.reset()

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(),
    ])
    train_trace_paths = ['20210527-131252_lexus_devens_center_outerloop',
                         '20210527-131709_lexus_devens_center_outerloop_reverse',
                         '20210609-122400_lexus_devens_outerloop_reverse',
                         '20210609-123703_lexus_devens_outerloop',
                         '20210609-133320_lexus_devens_outerloop',
                         '20210609-154525_lexus_devens_sideroad',
                         '20210609-154745_lexus_devens_outerloop_reverse',
                         '20210609-155238_lexus_devens_outerloop',
                         '20210609-155752_lexus_devens_subroad',
                         '20210609-175037_lexus_devens_outerloop_reverse',
                         '20210609-175503_lexus_devens_outerloop']
    train_trace_paths = [os.path.join(data_dir, 'traces', v) for v in train_trace_paths]
    train_dataset = VistaDataset(train_trace_paths, train_transform, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False, # NOTE: IterDataset doesn't allow shuffle
                              pin_memory=True,
                              num_workers=4,
                              drop_last=True,
                              worker_init_fn=worker_init_fn)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_trace_paths = ['20210613-171636_lexus_devens_outerloop',
                        '20210613-172102_lexus_devens_outerloop_reverse',
                        '20210613-194157_lexus_devens_subroad',
                        '20210613-194324_lexus_devens_subroad_reverse']
    test_trace_paths = [os.path.join(data_dir, 'traces', v) for v in test_trace_paths]
    test_dataset = VistaDataset(test_trace_paths, test_transform, train=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=0,
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
            save_dir = os.environ.get('RESULT_DIR', './ckpt/il2')
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep_{:03d}.ckpt'.format(epoch)))


if __name__ == '__main__':
    main()
