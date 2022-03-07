from turtle import bgcolor, shape

from zmq import device
import torch
from torch import nn
from pytorch3d.renderer.points.pulsar import Renderer

import numpy as np
       
class Scale(nn.Module):
    def __init__(self, mul):
        super().__init__()
        self.mul = mul

    def forward(self, X):
        return X * self.mul

class PulsarGanModel(nn.Module):
    def __init__(
        self,
        device,
        width,
        height,
        n_points=1,
        n_objects=100_000,
        ball_size=2.0
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.device = device
        self.n_points = n_points
        self.n_objects = n_objects
        self.latent_size = 30

        # need a model of points
        self.object_centers = nn.Sequential(
            nn.Linear(self.latent_size, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Sigmoid(),
            nn.Linear(60, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Tanh(),
            nn.Linear(60, self.n_objects * 3)
        )

        self.latent_pos = nn.Sequential(
            nn.Linear(self.latent_size, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Tanh(),
            nn.Linear(60, self.n_points * self.n_objects * 3),
            nn.Tanh(),
            Scale(10)
        )
        self.latent_col = nn.Sequential(
            nn.Linear(self.latent_size, 60),
            nn.Sigmoid(),
            nn.Linear(60, self.n_points * self.n_objects * 3),
            nn.Sigmoid(),
        )
        self.rad = nn.Parameter(
            torch.ones(self.n_points * self.n_objects, dtype=torch.float32) * ball_size, requires_grad=False
        )
        self.bg_color = nn.Parameter(
            torch.zeros(3, dtype=torch.float32), requires_grad=True
        )

        self.renderer = Renderer(width, height, self.n_points * self.n_objects, right_handed_system=True)

    def generate(self, n, random_cameras=True, random_zs = True):
        if random_zs:
            zs = torch.randn((n, self.latent_size), device=self.device)
        else:
            zs = torch.zeros((n, self.latent_size), device=self.device)

        if random_cameras:
            cameras = torch.rand((n, 6), device=self.device)
            # put it at a random x positi
            cameras[:, :] = 0.0
            cameras[:, 2] = 10.0 #+ torch.randn(n, 1) * 3.0
            cameras[:, 5] = 1.0 + torch.rand(n) * 1.0
        else:
            cameras = torch.zeros((n, 6), device=self.device)

        return self.forward(cameras, zs), cameras, zs

    def get_points(self, zs):
        batch_size = len(zs)

        centers = self.object_centers(zs).reshape(batch_size, self.n_objects, 3)
        centers = centers.unsqueeze(-2).expand(batch_size, self.n_objects, self.n_points, 3)

        pos = self.latent_pos(zs).reshape(batch_size, self.n_objects, self.n_points, 3)
        return (pos + centers).reshape(batch_size, -1, 3)

    def forward(self, cameras, zs):
        batch_size = len(zs)

        focal = torch.tensor([[20.0]] * batch_size, device=self.device)
        sensor_width = torch.tensor([[40.0]] * batch_size, device=self.device)
        principal_x = torch.tensor([[0]]*batch_size, device=self.device)
        principal_y = torch.tensor([[0]]*batch_size, device=self.device)
        # renderer._transfo
        # we have an N x 6 camera thing. we will it to to be N x (6+2)
        cam_params = torch.concat([cameras, focal, sensor_width, principal_x, principal_y], dim=1)

        pos = self.get_points(zs)
        col = self.latent_col(zs).reshape(batch_size, -1, 3)
        rad = self.expand_batch(batch_size, self.rad)
        image = self.renderer(
            pos,
            col,
            rad,
            cam_params,
            1.0e-1,
            max_depth=100.0,
            bg_col=self.bg_color
        )
        return image, cam_params

    def initialize(self):
        pass

    def expand_batch(self, batch_size, t):
        expanded = t[None]
        shape = list(expanded.shape)
        shape[0] = batch_size

        return expanded.expand(shape)

 