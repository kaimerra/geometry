import math

import torch
from torch import nn
from pytorch3d.renderer.points.pulsar import Renderer

import numpy as np

import matplotlib.pyplot as plt


import plotly.express as px


def _initial_points(n_points, center) -> torch.Tensor:
    return torch.tensor(
        np.random.multivariate_normal(center, np.identity(3) * 100.0, size=(n_points,)),
        dtype=torch.float32,
    )


class CameraModel(nn.Module):
    def __init__(self, n_images, positions, angles):
        super().__init__()

        self.camera_pos = nn.Parameter(positions, requires_grad=False)
        self.camera_angles = nn.Parameter(angles.float(), requires_grad=False)
        self.bg_color = nn.Parameter(
            torch.rand(3, dtype=torch.float32), requires_grad=True
        )

        #self.angle_to_rot = nn.Linear(3, 6)


class GeometryModel(nn.Module):
    def __init__(self, n_points, mean_pos, ball_size):
        super().__init__()

        self.pos = nn.Parameter(
            _initial_points(n_points, mean_pos).clone(), requires_grad=True
        )
        self.col = nn.Parameter(
            torch.rand(n_points, 3, dtype=torch.float32), requires_grad=True
        )
        self.rad = nn.Parameter(
            torch.ones(n_points, dtype=torch.float32) * ball_size, requires_grad=False
        )


class PulsarModel(nn.Module):
    def __init__(
        self,
        width,
        height,
        n_images,
        n_points=100000,
        ball_size=1.0,
        positions=None,
        angles=None,
    ):
        super().__init__()

        if positions is None:
            positions = torch.zeros(n_images, 3)

        self.mean_pos = positions.mean(axis=0)

        self.n_images = n_images

        self.alpha = 1.0
        self.geometry_model = GeometryModel(n_points, self.mean_pos, ball_size)
        self.camera = CameraModel(n_images, positions, angles)

        self.renderer = Renderer(width, height, n_points, right_handed_system=True)

    def set_grads(self, pos, camera):
        self.geometry_model.pos.requires_grad = pos
        #self.camera.angle_to_rot.requires_grad = camera

    def forward(self, i):
        batch_size = len(i)
        camera_pos = self.camera.camera_pos.data[i]
        camera_rot = self.camera.camera_angles.data[i]

        # Camera for rotation vs translation.
        cam_params = torch.stack(
            [
                # camera positinos
                camera_pos[:, 0],
                camera_pos[:, 1],
                camera_pos[:, 2],
                # rotations
                camera_rot[:, 0],
                camera_rot[:, 1],
                camera_rot[:, 2],
                torch.tensor([5.0] * batch_size, device=self.geometry_model.pos.device),
                torch.tensor([2.0] * batch_size, device=self.geometry_model.pos.device),
            ],
            axis=1,
        ).float()

        image = self.renderer(
            self.expand_batch(batch_size, self.geometry_model.pos),
            self.expand_batch(batch_size, torch.sigmoid(self.geometry_model.col)),
            self.expand_batch(
                batch_size, torch.max(self.geometry_model.rad, torch.tensor(0.01))
            ),  # * 100.0 + 0.001,
            cam_params,
            0.9,  # 1.0,  # Renderer blending parameter gamma, in [1., 1e-5].
            45.0,  # Maximum depth.
            bg_col=torch.sigmoid(self.camera.bg_color),
        )

        return image

    def expand_batch(self, batch_size, t):
        expanded = t[None]
        shape = list(expanded.shape)
        shape[0] = batch_size

        return expanded.expand(shape)

    def randomize(self):
        with torch.no_grad():
            indices = self.get_meaningful_points()
            print("randomizing", len(indices))

            # Updating position.
            self.geometry_model.pos[indices, :] = torch.tensor(
                np.random.multivariate_normal(
                    self.mean_pos, np.identity(3) * 100.0, size=(len(indices),)
                ),
                device=self.geometry_model.pos.device,
                dtype=torch.float32,
            )

    def get_meaningful_points(self, invert=False):
        with torch.no_grad():
            mag = torch.bmm(
                self.geometry_model.pos.grad.view(-1, 1, 3),
                self.geometry_model.pos.grad.view(-1, 3, 1),
            )
            (indices,) = (
                torch.where(mag.squeeze() == 0.0)
                if not invert
                else torch.where(mag.squeeze() > 0.0)
            )
            return indices


def plot3d(m):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(vertical_axis="y")
    indices = m.get_meaningful_points(True)
    print(len(indices))
    p = m.geometry_model.pos[indices].detach().cpu().numpy()
    c = torch.sigmoid(m.geometry_model.col[indices]).detach().cpu().numpy()
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], alpha=0.5, c=c)


def plot3d_plotly(m):
    indices = m.get_meaningful_points(True)
    p = m.geometry_model.pos[indices].detach().cpu().numpy()
    c = torch.sigmoid(m.geometry_model.col[indices]).detach().cpu().numpy()
    fig = px.scatter_3d(x=p[:, 0], y=p[:, 1], z=p[:, 2])  # , color="r")  # , color=c)
    return fig


def plot_camera(eye, at):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(vertical_axis="y")

    ax.plot(eye[:, 0], eye[:, 1], eye[:, 2])
    ax.quiver3D(
        eye[:, 0], eye[:, 1], eye[:, 2], at[:, 0], at[:, 1], at[:, 2], color="r"
    )
