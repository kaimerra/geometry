import torch
from torch import nn

import numpy as np

from .nerf_models import HarmonicEmbedding

from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    ray_bundle_to_ray_points,
    VolumeRenderer,
    NDCGridRaysampler,
    EmissionAbsorptionRaymarcher,
    look_at_view_transform,
    FoVPerspectiveCameras,
    RayBundle,
)


class TileBasedImplicitRenderer(nn.Module):
    def __init__(self):
        super().__init__()

        self.total_width = 10
        self.total_depth = 10

        # go  from 30 to 20 x 20 x dim
        self.latent_map = torch.Linear(30, 20 * 20 * 10)

    def render(self, ray_bundle: RayBundle, **kwargs):
        world_points = ray_bundle_to_ray_points()
        # Basically now
        # z creates a latent space a 2d map
        # that 2d map then parametrizes a function
        # nerf(local position, parameter)
        # lets take each
        # get indices
        indices_long = ((world_points[..., 0:0]) / self.total_width).long()
        indices_float = indices_long.float()

        localized_world_points = world_points - indices_float

        pass


def generate_cameras(n, device, random=True):
    Rs = []
    Ts = []
    for i in range(n):
        # random 2d point
        if random:
            rand_eye = torch.randn((1, 3), device=device)
            rand_at = torch.randn((1, 3), device=device)

            rand_eye[:, 0] *= 0.01
            rand_eye[:, 1] *= 0.0001
            rand_eye[:, 2] *= 0.01

            rand_at *= 0.01
        else:
            rand_eye = torch.tensor([[0.1, 0.0, 0.0]], device=device)
            rand_at = torch.tensor([[0.0, 0.0, 0.0]], device=device)

        R, T = look_at_view_transform(eye=rand_eye, at=rand_at)
        Rs.append(R[0])
        Ts.append(T[0])

    return FoVPerspectiveCameras(R=torch.stack(Rs), T=torch.stack(Ts), device=device)


class VolumeModel(nn.Module):
    def __init__(self, device, volume_size=[64] * 3, voxel_size=0.1):
        super().__init__()

        self.device = device
        self.voxel_size = voxel_size

        self.raysampler = NDCGridRaysampler(
            image_width=64,
            image_height=64,
            n_pts_per_ray=300,
            min_depth=0.1,
            max_depth=3.0,
        )

        # 2) Instantiate the raymarcher.
        # Here, we use the standard EmissionAbsorptionRaymarcher
        # which marches along each ray in order to render
        # each ray into a single 3D color vector
        # and an opacity scalar.
        self.raymarcher = EmissionAbsorptionRaymarcher()

        # Finally, instantiate the volumetric render
        # with the raysampler and raymarcher objects.
        self.renderer = VolumeRenderer(
            raysampler=self.raysampler,
            raymarcher=self.raymarcher,
        )

        self.latent_dim = 16

        # Generate a 32 x 32 x 32 latent space
        self.latent = nn.Sequential(
            nn.Linear(30, 60),
            nn.Sigmoid(),
            nn.Linear(60, 480),
            nn.Sigmoid(),
            nn.Linear(480, 16 * 16 * 16 * 4),
            nn.Sigmoid(),
        )

    def generate_volume(self, zs):
        latent = self.latent(zs).reshape(
            -1, 4, self.latent_dim, self.latent_dim, self.latent_dim
        )
        densities = latent[:, 0:3]
        colors = latent[:, 3:4]

        # Generate a bunch of volume
        # The output of the latent space is a set of volumes
        volumes = Volumes(
            densities=densities,
            features=colors,
            voxel_size=self.voxel_size,
        )

        return volumes

    def forward(self, cameras, zs):
        volumes = self.generate_volume(zs)
        # Given cameras and volumes, run the renderer
        # and return only the first output value
        # (the 2nd output is a representation of the sampled
        # rays which can be omitted for our purpose).
        imgs = self.renderer(cameras=cameras, volumes=volumes)[0]
        # return densities, and features
        return (
            imgs[..., 3:4],
            imgs[..., 0:3],
        )

    def generate(self, n):
        # first convert the ray origins, directions and lengths
        # to 3D ray point locations in world coords
        cameras = generate_cameras(n, self.device)
        z = torch.randn((n, 30), device=self.device)
        return self.forward(cameras, z)


class LatentSpace(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.latent_dim = 320
        # A latent represention.
        self.latent = nn.Sequential(
            nn.Linear(30, 40),
            nn.Sigmoid(),
            nn.Linear(40, 80),
            nn.Sigmoid(),
            nn.Linear(80, 320),
            nn.Sigmoid(),
        )

        # Now we go from latent
        # + pos to density
        self.func = nn.Sequential(
            nn.Linear(3 + self.latent_dim, 20),
            nn.Sigmoid(),
            nn.Linear(20, 4),
            nn.Sigmoid(),
        )

    def forward(self, X, z):
        # Let's assume one batch
        # latent will be b x latent_dim.
        latent = self.latent(z)

        input = X.reshape(-1, 3)
        # ok so now we should have changed the size
        latent = latent.expand(X.shape[0], input.shape[0], self.latent_dim)[0]
        concatted = torch.concat([input, latent], axis=1)

        sampled = self.func(concatted)
        # self.func(
        # for f_l, c_l in zip(self.func, self.conditioner):
        #    input = f_l(z)
        # input = f_l(input)

        # print(input.shape)

        new_shape = list(X.shape)
        new_shape[-1] = 4
        # X comes in as a batch X ... X 3
        return sampled.reshape(new_shape)

    def gen(self, ray_bundle, z, **kwargs):
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        res = self(rays_points_world, z)
        return res[..., 0:1], res[..., 1:]

    def random_gen(self, ray_bundle, **kwargs):
        # first convert the ray origins, directions and lengths
        # to 3D ray point locations in world coords
        return self.gen(ray_bundle, torch.randn((30,), device=self.device))
