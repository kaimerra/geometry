import torchvision
import torchvision.transforms
import json

import numpy as np


from matplotlib import pyplot as plt

from itertools import takewhile, islice

import scipy.interpolate

from itertools import chain

from urllib.request import urlopen

from torchvision import transforms
from PIL import Image

# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCGridRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)

def load_session1(path):
    with open(path) as f:
        data = json.load(f)
        
    frames = []
    player = []
    
    for messagePack in data['messages']:
        image = Image.open(urlopen(messagePack['frame']['data']))
        messages = messagePack['messages']
        # Pick the one of type player.
        for m in messages:
            if m['type'] == 'minecraft:player':
                player.append(m)
                break
        else:
            raise Exception("Could not find player")
        #player.append(messages[0])
        frames.append(transforms.ToTensor()(image))

    transformed_frames = [torchvision.transforms.Resize((128, 128))(f) for f in frames]
    target_images = torch.stack(transformed_frames)
    target_images = target_images.permute(0, 2, 3, 1)[:, :, :, :3]

    eye = np.array([[p['eyeX'], p['eyeY'], p['eyeZ']] for p in player])
    forward = np.array([[p['forwardX'], p['forwardY'], p['forwardZ']] for p in player])
    look = np.array([[p['lookAngleX'], p['lookAngleY'], p['lookAngleZ']] for p in player])

    return target_images, player, eye, forward, look, data
