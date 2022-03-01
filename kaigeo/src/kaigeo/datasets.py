import json
from urllib.request import urlopen

import numpy as np
import torch
import torchvision
import torchvision.transforms
from PIL import Image


def load_session1(path):
    with open(path) as f:
        data = json.load(f)

    frames = []
    player = []

    for messagePack in data["messages"]:
        image = Image.open(urlopen(messagePack["frame"]["data"]))
        messages = messagePack["messages"]
        # Pick the one of type player.
        for m in messages:
            if m["type"] == "minecraft:player":
                player.append(m)
                break
        else:
            raise Exception("Could not find player")
        # player.append(messages[0])
        frames.append(torchvision.transforms.ToTensor()(image))

    transformed_frames = [torchvision.transforms.Resize((128, 128))(f) for f in frames]
    target_images = torch.stack(transformed_frames)
    target_images = target_images.permute(0, 2, 3, 1)[:, :, :, :3]

    eye = np.array([[p["eyeX"], p["eyeY"], p["eyeZ"]] for p in player])
    forward = np.array([[p["forwardX"], p["forwardY"], p["forwardZ"]] for p in player])
    look = np.array(
        [[p["lookAngleX"], p["lookAngleY"], p["lookAngleZ"]] for p in player]
    )

    return target_images, player, eye, forward, look, data

