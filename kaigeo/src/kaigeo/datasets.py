from typing import Tuple
from dataclasses import dataclass
import os

import json
from urllib.request import urlopen

import numpy as np
import torch
import torchvision
import torchvision.transforms
from PIL import Image

files_root = os.path.join(*os.path.split(__file__)[:-1], "files")


def get_file_path(name: str) -> str:
    return os.path.join(files_root, name)


@dataclass
class Session:
    target_images: torch.Tensor
    player: torch.Tensor
    eye: torch.Tensor
    forward: torch.Tensor
    look: torch.Tensor


def load_session1() -> Session:
    return load_session("data-1645820672906.json")


def load_session2() -> Session:
    return load_session("data-1646180076202.json")


def load_session(path: str) -> Session:
    with open(get_file_path(path)) as f:
        data = json.load(f)

    frames = []
    player = []

    for messagePack in data["messages"]:
        if messagePack["frame"] is None:
            continue
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
        # frames.append(torchvision.transforms.ToTensor()(image))
        frames.append(torchvision.transforms.Resize((128, 128))(image))

    transformed_frames = [torchvision.transforms.ToTensor()(f) for f in frames]

    target_images = torch.stack(transformed_frames)
    target_images = target_images.permute(0, 2, 3, 1)[:, :, :, :3]

    eye = np.array([[p["eyeX"], p["eyeY"], p["eyeZ"]] for p in player])
    forward = np.array([[p["forwardX"], p["forwardY"], p["forwardZ"]] for p in player])
    look = np.array(
        [[p["lookAngleX"], p["lookAngleY"], p["lookAngleZ"]] for p in player]
    )

    return Session(
        target_images=target_images,
        player=player,
        eye=eye,
        forward=forward,
        look=look,
    )
