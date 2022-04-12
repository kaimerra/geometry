import requests

import requests, json
import torch
import torchvision
from torchvision.io import read_video
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
)
import itertools
import matplotlib.patches as patches
from ipywidgets import interact
import ipywidgets as widgets


from kaigeo import detect

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()

feeds = requests.get("https://backend.kaimerra.com/feed").json()

import subprocess

ids = ['62559d5677b919d0598885e7', '62559c7377b919d0598885e6']

# lets process one at a time.
for feed_item in feeds:
    print(feed_item)
    if len(feed_item.get('annotations', [])) > 0:
        continue

    if not feed_item['_id'] in ids:
        continue

    print('processing', feed_item['_id'])

    url = feed_item["video_url"]
    r = requests.get(url, allow_redirects=True)
    in_path = f'vids/{feed_item["_id"]}'
    out_path = f'out_vids/{feed_item["_id"]}.mp4'
    with open(in_path, "wb") as f:
        f.write(r.content)

    subprocess.run(
        f"ffmpeg -i {in_path} -vf scale=320:240,setsar=1:1 {out_path}",
        shell=True,
        check=True,
    )

    objs = detect.detect_for_api(headers, feed_item["_id"], model, out_path)

    print("PROCESSED ", feed_item["_id"])
