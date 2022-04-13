import requests

import os
import requests
import torchvision
import numpy as np
import logging
import subprocess
import tempfile

from contextlib import contextmanager


from kaigeo import detect

logger = logging.getLogger(__name__)

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()


class ExceptionThresholder:
    def __init__(self, limit=5):
        self.count = 0
        self.limit = limit

    @contextmanager
    def guard(self):
        try:
            yield
        except Exception:
            logger.exception("Caught exception (%d/%d)!", self.count, self.limit)
            self.count += 1

            if self.count > self.limit:
                raise Exception(f"Exception limit exceeded ({self.count}, {self.limit}")


def run():
    auth_client_id = os.environ["AUTH_CLIENT_ID"]
    auth_client_secret = os.environ["AUTH_CLIENT_SECRET"]
    backend = os.environ.get("GEOMETRY_BACKEND", "https://backend.kaimerra.com")
    payload = {
        "client_id": auth_client_id,
        "client_secret": auth_client_secret,
        "audience": "https://backend.kaimerra.com",
        "grant_type": "client_credentials",
    }

    resp = requests.post("https://dev-ajfk-6oq.us.auth0.com/oauth/token", json=payload)

    bearer = resp.json()["access_token"]

    headers = {"authorization": f"Bearer {bearer}"}

    feed = requests.get(f"{backend}/feed", headers=headers).json()

    thresholder = ExceptionThresholder()

    # For now, lets process anything that doesn't have any annotations.
    for feed_item in feed:
        with thresholder.guard():
            feed_item_id = feed_item["_id"]

            logger.debug("Determining if we should process %s", feed_item_id)
            if "annotations" in feed_item:
                logger.debug("Skipping processing %s", feed_item_id)
                continue

            logger.debug("Processing %s", feed_item_id)
            process_item(headers, backend, feed_item)
            logger.debug("Done processing %s", feed_item_id)


def process_item(headers, backend, feed_item):
    with tempfile.TemporaryDirectory() as workdir:
        print("processing", feed_item["_id"])

        url = feed_item["video_url"]

        # 1. Download the video
        r = requests.get(url, allow_redirects=True)
        in_path = os.path.join(workdir, "input_vid")
        with open(in_path, "wb") as f:
            f.write(r.content)

        # 2. Convert the video
        out_path = os.path.join(workdir, "output_vid.mp4")
        subprocess.run(
            f"ffmpeg -y -i {in_path} -vf scale=320:240,setsar=1:1 {out_path}",
            shell=True,
            check=True,
        )

        # 3. Run object detection
        detect.detect_for_api(
            headers, backend, feed_item["_id"], model, out_path, top=30
        )
