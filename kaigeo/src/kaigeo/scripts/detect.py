from tkinter import W
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
logger.setLevel(logging.DEBUG)
logging.basicConfig()

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
    backend = os.environ.get("BACKEND", "https://backend.kaimerra.com")
    payload = {
        "client_id": auth_client_id,
        "client_secret": auth_client_secret,
        "audience": "https://backend.kaimerra.com",
        "grant_type": "client_credentials",
    }

    resp = requests.post("https://dev-ajfk-6oq.us.auth0.com/oauth/token", json=payload)
    resp.raise_for_status()

    bearer = resp.json()["access_token"]

    print("bearer", bearer)

    headers = {"authorization": f"Bearer {bearer}"}

    resp = requests.get(f"{backend}/feed", headers=headers)
    resp.raise_for_status()

    feed = resp.json()

    thresholder = ExceptionThresholder()

    # For now, lets process anything that doesn't have any annotations.
    for feed_item in feed:
        with thresholder.guard():
            feed_item_id = feed_item["_id"]

            logger.info("Determining if we should process %s", feed_item_id)
            if "annotationsState" in feed_item:
                logger.info(
                    "Skipping processing feedItemId: %s due to annotationState: %s",
                    feed_item_id,
                    feed_item["annotationsState"],
                )
                continue

            logger.info("Processing %s", feed_item_id)
            process_item(headers, backend, feed_item)
            logger.info("Done processing %s", feed_item_id)

    logger.info("Done processing all feed items.")


def process_item(headers, backend, feed_item):
    with tempfile.TemporaryDirectory() as workdir:
        logger.info("Processing %s", feed_item["_id"])

        # 1. Marking the feed item as processing.
        logger.info("Marking %s as processing", feed_item["_id"])
        requests.post(
            f"{backend}/feed/annotate/state",
            json={"feedItemId": feed_item["_id"], "annotationsState": "processing"},
            headers=headers,
        ).raise_for_status()

        url = feed_item["video_url"]

        # 2. Download the video
        logger.info("Downloading %s...", url)
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        in_path = os.path.join(workdir, "input_vid")
        with open(in_path, "wb") as f:
            f.write(r.content)
        logger.info("Down downloading %s...", url)

        # 3. Convert the video
        logger.info("Converting... %s", in_path)
        out_path = os.path.join(workdir, "output_vid.mp4")
        subprocess.run(
            f"ffmpeg -y -i {in_path} -vf scale=320:240,setsar=1:1 {out_path}",
            shell=True,
            check=True,
        )
        logger.info("Done converting... %s", in_path)

        # 4. Run object detection
        logger.info("Detecting %s...", out_path)
        detect.detect_for_api(
            headers, backend, feed_item["_id"], model, out_path, top=30
        )
        logger.info("Done detecting %s...", out_path)

        # 5. Mark the feed item as complete.
        logger.info("Marking %s as complete.", feed_item["_id"])
        requests.post(
            f"{backend}/feed/annotate/state",
            json={"feedItemId": feed_item["_id"], "annotationsState": "complete"},
            headers=headers,
        ).raise_for_status()
