from collections import defaultdict
import base64
from io import BytesIO

import requests
import torch
import torchvision
import torchvision.transforms as T
import itertools

CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def _get_box(box):
    x = box[0]
    y = box[1]
    width = box[2] - box[0]
    height = box[3] - box[1]
    return x, y, width, height


def _get_extracted_frame(frame, box):
    return frame[:, int(box[0]) : int(box[2]), int(box[1]) : int(box[3])]


ToPILImage = T.ToPILImage()


def _get_encoded_png(tensor) -> str:
    image = ToPILImage(tensor)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class AnnotationRanker:
    """Keep the highest scoring objects in each category. We prioritize keeping representatives
    from each category first, then fall back on ranking across numbers"""

    def __init__(self, top: int = 10):
        self.top = top

        # keep 1 per category.
        # if we sort by category and then by score...
        # this would mean that

        self.things = defaultdict(list)

    def add(self, category, score, new_thing):
        # We are adding one new thing. Either we have enough space,
        # or we have to figure out the one thing to kick out.
        # Add it in, then look for the one candidate to remove.
        self.things[category].append((score, new_thing))

        # We take out the lowest scoring item, unless
        # that totally removes a category... unless
        # we have to remove a category because there are
        # 1 of everything.
        lens = [len(t) for t in self.things.values()]
        more_than_1 = any([l > 1 for l in lens])
        if sum(lens) <= self.top:
            return

        # otherwise we gotta delete something...
        # pick something with the lowest score, but
        # with more than one category.. unless
        # there is only the smallest category

        delete_thing, delete_i, delete_score = (
            None,
            None,
            2.0,
        )  # The highest score is 1.0
        for thing_k, thing in self.things.items():
            # if there is at least one category
            # bigger than 1. we
            if len(thing) > 1 or not more_than_1:
                for i, (s, _) in enumerate(thing):
                    if s < delete_score:
                        delete_thing = thing_k
                        delete_i = i
                        delete_score = s

        del self.things[delete_thing][delete_i]

    def get_things(self):
        return sum([[t[1] for t in thing] for thing in self.things.values()], [])


def detect(model, video_file: str, top: int = 10):
    """Returns the top objects detected in a video."""
    reader = torchvision.io.VideoReader(video_file)
    ranker = AnnotationRanker(top)

    # we only need to process every few frames
    for data in itertools.islice(reader, None, None, 25):
        frame = data["data"]
        time = data["pts"]

        with torch.no_grad():
            outputs = model(frame.permute(0, 2, 1)[None] / 255.0)[0]

        for box, label_idx, mask, score in zip(
            outputs["boxes"], outputs["labels"], outputs["masks"], outputs["scores"]
        ):
            label = CATEGORY_NAMES[label_idx]
            box = box.detach().numpy()
            x, y, width, height = _get_box(box)

            # strangely this can happen.
            if width == 0 or height == 0:
                print("skipping", width, height)
                continue

            score = float(score.detach())

            extracted_frame = _get_extracted_frame(frame, box)
            extracted_mask = _get_extracted_frame(mask, box)

            if extracted_frame.shape[1] == 0 or extracted_frame.shape[2] == 0:
                print("skipping", extracted_frame.shape)
                continue

            if extracted_mask.shape[1] == 0 or extracted_mask.shape[2] == 0:
                print("skipping", extracted_mask.shape)
                continue

            annotation = {}
            annotation["frameImageData"] = {
                "width": float(frame.shape[1]),
                "height": float(frame.shape[2]),
                "data": frame,  # _get_encoded_png(frame),
            }
            annotation["extractedImageData"] = {
                "width": float(extracted_frame.shape[1]),
                "height": float(extracted_frame.shape[2]),
                "data": extracted_frame,  # _get_encoded_png(extracted_frame),
            }
            annotation["maskImageData"] = {
                "width": float(mask.shape[1]),
                "height": float(mask.shape[2]),
                "data": mask,
            }
            annotation["extractedMaskImageData"] = {
                "width": float(extracted_mask.shape[1]),
                "height": float(extracted_mask.shape[2]),
                "data": extracted_mask,
            }
            annotation["category"] = label
            annotation["score"] = float(score)
            annotation["x"] = float(x)
            annotation["y"] = float(y)
            annotation["width"] = float(width)
            annotation["height"] = float(height)
            annotation["time"] = float(time)

            ranker.add(label, score, annotation)

    return ranker.get_things()


def detect_for_api(headers, backend, feedItemId, model, video_file: str, top: int = 30):
    annotations = detect(model, video_file, top)

    for annotation in annotations:
        annotation["frameImageData"]["data"] = _get_encoded_png(
            annotation["frameImageData"]["data"]
        )
        annotation["extractedImageData"]["data"] = _get_encoded_png(
            annotation["extractedImageData"]["data"]
        )

        annotation["maskImageData"]["data"] = _get_encoded_png(
            annotation["maskImageData"]["data"]
        )
        annotation["extractedMaskImageData"]["data"] = _get_encoded_png(
            annotation["extractedMaskImageData"]["data"]
        )

        annotation["feedItemId"] = feedItemId

        requests.post(
            f"{backend}/feed/annotate",
            json=annotation,
            headers=headers,
        ).raise_for_status()

    return annotations
