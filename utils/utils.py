import time

import pandas as pd
from PIL import Image, ImageDraw

def draw_image_boxes(
        predicted: pd.DataFrame,
        ground_truth: pd.DataFrame = None,
        bbox_width: int = 2,
        config: dict = None,
) -> None:
    """
       Function which draw image and its bboxes by PIL.
       You could specify bbox line width, text and line colour
       :param config:
       :param bbox_width:
       :param ground_truth:
       :param predicted:
    """
    default_label = "aircraft"

    if not config:
        config = {
            # "pred_obj_title": "",
            # "pred_text_color": "",
            # "pred_box_color": "",
            "metric_name": "IoU",
            # "gt_obj_title": "",
            # "gt_text_color": "",
            # "gt_box_color": "",
        }

    image_ids = predicted["image_id"].unique()

    for img_id in image_ids:

        draw = ImageDraw.Draw(Image.open(f"src/data/{img_id}.jpg"))

        predictions_for_img = predicted[predicted["image_id"] == img_id]

        for index, row in predictions_for_img.iterrows():
            ann_pred = tuple(map(int, row["boxes"]))

            draw.rectangle(xy=ann_pred, outline=config.get("pred_box_color", "blue"), width=bbox_width)

            draw.text(
                xy=(ann_pred[0], ann_pred[1]),
                text=config.get("pred_obj_title", default_label),
                fill=config.get("pred_text_color", "black"),
                stroke_width=0.2
            )

            metric = config.get('metric_name', "IoU")
            draw.text(
                xy=(ann_pred[0], ann_pred[3]),
                text=f"{metric}: -> {row[metric] if metric else 'no data'}",
                fill=config.get("metric_color", "black"),
                stroke_width=0.2
            )


        if ground_truth is not None:

            ground_truth_for_img = ground_truth[ground_truth["image_id"] == img_id]

            for index, row in ground_truth_for_img.iterrows():

                ann_gt = tuple(map(int, row["boxes"]))

                draw.rectangle(xy=ann_gt, outline=config.get("gt_box_color", "red"), width=bbox_width)

                draw.text(
                    xy=(ann_gt[0], ann_gt[1]),
                    text=config.get("gt_obj_title", default_label),
                    fill=config.get("gt_text_color", "red"),
                    stroke_width=0.2
                )

        draw._image.show()
        time.sleep(2)


def get_image_paths(indexes_file: str) -> list[str]:
    """
        Extracts paths for given images indexes
    """
    with open(file=indexes_file, mode="r") as f:
        return [f"src/data/{ind}.jpg" for ind in f.read().split("\n")]

class Averager:
    """
        loss averager
    """
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))
