from __future__ import annotations

import bentoml
from PIL.Image import Image
from PIL.ImageDraw import Draw

with bentoml.importing():
    import numpy as np
    import torch
    from torchvision.transforms import v2


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class SpyAircraftDetector:
    """
    Detects spy aircraft in images using a BentoML model."""

    # loading the fine-tuned model
    bento_model = bentoml.models.get("aircraft_detection_faster_rcnn:latest")

    def __init__(self):
        """
        Initializes the model with device, loads the BentoML model, and sets up image transforms.

            Args:
                None

            Returns:
                None
        """
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = bentoml.pytorch.load_model(self.bento_model)

        # image transforms list
        self.processor = v2.Compose(
            [
                v2.Resize((800, 800)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @bentoml.api
    async def predict(self, img: Image) -> Image:
        """predicts bounding boxes for given image"""

        # switching model to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # preparing image to draw predicted bounding boxes
        img = img.resize(size=(800, 800))
        draw = Draw(img)

        # image processing, torch.unsqueeze are required for avoid error caused by tensor dimensions
        processed_img = self.processor(img.convert(mode="RGB"))
        processed_img = list(torch.unsqueeze(processed_img, dim=0).to(self.device))

        # getting the predictions
        outputs = self.model(processed_img)

        # access to predicted bounding boxes and their scores
        predictions = outputs[0]["boxes"].data.cpu().numpy()
        scores = outputs[0]["scores"].data.cpu().numpy()

        predictions = predictions[scores >= 0.5].astype(np.int32)
        scores = scores[scores >= 0.5]
        print(predictions, scores)

        # draw each bounding box and it's score in our predictions
        for box, score in zip(predictions, scores):

            box = tuple(map(int, box))

            draw.rectangle(xy=box, outline="blue", width=2)
            draw.text(
                xy=(box[0], box[1]),
                text=f"box score: {round(float(score), 4)}",
                stroke_width=0.35,
                fill="black",
            )

        return img
