from utils.utils import Averager, collate_fn

from typing import Union
from time import time

import bentoml

import torch
import torch.onnx
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from clearml import Logger
from PIL import Image, ImageDraw


class Experiment:

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler,
            logger: Logger,
            config: dict
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.collate_fn = collate_fn
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.config = config

    """
        Class that builds our experiment with neural network.
        Just specify parameters of experiment such as optimizer,
        device, ml/dl model, etc. 
    """

    def train(self, dataloader: DataLoader) -> None:
        """
            Implements network training process
            :param dataloader: PyTorch standard DataLoader which contains custom dataset
        """

        # setting some counters and such things
        itr = 1
        start_time = time()
        model_name, optim_name, checkpoint_path, num_epochs = self.get_constants_from_config()

        # switching model into train mode
        self.model.to(self.device)
        self.model.train()

        # losses history
        loss_hist = Averager()

        self.logger.report_text("Training process is started!")

        try:

            # training loop
            for epoch in range(num_epochs):

                loss_hist.reset()

                for images, targets, image_ids in dataloader:

                    images = list(image.to(self.device) for image in images)

                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    # forward pass
                    loss_dict = self.model(images, targets)

                    # compute and send average batch loss
                    losses = sum(loss for loss in loss_dict.values())
                    loss_value = losses.item()

                    loss_hist.send(loss_value)

                    # backward pass
                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                    # logging loss
                    self.logger.report_scalar(
                        title="training loss",
                        series="avg batch loss",
                        iteration=itr,
                        value=loss_hist.value
                    )

                    # logging image samples from training set
                    [
                        self.logger.report_image(
                            title="image",
                            series=f"Batch of images on itr: {itr}",
                            iteration=itr,
                            image=Image.open(f"src/data/{img}.jpg")
                        )
                        for img
                        in image_ids
                    ]

                    self.logger.report_text(f"Iteration: {itr}")
                    itr += 1

                # logging learning rate
                self.logger.report_scalar(
                    title="learning rate",
                    series="lr",
                    iteration=itr,
                    value=self.lr_scheduler.get_last_lr()[0]
                )

                # learning rate update after each train epoch
                self.lr_scheduler.step()

                self.logger.report_text(f"Train Epoch: {epoch}, Iteration: {itr}")

        # checkpoint for unplanned or manual training interruption
        except Union[KeyboardInterrupt, TypeError, Exception]:

            torch.save(
                self.model.state_dict(),
                f=f"{checkpoint_path}/{model_name}_{itr}.pth"
            )

            self.logger.report_text("Process are finished manually! Model saved into checkpoint")

        # model saving after successfully train
        torch.save(self.model.state_dict(), f=f"{checkpoint_path}/{model_name}_{optim_name}_tuned.pth")

        # logging training time
        end_time = time()
        self.logger.report_text(f"Execution time: {(end_time - start_time) / 60} minutes")

    def get_predictions_dataset(self, dataloader: DataLoader, threshold: float = 0.5) -> pd.DataFrame:
        """
            Implements network performance evaluation process
            :param dataloader: PyTorch standard DataLoader which contains custom dataset
            :param threshold: filtering bound for predicted samples
        """
        predictions = {
            "image_id": [],
            "boxes": [],
            "IoU": []
        }

        # switching model into evaluation mode
        self.model.to(self.device)
        self.model.eval()

        itr = 1

        # evaluation loop
        for images, targets, image_ids in dataloader:

            images = list(image.to(self.device) for image in images)

            # get predictions of the model
            outputs = self.model(images)

            for i, image in enumerate(images):

                ground_truth = targets[i]["boxes"]
                output_boxes = outputs[i]["boxes"].data.cpu().numpy()

                scores = outputs[i]["scores"].data.cpu().numpy()

                output_boxes = output_boxes[scores >= threshold].astype(np.int32)

                image_id = image_ids[i]

                # Pillow class for drawing predicted bounding boxes
                draw = ImageDraw.Draw(Image.open(f"src/data/{image_id}.jpg"))

                # storing the iou's for each predicted object
                pred_objects_iou = []

                for box in output_boxes:

                    # iou computation for each predicted bbox
                    # torchvision.ops.box_iou returns Tensor after call
                    tensor_iou = box_iou(torch.tensor(ground_truth), torch.tensor(box.reshape(-1, 4)))

                    # tensor conversion into python dtype
                    iou = round(float(torch.max(tensor_iou)), 4)

                    pred_objects_iou.append(iou)

                    predictions["image_id"].append(image_id)
                    predictions["boxes"].append(box)
                    predictions["IoU"].append(iou)

                    draw.rectangle(xy=tuple(map(int, box)), outline="blue", width=2)
                    draw.text(xy=(box[0], box[1]), text=f"IoU: {iou}", fill="black", stroke=0.2)

                # logging the image with predicted bounding boxes
                self.logger.report_image(
                    title="image",
                    series=f"image sample: {image_id}",
                    iteration=itr,
                    image=draw._image
                )

                # scoring avg iou for image and logging it
                self.logger.report_scalar(
                    title="avg IoU (Jaccard index)",
                    series=f"avg IoU",
                    iteration=itr,
                    value=sum(pred_objects_iou) / (len(output_boxes) if len(output_boxes) > 0 else 1)
                )

                itr += 1

        return pd.DataFrame.from_dict(predictions)

    def execute(self) -> Union[pd.DataFrame, None]:
        """
            Method for triggering pipeline execution
        """
        self.logger.report_text(f"execution configuration: {self.config}")

        if self.config["mode"] == "train":

            loader = DataLoader(
                self.dataset,
                batch_size=self.config.get("train_loader_batch", 1),
                shuffle=False,
                collate_fn=self.collate_fn
            )
            self.train(dataloader=loader)

        if self.config["mode"] == "eval":

            loader = DataLoader(
                self.dataset,
                batch_size=self.config.get("eval_loader_batch", 1),
                shuffle=False,
                collate_fn=self.collate_fn
            )
            return self.get_predictions_dataset(dataloader=loader)

    def export_to_onnx(self, export_path: str = "models_history/models_onnx", model_name: str = "faster_rcnn_tuned") -> None:
        """
            Converting current model into ONNX format
            :param model_name: converted model file name
            :param export_path: destination directory for converted model

            *required ONNX version: 1.16.1; pip install onnx=1.16.1
        """
        model_to_export = self.model

        # this operation requires switching the model into eval mode
        model_to_export.eval()

        # input template for ONNX converted network should be the same
        input_tensor = torch.randn((4, 3, 800, 800), dtype=torch.float32, requires_grad=True)
        # input_tensor = input_tensor.to(self.device)

        torch.onnx.export(
            model_to_export,
            (input_tensor,),
            f"{export_path}/{model_name}.onnx",
            export_params=True,
            do_constant_folding=True,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamo=False
        )
        self.logger.report_text("Model successfully exported into ONNX format")

    def export_to_bento(self, bento_name: str, tags: dict) -> bentoml.Model:
        """Saves model into BentoML"""
        return bentoml.pytorch.save_model(name=bento_name, labels=tags, model=self.model)

    def get_constants_from_config(self) -> tuple:
        """Parses given constants into tuple"""

        model = self.config.get("model_name", "model")
        optim = self.config.get("optim_name", "")
        checkpoint_path = self.config.get("checkpoint_path", "models_history/models_torch")

        num_epochs = self.config.get("num_epochs", 5)

        return model, optim, checkpoint_path, num_epochs
