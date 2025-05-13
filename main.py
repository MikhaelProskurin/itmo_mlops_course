from utils.utils import get_image_paths
from core.experiment import Experiment
from core.custom_datasets import AirfieldDataset
from core.preprocessing import get_processed_annotations

import torch
from torch import optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn, faster_rcnn

from clearml import Task


model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
    in_features, num_classes=2
)
# model.load_state_dict(torch.load(f="models_history/models_torch/faster_rcnn_tuned.pth", weights_only=True))

# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
optimizer = optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.01,
    momentum=0.9,
    weight_decay=0.00001,
)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)


def main(
    mode: str,
    train_sample_size: str = "medium",
    eval_sample_size: str = "medium",
    export_to_onnx: bool = False,
    export_to_bento: bool = False,
) -> None:
    """
    Main function to handle training or evaluation of the model.

        Args:
            mode:  Specifies whether to 'train' or 'eval'uate the model.
            train_sample_size: The size of the training dataset ("small", "medium", or "large"). Defaults to "medium".
            eval_sample_size: The size of the evaluation dataset ("small", "medium", or "large"). Defaults to "medium".
            export_to_onnx:  A boolean indicating whether to export the model to ONNX format. Defaults to False.
            export_to_bento: A boolean indicating whether to export the model to BentoML format. Defaults to False.

        Returns:
            None
    """

    if mode == "train":

        # Clearml task initiation
        task = Task.init(
            project_name="itmo_mlops_2024",
            task_name="training",
            task_type=Task.TaskTypes.training,
            tags=["fasterrcnn_resnet50_fpn", "detection"],
        )

        dataframe = get_processed_annotations(
            images_set=f"src/prepared_samples/train_{train_sample_size}.txt",
            xml_path="src/annotations",
        )

        # uploading training set to Clearml
        task.upload_artifact(
            name=f"training sample: size -> {train_sample_size}",
            artifact_object=dataframe,
        )

        torch_dataset = AirfieldDataset(
            dataframe=dataframe,
            paths=get_image_paths("src/prepared_samples/train_medium.txt"),
        )

        # configuring the experiment
        experiment = Experiment(
            dataset=torch_dataset,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=task.get_logger(),
            config={
                "model_name": "fasterrcnn_resnet50_fpn",
                "optim_name": "sgd",
                "checkpoint_path": "models_history/models_torch",
                "train_loader_batch": 8,
                "eval_loader_batch": 4,
                "num_epochs": 5,
                "mode": "train",
            },
        )

        # experiment.execute()
        if export_to_onnx:
            experiment.export_to_onnx(model_name="faster_rcnn_tuned_v3")

        if export_to_bento:
            experiment.export_to_bento(
                bento_name="aircraft_detection_faster_rcnn",
                tags={"stage": "dev", "team": "cv"},
            )

        task.close()

    if mode == "eval":

        # Clearml task initiation
        task = Task.init(
            project_name="itmo_mlops_2024",
            task_name="evaluation",
            task_type=Task.TaskTypes.inference,
            tags=["fasterrcnn_resnet50_fpn", "detection"],
        )

        dataframe = get_processed_annotations(
            images_set=f"src/prepared_samples/test_{eval_sample_size}.txt",
            xml_path="src/annotations",
        )

        # uploading evaluation set to Clearml
        task.upload_artifact(
            name=f"eval sample: size -> {eval_sample_size}", artifact_object=dataframe
        )

        torch_dataset = AirfieldDataset(
            dataframe=dataframe,
            paths=get_image_paths("src/prepared_samples/test_medium.txt"),
        )

        # configuring the experiment
        experiment = Experiment(
            dataset=torch_dataset,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=task.get_logger(),
            config={
                "model_name": "fasterrcnn_resnet50_fpn",
                "optim_name": "sgd",
                "eval_loader_batch": 4,
                "mode": "eval",
            },
        )

        # getting the predictions
        predictions = experiment.execute()

        # uploading predictions df to Clearml
        task.upload_artifact(name=f"predictions dataframe", artifact_object=predictions)
        task.close()


if __name__ == "__main__":
    # training
    model.load_state_dict(
        torch.load(
            f=f"models_history/models_torch/fasterrcnn_resnet50_fpn_sgd_tuned.pth",
            weights_only=True,
        )
    )
    main(mode="train", train_sample_size="medium")

    # evaluation
    model.load_state_dict(
        torch.load(
            f=f"models_history/models_torch/fasterrcnn_resnet50_fpn_sgd_tuned.pth",
            weights_only=True,
        )
    )
    main(mode="eval", eval_sample_size="medium")
