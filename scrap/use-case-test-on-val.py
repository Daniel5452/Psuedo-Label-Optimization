#!/usr/bin/env python3

from onedl.client import connect_to_project
from onedl.zoo.eval import EvaluationConfig, Device, REPORT_TEMPLATE


def create_pseudo_training_dataset():
    PROJECT_NAME = "vbti/interreg-broccoli"
    MODEL_UID = "sad-shape-0"
    VAL = "broccoli-semantic-segmentation-part4-may23-val-pseudo-labels"

    client = connect_to_project(PROJECT_NAME)

    # Run inference to generate pseudo-labels
    config = EvaluationConfig(
        model_name=MODEL_UID,
        dataset_name=VAL,
        report_template=REPORT_TEMPLATE.EMPTY,  # No evaluation, just predictions
        device=Device.CPU
    )

    job = client.jobs.submit(config)
    pseudo_train_dataset_name = client.jobs.get_dataset(job)

    print(f"Pseudo training dataset created: {pseudo_train_dataset_name}")
    return pseudo_train_dataset_name


if __name__ == "__main__":
    pseudo_train_dataset_name = create_pseudo_training_dataset()