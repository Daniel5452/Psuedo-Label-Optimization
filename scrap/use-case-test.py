#!/usr/bin/env python3

from onedl.client import connect_to_project
from onedl.zoo.eval import EvaluationConfig, REPORT_TEMPLATE


def generate_validation_pseudo_labels():
    PROJECT_NAME = "vbti/interreg-broccoli"
    VALIDATION_DATASET = "broccoli-semantic-segmentation-part4-may23-val"
    MODEL_UID = "smoky-shepherd-0"
    MIN_CONFIDENCE = 0.75
    OUTPUT_DATASET_NAME = f"{VALIDATION_DATASET}-pseudo-labels"

    client = connect_to_project(PROJECT_NAME)

    # Run inference
    config = EvaluationConfig(
        model_name=MODEL_UID,
        dataset_name=VALIDATION_DATASET,
        report_template=REPORT_TEMPLATE.EMPTY,
        batch_size=1
    )

    job = client.jobs.submit(config)
    predictions_dataset_name = client.jobs.get_dataset(job)

    # Load and filter predictions
    predictions_dataset = client.datasets.load(predictions_dataset_name)

    filtered_predictions = []
    for pred in predictions_dataset.predictions:
        if hasattr(pred, 'filter_by_confidence'):
            filtered_pred = pred.filter_by_confidence(MIN_CONFIDENCE)
            filtered_predictions.append(filtered_pred)
        else:
            filtered_predictions.append(pred)

    predictions_dataset.predictions = filtered_predictions
    predictions_dataset.targets = predictions_dataset.predictions

    # Save dataset
    client.datasets.save(OUTPUT_DATASET_NAME, predictions_dataset, exist="overwrite")
    client.datasets.push(OUTPUT_DATASET_NAME, push_policy='version')

    print(f"Saved: {OUTPUT_DATASET_NAME}")
    return OUTPUT_DATASET_NAME


if __name__ == "__main__":
    generate_validation_pseudo_labels()