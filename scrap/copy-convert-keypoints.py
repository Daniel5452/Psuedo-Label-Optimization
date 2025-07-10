#!/usr/bin/env python3
"""
Script to convert KeyPointDetectionInstances to ObjectDetectionInstances
and save the converted dataset to another OneDL project.
"""

from onedl.client import connect_to_project
from onedl.data_types import KeyPointDetectionInstances, ObjectDetectionInstances


def convert_keypoints_to_object_detections(dataset):
    """
    Converts KeyPointDetectionInstances to ObjectDetectionInstances in a dataset.

    Args:
        dataset: OneDL dataset containing KeyPointDetectionInstances

    Returns:
        dataset: Modified dataset with ObjectDetectionInstances
    """
    converted_targets = []

    for i, kp_instances in enumerate(dataset.targets):
        if not isinstance(kp_instances, KeyPointDetectionInstances):
            raise TypeError(f"Target at index {i} is not KeyPointDetectionInstances, found {type(kp_instances)}")

        od_instances = ObjectDetectionInstances()
        for kp in kp_instances:
            od_instances.append(bbox=kp.bbox, label=kp.label)

        converted_targets.append(od_instances)

    dataset.targets = converted_targets
    return dataset


def main():
    """
    Main function to load dataset, convert it, and save to target project.
    """
    # Configuration
    SOURCE_PROJECT = "daniel-osman---streamlining-annotation-bootstrapping/use-1"
    TARGET_PROJECT = "daniel-osman---streamlining-annotation-bootstrapping/pipeline-test"
    SOURCE_DATASET_NAME = "validation-0--cpu--072bc"
    TARGET_DATASET_NAME = "demo-val-object-detection"

    print(f"Starting dataset conversion process...")
    print(f"Source project: {SOURCE_PROJECT}")
    print(f"Target project: {TARGET_PROJECT}")
    print(f"Source dataset: {SOURCE_DATASET_NAME}")
    print(f"Target dataset: {TARGET_DATASET_NAME}")

    try:
        # Connect to source project and load dataset
        print("\n1. Connecting to source project...")
        source_client = connect_to_project(SOURCE_PROJECT)

        print(f"2. Loading dataset '{SOURCE_DATASET_NAME}'...")
        dataset = source_client.datasets.load(SOURCE_DATASET_NAME, pull_blobs=True)

        print(f"   Dataset loaded successfully. Shape: {len(dataset)} samples")
        print(f"   Original target type: {type(dataset.targets[0]) if dataset.targets else 'No targets'}")

        # Convert keypoints to object detections
        print("\n3. Converting KeyPointDetectionInstances to ObjectDetectionInstances...")
        converted_dataset = convert_keypoints_to_object_detections(dataset)

        print(f"   Conversion completed successfully!")
        print(
            f"   New target type: {type(converted_dataset.targets[0]) if converted_dataset.targets else 'No targets'}")

        # Connect to target project
        print("\n4. Connecting to target project...")
        target_client = connect_to_project(TARGET_PROJECT)

        # Save converted dataset to target project
        print(f"5. Saving converted dataset as '{TARGET_DATASET_NAME}'...")
        saved_dataset_name = target_client.datasets.save(
            TARGET_DATASET_NAME,
            dataset=converted_dataset,
            exist="overwrite"
        )

        print(f"   Dataset saved successfully as: {saved_dataset_name}")

        # Push dataset to remote
        print("6. Pushing dataset to remote...")
        target_client.datasets.push(saved_dataset_name)

        print(f"\nSuccess! Dataset conversion completed.")
        print(f"   Converted dataset '{SOURCE_DATASET_NAME}' from keypoint detection to object detection")
        print(f"   Saved as '{saved_dataset_name}' in project '{TARGET_PROJECT}'")

        return saved_dataset_name

    except Exception as e:
        print(f"\n Error during conversion process: {str(e)}")
        raise


if __name__ == "__main__":
    main()