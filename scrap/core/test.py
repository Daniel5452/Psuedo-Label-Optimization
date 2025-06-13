from onedl.client import connect_to_project
from onedl.core.structures.instances.object_detection import (
    ObjectDetectionInstances,
    KeyPointDetectionInstances
)

def convert_keypoints_to_object_detections_and_save(client, dataset_name):
    """
    Converts a dataset from KeyPointDetectionInstances to ObjectDetectionInstances,
    then saves and pushes it back to OneDL under the same name.

    Args:
        client: OneDL client instance
        dataset_name: Name of the dataset to convert

    Returns:
        str: Name of the converted dataset
    """
    dataset = client.datasets.load(dataset_name, pull_policy="missing")

    converted_targets = []
    for i, kp_instances in enumerate(dataset.targets):
        if not isinstance(kp_instances, KeyPointDetectionInstances):
            raise TypeError(f"Target at index {i} is not KeyPointDetectionInstances, found {type(kp_instances)}")

        od_instances = ObjectDetectionInstances()
        for kp in kp_instances:
            od_instances.append(bbox=kp.bbox, label=kp.label)

        converted_targets.append(od_instances)

    dataset.targets = converted_targets

    client.datasets.save(dataset_name, dataset, exist="versions")
    client.datasets.push(dataset_name, push_policy="version")

    print(f"Dataset '{dataset_name}' successfully converted to ObjectDetectionInstances and pushed.")
    return dataset_name


client = connect_to_project('daniel-osman---streamlining-annotation-bootstrapping/testing')
converted_dataset = convert_keypoints_to_object_detections_and_save(client, "train-f0")