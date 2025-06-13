
from onedl.client import connect_to_project
from onedl.core.structures.instances.object_detection import (
    ObjectDetectionInstances,
    KeyPointDetectionInstances
)
project_name = "daniel-osman---streamlining-annotation-bootstrapping/testing"
initial_annotated_dataset_name = "initial-annotations"


client = connect_to_project(project_name)
data = client.datasets.load(initial_annotated_dataset_name)
converted_targets = []
for i, kp_instances in enumerate(data.targets):
    if not isinstance(kp_instances, KeyPointDetectionInstances):
        raise TypeError(f"Target at index {i} is not KeyPointDetectionInstances, found {type(kp_instances)}")

    od_instances = ObjectDetectionInstances()
    for kp in kp_instances:
        od_instances.append(bbox=kp.bbox, label=kp.label)

    converted_targets.append(od_instances)

data.targets = converted_targets

client.datasets.save(initial_annotated_dataset_name, data, exist="versions")
client.datasets.push(initial_annotated_dataset_name, push_policy="version")

print(f"Init dataset '{initial_annotated_dataset_name}' successfully converted to ObjectDetectionInstances and pushed.")
