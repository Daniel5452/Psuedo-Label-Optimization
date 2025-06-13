import os
import glob
import time
import json
from pathlib import Path
import requests


CVAT_URL = "https://cvat2.vbti.nl"
CVAT_API_URL = f"{CVAT_URL}/api"


def authenticate_cvat(username: str, password: str):
    session = requests.Session()
    resp = session.post(f"{CVAT_API_URL}/auth/login", json={"username": username, "password": password})
    if resp.status_code == 200:
        token = resp.json()["key"]
        session.headers.update({"Authorization": f"Token {token}"})
        print("Authenticated successfully.")
        return session
    else:
        raise RuntimeError("Authentication failed. Please check your credentials.")


def get_or_create_project(session, project_name: str):
    """Creates a new CVAT project. Assumes the project does not exist."""
    payload = {
        "name": project_name,
        "labels": [{"name": "object", "attributes": []}]
    }
    resp = session.post(f"{CVAT_API_URL}/projects", json=payload)
    if resp.status_code == 201:
        pid = resp.json()["id"]
        print(f"Project '{project_name}' created with ID: {pid}")
        return pid
    else:
        raise RuntimeError(f"Project creation failed: {resp.text}")


def create_task(session, project_id: int, task_name: str):
    payload = {
        "name": task_name,
        "project_id": project_id,
        "mode": "annotation",
        "overlap": 0,
        "segment_size": 0,
        "dimension": "2d"
    }
    resp = session.post(f"{CVAT_API_URL}/tasks", json=payload)
    if resp.status_code == 201:
        tid = resp.json()["id"]
        print(f"Task '{task_name}' created with ID: {tid}")
        return tid
    else:
        raise RuntimeError(f"Task creation failed: {resp.text}")


def upload_images(session, task_id: int, image_dir: str):
    url = f"{CVAT_API_URL}/tasks/{task_id}/data"
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.[jp][pn]g")) + glob.glob(os.path.join(image_dir, "*.jpeg")))
    if not image_files:
        raise FileNotFoundError(f"No images found in folder: {image_dir}")

    print(f"Uploading {len(image_files)} images...")
    files = {
        f"client_files[{i}]": (os.path.basename(path), open(path, "rb"), "image/jpeg")
        for i, path in enumerate(image_files)
    }

    payload = {
        "image_quality": 70,
        "sorting_method": "lexicographical"
    }
    resp = session.post(url, files=files, data=payload)

    # Close file handles
    for f in files.values():
        f[1].close()

    if resp.status_code != 202:
        raise RuntimeError(f"Image upload failed: {resp.text}")

    print("Images uploaded successfully.")


def upload_annotations(session, task_id: int, annotation_file: str):
    format_name = "COCO 1.0"
    url = f"{CVAT_API_URL}/tasks/{task_id}/annotations?format={format_name}&location=local"

    with open(annotation_file, "rb") as f:
        files = {
            "annotation_file": (os.path.basename(annotation_file), f)
        }

        print(f"Uploading annotations to task {task_id} ...")
        resp = session.post(url, files=files)

        if resp.status_code == 202:
            rq_id = resp.json().get("rq_id")
            print("Annotation upload started. Request ID:", rq_id)
            _check_annotation_status(session, rq_id)
        else:
            raise RuntimeError(f"Annotation upload failed: {resp.status_code} - {resp.text}")


def _check_annotation_status(session, rq_id: str):
    url = f"{CVAT_API_URL}/requests/{rq_id}"
    print("Waiting for annotation import to complete...")
    while True:
        resp = session.get(url)
        if resp.status_code != 200:
            print("Failed to check status:", resp.text)
            break
        state = resp.json().get("state")
        if state == "finished":
            print("Annotation import completed.")
            break
        elif state == "failed":
            raise RuntimeError("Annotation import failed.")
        else:
            print("Still processing...")
            time.sleep(5)


def export_coco_predictions(
    dataset_name: str,
    project_name: str,
    export_path: Path,
    json_filename: str = "annotations/instances_default.json",
    min_confidence: float = 0.0,
    shift_category_ids: bool = True,
):
    client = connect_to_project(project_name)
    raw_ds = client.datasets.load(dataset_name)
    raw_ds.targets = raw_ds.predictions.map(lambda x: x.filter_by_confidence(min_confidence))

    raw_ds.export_coco(path=export_path, json_filename=json_filename)
    print(f"COCO export complete (confidence ≥ {min_confidence})")

    json_path = export_path / json_filename
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    for image_entry in coco_data.get("images", []):
        image_entry["file_name"] = Path(image_entry["file_name"]).name

    if shift_category_ids:
        for category in coco_data.get("categories", []):
            category["id"] += 1
        for annotation in coco_data.get("annotations", []):
            annotation["category_id"] += 1
        print("Category IDs shifted by +1 for CVAT compatibility.")

    with open(json_path, "w") as f:
        json.dump(coco_data, f)
    print(f"COCO file saved → {json_path}")


def run_cvat_importer():
    print("CVAT Login")
    username = input("Enter your CVAT username: ").strip()
    password = input("Enter your CVAT password: ").strip()

    session = authenticate_cvat(username, password)
    if not session:
        return

    # Step 1 – Project selection or creation
    has_project = input("Do you already have a CVAT project? (y/n): ").strip().lower()
    if has_project == "y":
        project_id = input("Enter the project ID: ").strip()
        project_id = int(project_id)
    else:
        project_name = input("Enter a name for the new project: ").strip()
        project_id = get_or_create_project(session, project_name)
        if not project_id:
            return

    # Step 2 – Create task
    task_name = input("Enter a name for the new task: ").strip()
    task_id = create_task(session, project_id, task_name)
    if not task_id:
        return

    # Step 3 – Upload image folder
    image_dir = input("Enter full path to image directory: ").strip()
    if not upload_images(session, task_id, image_dir):
        return

    # Step 4 – Upload COCO annotations
    annotation_path = input("Enter full path to COCO annotation file (JSON): ").strip()
    if not upload_annotations(session, task_id, annotation_path):
        return

    print(f"View your CVAT task here: {CVAT_URL}/tasks/{task_id}")
