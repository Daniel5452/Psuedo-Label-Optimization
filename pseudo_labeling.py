import random
import json
import sqlite3
import os
import glob
import time
from pathlib import Path
from sys import version

import requests
from onedl.datasets import Dataset
from onedl.zoo.eval import EvaluationConfig, REPORT_TEMPLATE, Device
from onedl.core import LabelMap
from onedl.datasets.columns import ObjectIDColumn
from onedl.zoo.instance_segmentation.mmdetection import MaskRCNNConfig
from onedl.zoo.object_detection.mmdetection import FasterRCNNConfig

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


class DatabaseManager:
    """Handles all database operations and metadata logging with continuous updates."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn, self.cursor = self._connect_db()
        self._initialize_metadata_db()

    def _connect_db(self):
        """Connect to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        return conn, cursor

    def _initialize_metadata_db(self):
        """Initialize the metadata database table with status tracking."""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS iteration_metadata (
            flow_id TEXT,
            iteration INTEGER,
            status TEXT DEFAULT 'INITIALIZED',
            num_gt_images INTEGER,
            num_gt_images_added INTEGER,
            num_pseudo_images INTEGER,
            num_pseudo_images_added INTEGER,
            total_train_size INTEGER,
            main_dataset TEXT,
            validation_set TEXT,
            train_dataset TEXT,
            pseudo_input_dataset_name TEXT,
            pseudo_output_dataset_name TEXT,
            inference_model_uid TEXT,
            model_uid TEXT,
            evaluation_uid TEXT,
            evaluation_info TEXT,
            manual_correction BOOLEAN,
            cvat_project_id INTEGER,
            train_cfg TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_timestamp TEXT,
            PRIMARY KEY (flow_id, iteration)
        )
        ''')
        self.conn.commit()

    def initialize_iteration(self, **kwargs):
        """Initialize a new iteration with basic metadata."""
        train_cfg_str = str(kwargs.get('train_cfg')) if kwargs.get('train_cfg') is not None else None

        self.cursor.execute('''
            INSERT OR REPLACE INTO iteration_metadata (
                flow_id, iteration, status,
                num_gt_images, num_gt_images_added,
                num_pseudo_images, num_pseudo_images_added,
                total_train_size,
                main_dataset, validation_set, train_dataset,
                inference_model_uid, manual_correction, cvat_project_id, train_cfg
            ) VALUES (?, ?, 'INITIALIZED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            kwargs['flow_id'], kwargs['iteration'],
            kwargs['num_gt_images'], kwargs['num_gt_images_added'],
            kwargs['num_pseudo_images'], kwargs['num_pseudo_images_added'],
            kwargs['total_train_size'],
            kwargs['main_dataset'], kwargs['validation_dataset'], kwargs['train_dataset'],
            kwargs['inference_model_uid'], kwargs['manual_correction'],
            kwargs['cvat_project_id'], train_cfg_str
        ))
        self.conn.commit()

    def update_iteration_field(self, flow_id, iteration, **updates):
        """Update specific fields for an iteration."""
        if not updates:
            return

        set_clauses = []
        values = []

        for field, value in updates.items():
            set_clauses.append(f"{field} = ?")
            values.append(value)

        values.extend([flow_id, iteration])

        query = f"""
            UPDATE iteration_metadata 
            SET {', '.join(set_clauses)}
            WHERE flow_id = ? AND iteration = ?
        """

        self.cursor.execute(query, values)
        self.conn.commit()

    def update_status(self, flow_id, iteration, status):
        """Update the status of an iteration."""
        self.update_iteration_field(flow_id, iteration, status=status)

    def complete_iteration(self, flow_id, iteration):
        """Mark an iteration as completed with timestamp."""
        self.update_iteration_field(
            flow_id, iteration,
            status='COMPLETED',
            completed_timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

    def get_iteration_status(self, flow_id, iteration):
        """Get the current status of an iteration."""
        self.cursor.execute(
            'SELECT status FROM iteration_metadata WHERE flow_id = ? AND iteration = ?',
            (flow_id, iteration)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None

    def get_last_completed_iteration(self, flow_id):
        """Get the last completed iteration number for a flow."""
        self.cursor.execute(
            'SELECT MAX(iteration) FROM iteration_metadata WHERE flow_id = ? AND status = "COMPLETED"',
            (flow_id,)
        )
        result = self.cursor.fetchone()
        return result[0] if result and result[0] is not None else None

    def get_last_iteration(self, flow_id):
        """Get the last iteration number for a flow (completed or not)."""
        self.cursor.execute('SELECT MAX(iteration) FROM iteration_metadata WHERE flow_id = ?', (flow_id,))
        result = self.cursor.fetchone()
        return result[0] if result and result[0] is not None else None

    def get_previous_model_data(self, flow_id, iteration):
        """Get model UID and image counts from previous iteration."""
        self.cursor.execute('''
            SELECT model_uid, num_gt_images, num_pseudo_images
            FROM iteration_metadata
            WHERE flow_id = ? AND iteration = ?
        ''', (flow_id, iteration))
        return self.cursor.fetchone()

    def flow_exists(self, flow_id):
        """Check if a flow already exists in the database."""
        self.cursor.execute('SELECT COUNT(*) FROM iteration_metadata WHERE flow_id = ?', (flow_id,))
        result = self.cursor.fetchone()
        return result[0] > 0

    def log_iteration_0(self, **kwargs):
        """Log iteration 0 to the database (legacy method for compatibility)."""
        train_cfg_str = str(kwargs.get('train_cfg')) if kwargs.get('train_cfg') is not None else None

        self.cursor.execute('''
            INSERT OR REPLACE INTO iteration_metadata (
                flow_id, iteration, status,
                num_gt_images, num_gt_images_added,
                num_pseudo_images, num_pseudo_images_added,
                total_train_size,
                main_dataset, validation_set, train_dataset,
                pseudo_input_dataset_name, pseudo_output_dataset_name,
                inference_model_uid, model_uid, evaluation_uid, evaluation_info,
                manual_correction, cvat_project_id, train_cfg,
                completed_timestamp
            ) VALUES (?, ?, 'COMPLETED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            kwargs['flow_id'], kwargs['iteration'],
            kwargs['num_gt_images'], kwargs['num_gt_images_added'],
            kwargs['num_pseudo_images'], kwargs['num_pseudo_images_added'],
            kwargs['total_train_size'],
            kwargs['main_dataset'], kwargs['validation_dataset'], kwargs['train_dataset'],
            kwargs['pseudo_input_dataset_name'], kwargs['pseudo_output_dataset_name'],
            kwargs['inference_model_uid'], kwargs['model_uid'], kwargs['evaluation_uid'], kwargs['evaluation_info'],
            kwargs['manual_correction'], kwargs['cvat_project_id'], train_cfg_str,
            time.strftime('%Y-%m-%d %H:%M:%S')
        ))
        self.conn.commit()


class CVATManager:
    """Handles CVAT API operations and annotation management."""

    def __init__(self, cvat_url="https://cvat2.vbti.nl"):
        self.cvat_url = cvat_url
        self.cvat_api_url = f"{cvat_url}/api"
        self.session = None

    def authenticate(self, username, password):
        """Authenticate with CVAT server."""
        self.session = requests.Session()
        resp = self.session.post(f"{self.cvat_api_url}/auth/login",
                                 json={"username": username, "password": password})
        if resp.status_code == 200:
            token = resp.json()["key"]
            self.session.headers.update({"Authorization": f"Token {token}"})
            print("CVAT authentication successful.")
            return True
        else:
            raise RuntimeError("CVAT authentication failed. Check credentials.")

    def get_or_create_project(self, project_name, dataset_labels):
        """Get existing project or create new one with labels from dataset."""
        # First, try to find existing project
        resp = self.session.get(f"{self.cvat_api_url}/projects")
        if resp.status_code == 200:
            projects = resp.json()["results"]
            for project in projects:
                if project["name"] == project_name:
                    print(f"Using existing CVAT project: {project_name} (ID: {project['id']})")
                    return project["id"]

        # Create new project if not found - with dynamic labels from dataset
        labels = [{"name": label, "attributes": []} for label in dataset_labels]

        payload = {
            "name": project_name,
            "labels": labels
        }
        resp = self.session.post(f"{self.cvat_api_url}/projects", json=payload)
        if resp.status_code == 201:
            pid = resp.json()["id"]
            print(f"Created new CVAT project: {project_name} (ID: {pid})")
            print(f"Project created with labels: {', '.join(dataset_labels)}")
            return pid
        else:
            raise RuntimeError(f"CVAT project creation failed: {resp.text}")

    def create_task(self, project_id, task_name):
        """Create a new CVAT task."""
        payload = {
            "name": task_name,
            "project_id": project_id,
            "mode": "annotation",
            "overlap": 0,
            "segment_size": 0,
            "dimension": "2d"
        }
        resp = self.session.post(f"{self.cvat_api_url}/tasks", json=payload)
        if resp.status_code == 201:
            tid = resp.json()["id"]
            print(f"CVAT task created: {task_name} (ID: {tid})")
            return tid
        else:
            raise RuntimeError(f"CVAT task creation failed: {resp.text}")

    def upload_images(self, task_id, image_dir):
        """Upload images to CVAT task and wait for processing to complete."""
        url = f"{self.cvat_api_url}/tasks/{task_id}/data"
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.[jp][pn]g")) +
                             glob.glob(os.path.join(image_dir, "*.jpeg")))
        if not image_files:
            raise FileNotFoundError(f"No images found in folder: {image_dir}")

        print(f"Uploading {len(image_files)} images to CVAT...")
        files = {
            f"client_files[{i}]": (os.path.basename(path), open(path, "rb"), "image/jpeg")
            for i, path in enumerate(image_files)
        }

        payload = {"image_quality": 70, "sorting_method": "lexicographical"}
        resp = self.session.post(url, files=files, data=payload)

        # Close file handles
        for f in files.values():
            f[1].close()

        if resp.status_code != 202:
            raise RuntimeError(f"CVAT image upload failed: {resp.text}")

        rq_id = resp.json().get("rq_id")
        print(f"Images upload started. Request ID: {rq_id}")

        # Wait for image upload to complete
        self._wait_for_data_upload(task_id, rq_id)
        print("Images uploaded and processed successfully.")

    def _wait_for_data_upload(self, task_id, rq_id):
        """Wait for data upload to complete by checking task status."""
        print("Waiting for image processing to complete...")
        max_attempts = 60  # 5 minutes max wait (5 seconds * 60)
        attempt = 0

        while attempt < max_attempts:
            # Check the task status instead of request status
            resp = self.session.get(f"{self.cvat_api_url}/tasks/{task_id}")
            if resp.status_code != 200:
                print(f"Failed to check task status: {resp.text}")
                break

            task_data = resp.json()
            # Check if the task has data (images have been processed)
            if task_data.get("size", 0) > 0:
                print(f"Image processing complete. Task contains {task_data['size']} images.")
                return

            # Also check the original request status
            req_resp = self.session.get(f"{self.cvat_api_url}/requests/{rq_id}")
            if req_resp.status_code == 200:
                req_state = req_resp.json().get("state")
                if req_state == "failed":
                    raise RuntimeError("Image upload failed.")
                elif req_state == "finished":
                    # Double-check that images are actually loaded
                    time.sleep(2)  # Give CVAT a moment to update
                    continue

            print(f"Still processing... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(5)
            attempt += 1

        if attempt >= max_attempts:
            raise RuntimeError("Timeout waiting for image upload. Check CVAT manually.")

    def upload_annotations(self, task_id, annotation_file):
        """Upload COCO annotations to CVAT task."""
        format_name = "COCO 1.0"
        url = f"{self.cvat_api_url}/tasks/{task_id}/annotations?format={format_name}&location=local"

        with open(annotation_file, "rb") as f:
            files = {"annotation_file": (os.path.basename(annotation_file), f)}
            print(f"Uploading annotations to CVAT task {task_id}...")
            resp = self.session.post(url, files=files)

            if resp.status_code == 202:
                rq_id = resp.json().get("rq_id")
                print("CVAT annotation upload started. Request ID:", rq_id)
                self._check_annotation_status(rq_id)
            else:
                print(f"CVAT annotation upload failed: {resp.status_code} - {resp.text}")
                print("Task created successfully, but you'll need to manually upload annotations in CVAT.")

    def _check_annotation_status(self, rq_id):
        """Check CVAT annotation upload status."""
        url = f"{self.cvat_api_url}/requests/{rq_id}"
        print("Waiting for CVAT annotation import to complete...")
        max_attempts = 24  # 2 minutes max wait
        attempt = 0

        while attempt < max_attempts:
            resp = self.session.get(url)
            if resp.status_code != 200:
                print("Failed to check CVAT status:", resp.text)
                break

            status_data = resp.json()
            state = status_data.get("state")

            if state == "finished":
                print("CVAT annotation import completed successfully.")
                break
            elif state == "failed":
                error_msg = status_data.get("message", "Unknown error")
                print(f"CVAT annotation import failed: {error_msg}")
                raise RuntimeError(f"CVAT annotation import failed: {error_msg}")
            else:
                print(f"Still processing... State: {state}")
                time.sleep(5)
                attempt += 1

        if attempt >= max_attempts:
            print("Timeout waiting for annotation import. Check CVAT manually.")

    def get_task_by_name(self, project_id, task_name):
        """Find task ID by name within a project."""
        resp = self.session.get(f"{self.cvat_api_url}/tasks")
        if resp.status_code == 200:
            tasks = resp.json()["results"]
            for task in tasks:
                if task["project_id"] == project_id and task["name"] == task_name:
                    return task["id"]
        return None

    def export_annotations(self, task_id, export_path, format_name="COCO 1.0"):
        """Export annotations from CVAT task using the new API."""
        # Step 1: Initiate export
        url = f"{self.cvat_api_url}/tasks/{task_id}/dataset/export"
        params = {"save_images": "false", "format": format_name}

        print(f"Initiating export of annotations from CVAT task {task_id}...")
        resp = self.session.post(url, params=params)

        if resp.status_code != 202:
            raise RuntimeError(f"CVAT export initiation failed: {resp.status_code} - {resp.text}")

        rq_id = resp.json().get("rq_id")
        print(f"Export initiated. Request ID: {rq_id}")

        # Step 2: Wait for export to complete and get download URL
        download_url = self._wait_for_export_completion(rq_id)

        # Step 3: Download the exported file
        return self._download_exported_file(download_url, export_path)

    def _wait_for_export_completion(self, rq_id):
        """Wait for export to complete and return download URL."""
        url = f"{self.cvat_api_url}/requests/{rq_id}"
        print("Waiting for export to complete...")
        max_attempts = 600
        attempt = 0

        while attempt < max_attempts:
            resp = self.session.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to check export status: {resp.text}")

            status_data = resp.json()
            state = status_data.get("state")

            if state == "finished":
                result_url = status_data.get("result_url")
                if result_url:
                    print("Export completed successfully.")
                    return result_url
                else:
                    raise RuntimeError("Export finished but no result URL provided")
            elif state == "failed":
                error_msg = status_data.get("message", "Unknown error")
                raise RuntimeError(f"CVAT export failed: {error_msg}")
            else:
                print(f"Export in progress... State: {state} (attempt {attempt + 1}/{max_attempts})")
                time.sleep(5)
                attempt += 1

        raise RuntimeError("Timeout waiting for export to complete")

    def _download_exported_file(self, download_url, export_path):
        """Download the exported file and extract annotations."""
        print("Downloading exported annotations...")

        # Download the file
        resp = self.session.get(download_url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download export: {resp.status_code} - {resp.text}")

        # Save as temporary zip file
        import tempfile
        import zipfile

        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            temp_file.write(resp.content)
            temp_zip_path = temp_file.name

        try:
            # Extract the zip file
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                # Find the annotations file in the zip
                annotation_files = [f for f in zip_ref.namelist() if f.endswith('instances_default.json')]

                if not annotation_files:
                    raise RuntimeError("No COCO annotation file found in export")

                annotation_file = annotation_files[0]

                # Define the target path
                export_file = export_path / "annotations" / "instances_default.json"
                export_file.parent.mkdir(parents=True, exist_ok=True)

                # Delete existing file if it exists (no backup needed)
                if export_file.exists():
                    os.remove(export_file)
                    print(f"Deleted existing annotation file: {export_file}")

                # Extract and save the annotation file
                with zip_ref.open(annotation_file) as source, open(export_file, 'wb') as target:
                    target.write(source.read())

                print(f"Annotations exported and replaced successfully: {export_file}")
                return str(export_file)

        finally:
            # Clean up temporary file
            os.unlink(temp_zip_path)


class PseudoLabelingPipeline:
    """
    Main pseudo-labeling pipeline for object detection and instance segmentation.
    Handles complete workflow from data preparation to model training and evaluation with continuous logging.
    """

    def __init__(self, project_name, main_dataset_name, initial_annotated_dataset_name,
                 validation_dataset, sample_size_per_iter, current_flow, min_confidence=0.8,
                 local_path=None, cvat_project_id=None, db_path="pseudo_labeling_metadata.db"):
        """
        Initialize the pseudo-labeling pipeline.

        Args:
            project_name (str): OneDL project name
            main_dataset_name (str): Full unlabeled dataset name
            initial_annotated_dataset_name (str): Initial GT dataset name (shared across all flows)
            validation_dataset (str): Validation dataset name
            sample_size_per_iter (int): Number of samples per iteration
            current_flow (int): Flow number to initialize (e.g., 0, 1, 2)
            min_confidence (float): Minimum confidence threshold for pseudo-labels
            local_path (str): Local path for data storage
            cvat_project_id (int): CVAT project ID
            db_path (str): Database path for metadata
        """
        # Core configuration
        self.project_name = project_name
        self.main_dataset_name = main_dataset_name
        self.initial_annotated_dataset_name = initial_annotated_dataset_name
        self.validation_dataset = validation_dataset
        self.sample_size_per_iter = sample_size_per_iter
        self.min_confidence = min_confidence
        self.local_path = local_path
        self.cvat_project_id = cvat_project_id

        # Initialize helper classes
        self.db = DatabaseManager(db_path)
        self.cvat = CVATManager()

        from onedl.client import connect_to_project
        self.client = connect_to_project(project_name)
        self.full_dataset = self.client.datasets.load(main_dataset_name)

        # Flow setup (merged from setup_initial_flow)
        self.current_flow = current_flow
        self.flow_id = f'f{current_flow}'
        self.train_dataset_name = f"train-{self.flow_id}"
        self.pseudo_input_dataset_name = None
        self.predicted_dataset_name = None
        self.n_initial_samples = None

        # Determine current iteration based on database state
        last_completed = self.db.get_last_completed_iteration(self.flow_id)
        if last_completed is None:
            self.current_iteration = 0  # New flow, start at 0
        else:
            # Check if there's an incomplete iteration
            last_any = self.db.get_last_iteration(self.flow_id)
            current_status = self.db.get_iteration_status(self.flow_id, last_any) if last_any is not None else None

            if current_status == 'COMPLETED':
                self.current_iteration = last_any + 1  # Start next iteration
            else:
                self.current_iteration = last_any  # Resume incomplete iteration
                print(f"⚠️  Resuming incomplete iteration {self.current_iteration} (status: {current_status})")

        # models and eval
        self.train_cfg = None
        self.model_uid = None
        self.evaluation_uid = None
        self.evaluation_info_str = None
        self.inference_model_uid = None

        # Iteration tracking
        self.manual_corrections_global = None
        self.num_gt_images_after_iter = None
        self.num_gt_images_added = None
        self.num_pseudo_images_after_iter = None
        self.num_pseudo_images_added = None

        random.seed(42)

        print("=" * 60)
        print("GLOBAL INITIALIZATIONS INITIALIZED")
        print("=" * 60)
        print(f"Project: {project_name}")
        print(f"Main dataset: {main_dataset_name}")
        print(f"Initial annotated dataset: {initial_annotated_dataset_name}")
        print(f"Sample size per iteration: {sample_size_per_iter}")
        print(f"Selected flow: {self.flow_id}")

        # Load initial dataset to get sample count (always available)
        initial_dataset = self.client.datasets.load(self.initial_annotated_dataset_name, pull_blobs=True)
        self.n_initial_samples = len(initial_dataset)
        print(f"Initial annotated dataset contains: {self.n_initial_samples} samples")

        # Check if this flow already exists and set up accordingly
        if self.db.flow_exists(self.flow_id):
            print(f"Flow {self.flow_id} already exists in database - ready to resume")
            print(f"Last completed iteration: {last_completed}")

            # Check if training dataset exists
            try:
                existing_train_dataset = self.client.datasets.load(self.train_dataset_name, pull_policy="missing")
                print(f"Training dataset {self.train_dataset_name} exists with {len(existing_train_dataset)} samples")
            except Exception as e:
                print(f"Warning: Training dataset {self.train_dataset_name} not found: {e}")
                print("This may be normal if you're resuming from a different environment")

        else:
            # New flow - create the training dataset
            print(f"Creating new flow {self.flow_id}...")

            # Check if training dataset already exists (skip creation if it does)
            try:
                existing_train_dataset = self.client.datasets.load(self.train_dataset_name, pull_policy="missing")
                print(f"Training dataset {self.train_dataset_name} already exists - skipping creation")
            except:
                # Training dataset doesn't exist, create it
                self.client.datasets.save(self.train_dataset_name, dataset=initial_dataset, exist="overwrite")
                self.client.datasets.push(self.train_dataset_name, push_policy='version')
                print(f"Created training dataset: {self.train_dataset_name}")

            print(f"Flow {self.flow_id} initialized with {self.n_initial_samples} initial samples")

        print(f"Ready for iteration {self.current_iteration}")
        print("=" * 60)

    def _get_dataset_labels(self):
        """Extract labels from the initial annotated dataset."""
        initial_dataset = self.client.datasets.load(self.initial_annotated_dataset_name)

        if initial_dataset.targets.has_frozen_label_map():
            label_map = initial_dataset.targets.get_frozen_label_map()
            labels = list(label_map.values())
        elif initial_dataset.targets.has_frozen_labels():
            labels = initial_dataset.targets.get_frozen_labels()
        else:
            label_map = initial_dataset.targets.generate_label_map()
            labels = list(label_map.values())

        print(f"Extracted labels from dataset: {labels}")
        return labels

    def update_predicted_dataset_from_cvat(self, cvat_export_path):
        """Update the predicted dataset with manually corrected annotations from CVAT."""

        # Load the original predicted dataset
        predicted_dataset = self.client.datasets.load(self.predicted_dataset_name)

        # Import corrected annotations from CVAT export
        corrected_dataset = Dataset.import_coco(
            path=cvat_export_path,
            json_filename="annotations/instances_default.json"
        )

        # Update the targets with corrected annotations
        predicted_dataset.targets = corrected_dataset.targets

        # Save the updated dataset
        updated_dataset_name = f"{self.predicted_dataset_name}-corrected"
        self.client.datasets.save(updated_dataset_name, predicted_dataset, exist="overwrite")
        self.client.datasets.push(updated_dataset_name, push_policy='version')

        # Update the pipeline to use the corrected dataset
        self.predicted_dataset_name = updated_dataset_name

        print(f"Updated predicted dataset with CVAT corrections: {updated_dataset_name}")
        return updated_dataset_name

    # ========== TRAINING CONFIGURATION ==========

    def setup_training_config(self):
        """Interactive training configuration setup for Jupyter notebooks."""
        print("=== TRAINING CONFIGURATION SETUP ===")

        if not JUPYTER_AVAILABLE:
            raise RuntimeError("Jupyter widgets not available. Please install ipywidgets.")

        config = {}

        # Model type selection
        model_dropdown = widgets.Dropdown(
            options=[
                ('FasterRCNNConfig (Object Detection)', 'FasterRCNNConfig'),
                ('MaskRCNNConfig (Instance Segmentation)', 'MaskRCNNConfig')
            ],
            description='Model Type:',
            style={'description_width': 'initial'}
        )

        backbone_dropdown = widgets.Dropdown(
            options=[],
            description='Backbone:',
            style={'description_width': 'initial'}
        )

        epochs_input = widgets.IntText(
            value=50,
            description='Epochs:',
            style={'description_width': 'initial'}
        )

        batch_size_input = widgets.IntText(
            value=6,
            description='Batch Size:',
            style={'description_width': 'initial'}
        )

        submit_button = widgets.Button(
            description='Confirm Configuration',
            button_style='success'
        )

        output = widgets.Output()

        def update_backbones(change):
            model_type = change['new']
            available_backbones = self._get_available_backbones(model_type)
            if available_backbones:
                backbone_dropdown.options = [(str(b), b) for b in available_backbones]
            else:
                backbone_dropdown.options = [('Default', None)]

        def on_submit(b):
            with output:
                clear_output()
                model_type = model_dropdown.value
                task_type = 'object_detection' if model_type == 'FasterRCNNConfig' else 'instance_segmentation'

                config.update({
                    'model_type': model_type,
                    'task_type': task_type,
                    'backbone': backbone_dropdown.value,
                    'epochs': epochs_input.value,
                    'batch_size': batch_size_input.value
                })

                print("=== CONFIGURATION COMPLETE ===")
                print(f"Model: {model_type}")
                print(f"Task: {task_type}")
                print(f"Backbone: {backbone_dropdown.value}")
                print(f"Epochs: {epochs_input.value}")
                print(f"Batch size: {batch_size_input.value}")
                print("Configuration saved!")

        model_dropdown.observe(update_backbones, names='value')
        submit_button.on_click(on_submit)
        update_backbones({'new': model_dropdown.value})

        display(widgets.VBox([
            model_dropdown,
            backbone_dropdown,
            epochs_input,
            batch_size_input,
            submit_button,
            output
        ]))

        self.train_cfg = config
        return config

    def _get_available_backbones(self, model_type):
        """Get available backbones for the specified model type."""
        try:
            if model_type == "FasterRCNNConfig":
                from onedl.zoo.object_detection.mmdetection import FasterRCNNBackbone
                return [b for b in FasterRCNNBackbone]
            elif model_type == "MaskRCNNConfig":
                from onedl.zoo.instance_segmentation.mmdetection import MaskRCNNBackbone
                return [b for b in MaskRCNNBackbone]
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            print(f"Could not load backbones for {model_type}: {e}")
            return []

    # ========== FLOW SETUP METHODS ==========

    def setup_next_iteration(self, manual_corrections, current_flow=None):
        """Set up the next iteration for the current flow with status validation."""

        # If current_flow is provided, set up flow variables (for kernel restart scenarios)
        if current_flow is not None:
            self.current_flow = current_flow
            self.flow_id = f'f{current_flow}'
            self.train_dataset_name = f"train-{self.flow_id}"

        # Ensure flow_id is set
        if self.flow_id is None:
            raise RuntimeError(
                "Flow ID not set. Please provide current_flow parameter or run setup_initial_flow first.")

        # Check if current iteration is complete before proceeding
        if self.current_iteration > 0:
            current_status = self.db.get_iteration_status(self.flow_id, self.current_iteration)
            if current_status == 'COMPLETED':
                # Move to next iteration
                self.current_iteration += 1
                print(f"Previous iteration completed. Moving to iteration {self.current_iteration}")
            elif current_status is None:
                # This shouldn't happen, but handle gracefully
                print(f"No status found for iteration {self.current_iteration}. Proceeding...")
            else:
                print(f"Current iteration {self.current_iteration} status: {current_status}")
                user_input = input("Do you want to restart this iteration? (y/n): ").lower().strip()
                if user_input == 'y':
                    print(f"Restarting iteration {self.current_iteration}")
                else:
                    print("Continuing with current iteration...")

        self.manual_corrections_global = manual_corrections

        # Get the last completed iteration for the specified flow
        last_completed = self.db.get_last_completed_iteration(self.flow_id)
        if last_completed is None and self.current_iteration > 0:
            raise RuntimeError(f"No completed iteration history found for flow: {self.flow_id}")

        # For iteration 0, use initial dataset; for others, get previous model data
        if self.current_iteration == 0:
            previous_gt_total = 0
            previous_pseudo_total = 0
            self.inference_model_uid = ""
        else:
            # Get previous model UID and image counts
            result = self.db.get_previous_model_data(self.flow_id, self.current_iteration - 1)
            if result is None:
                raise RuntimeError(
                    f"No model metadata found for {self.flow_id} @ iteration {self.current_iteration - 1}")
            self.inference_model_uid, previous_gt_total, previous_pseudo_total = result

        # Define image additions based on correction mode
        if manual_corrections:
            self.num_gt_images_added = self.sample_size_per_iter
            self.num_pseudo_images_added = 0
        else:
            self.num_gt_images_added = 0
            self.num_pseudo_images_added = self.sample_size_per_iter

        self.num_gt_images_after_iter = previous_gt_total + self.num_gt_images_added
        self.num_pseudo_images_after_iter = previous_pseudo_total + self.num_pseudo_images_added
        total_train_size = self.num_gt_images_after_iter + self.num_pseudo_images_after_iter

        # Define dataset names
        self.pseudo_input_dataset_name = f"pseudo-iter{self.current_iteration}-{self.flow_id}"

        # Initialize the iteration in the database
        self.db.initialize_iteration(
            flow_id=self.flow_id,
            iteration=self.current_iteration,
            num_gt_images=self.num_gt_images_after_iter,
            num_gt_images_added=self.num_gt_images_added,
            num_pseudo_images=self.num_pseudo_images_after_iter,
            num_pseudo_images_added=self.num_pseudo_images_added,
            total_train_size=total_train_size,
            main_dataset=self.main_dataset_name,
            validation_dataset=self.validation_dataset,
            train_dataset=self.train_dataset_name,
            inference_model_uid=self.inference_model_uid,
            manual_correction=manual_corrections,
            cvat_project_id=self.cvat_project_id,
            train_cfg=self.train_cfg
        )

        # Enhanced output formatting
        print("-" * 40)
        print("ITERATION INITIALIZED")
        print("=" * 40)
        print(f"Flow ID: {self.flow_id}")
        print(f"Current iteration: {self.current_iteration}")
        print(f"Manual corrections: {manual_corrections}")
        print(f"Sample size this iteration: {self.sample_size_per_iter}")
        print(f"GT added this iteration: {self.num_gt_images_added}")
        print(f"Pseudo added this iteration: {self.num_pseudo_images_added}")
        print(f"Total GT images after this step: {self.num_gt_images_after_iter}")
        print(f"Total pseudo-labeled images after this step: {self.num_pseudo_images_after_iter}")
        print(f"Total expected training set size: {total_train_size}")
        print(f"Train dataset name: {self.train_dataset_name}")
        print(f"Pseudo input dataset name: {self.pseudo_input_dataset_name}")
        print(f"Using initial annotations: {self.initial_annotated_dataset_name}")
        print(f"Inference model UID: {self.inference_model_uid}")
        print("-" * 40)

    def set_inference_model_uid(self, model_uid):
        """Set the inference model UID manually."""
        self.inference_model_uid = model_uid
        print(f"Inference model UID set to: {self.inference_model_uid}")

        # Update database
        self.db.update_iteration_field(
            self.flow_id, self.current_iteration,
            inference_model_uid=self.inference_model_uid
        )

    # ========== PIPELINE METHODS WITH CONTINUOUS LOGGING ==========

    def sample_unseen_inputs(self):
        """
        Enhanced sampling function with continuous logging.
        """
        print(f"\n=== ENHANCED SAMPLING FOR ITERATION {self.current_iteration} ===")

        # Update status
        self.db.update_status(self.flow_id, self.current_iteration, 'SAMPLING')

        # Step 1: Load initial annotations (shared across all flows)
        try:
            initial_dataset = self.client.datasets.load(self.initial_annotated_dataset_name, pull_policy="missing")
            print(f"Loaded initial annotations: {len(initial_dataset)} images")
        except Exception as e:
            print(f"Could not load initial annotations {self.initial_annotated_dataset_name}: {e}")
            raise

        # Step 2: Sample new unseen inputs for current iteration
        full_hashes = list(self.full_dataset.inputs.hash_iterator())
        initial_hashes = set(initial_dataset.inputs.hash_iterator())
        remaining_hashes = list(set(full_hashes) - initial_hashes)

        if len(remaining_hashes) < self.sample_size_per_iter:
            raise ValueError(
                f"Only {len(remaining_hashes)} unseen images left, but {self.sample_size_per_iter} requested.")

        # Sample new images for this iteration
        sampled_hashes = random.sample(remaining_hashes, self.sample_size_per_iter)
        index_map = {h: i for i, h in enumerate(full_hashes)}
        sampled_indices = [index_map[h] for h in sampled_hashes]
        new_sampled_data = self.full_dataset[sampled_indices]

        # Create input-only dataset for new samples
        new_input_dataset = Dataset(inputs=new_sampled_data.inputs)
        print(f"Sampled {self.sample_size_per_iter} new images for iteration {self.current_iteration}")

        # Step 3: Find ALL past pseudo-labeled datasets from this flow
        past_pseudo_datasets = []

        self.db.cursor.execute('''
            SELECT iteration, pseudo_input_dataset_name, manual_correction
            FROM iteration_metadata
            WHERE flow_id = ? AND iteration < ? AND manual_correction = 0 AND status = 'COMPLETED'
            ORDER BY iteration
        ''', (self.flow_id, self.current_iteration))

        past_pseudo_results = self.db.cursor.fetchall()

        for past_iter, past_dataset_name, was_manual in past_pseudo_results:
            if past_dataset_name and past_dataset_name.strip():
                try:
                    print(f"  Found past pseudo dataset: {past_dataset_name} (iteration {past_iter})")
                    past_dataset = self.client.datasets.load(past_dataset_name, pull_policy="missing")
                    past_input_dataset = Dataset(inputs=past_dataset.inputs)
                    past_pseudo_datasets.append(past_input_dataset)
                    print(f"Loaded {len(past_input_dataset)} images from iteration {past_iter}")
                except Exception as e:
                    print(f"  ✗ Warning: Could not load {past_dataset_name}: {e}")

        # Step 4: Merge new samples with ALL past pseudo datasets
        combined_dataset = new_input_dataset

        if past_pseudo_datasets:
            print(f"Merging new samples with {len(past_pseudo_datasets)} past pseudo datasets")

            for i, past_dataset in enumerate(past_pseudo_datasets):
                combined_dataset = combined_dataset + past_dataset
                print(f"  Merged past dataset {i + 1}: total size now {len(combined_dataset)}")

            total_images = len(combined_dataset)
            past_images = total_images - self.sample_size_per_iter
            print(
                f"Final combined dataset: {self.sample_size_per_iter} new + {past_images} past = {total_images} total images")
        else:
            print(f"No past pseudo datasets found. Using only {len(combined_dataset)} new samples.")

        # Step 5: Save the combined dataset for inference
        self.client.datasets.save(self.pseudo_input_dataset_name, dataset=combined_dataset, exist="overwrite")
        self.client.datasets.push(self.pseudo_input_dataset_name, push_policy="version")

        print(
            f"Saved combined dataset '{self.pseudo_input_dataset_name}' with {len(combined_dataset)} images for inference")

        # Update database with sampling completion
        self.db.update_iteration_field(
            self.flow_id, self.current_iteration,
            pseudo_input_dataset_name=self.pseudo_input_dataset_name,
            status='SAMPLING_COMPLETE'
        )

    def run_inference(self):
        """Run inference on pseudo input dataset to generate predictions with logging."""
        if self.inference_model_uid is None:
            raise ValueError("Inference model UID not set. Use set_inference_model_uid() first.")

        print(f"Running inference with model: {self.inference_model_uid}")
        self.db.update_status(self.flow_id, self.current_iteration, 'INFERENCE')

        config = EvaluationConfig(
            model_name=self.inference_model_uid,
            dataset_name=self.pseudo_input_dataset_name,
            report_template=REPORT_TEMPLATE.EMPTY,
            batch_size=1
        )

        job = self.client.jobs.submit(config)
        self.predicted_dataset_name = self.client.jobs.get_dataset(job)

        print(f"Inference complete")
        print(f"Predictions saved as: {self.predicted_dataset_name}")

        # Update database
        self.db.update_iteration_field(
            self.flow_id, self.current_iteration,
            pseudo_output_dataset_name=self.predicted_dataset_name,
            status='INFERENCE_COMPLETE'
        )

    def set_predicted_dataset(self, dataset_name):
        """Manually set the predicted dataset name with logging."""
        self.predicted_dataset_name = dataset_name
        print(f"Predicted dataset set to: {self.predicted_dataset_name}")
        print("Ready to proceed to manual corrections or merge.")

        # Update database
        self.db.update_iteration_field(
            self.flow_id, self.current_iteration,
            pseudo_output_dataset_name=self.predicted_dataset_name,
            status='INFERENCE_COMPLETE'
        )

    def manually_correct_cvat(self):
        """Export predictions to CVAT for manual correction with logging."""
        if not self.manual_corrections_global:
            print("Manual corrections not enabled for this iteration.")
            return

        print("Starting CVAT export process...")
        self.db.update_status(self.flow_id, self.current_iteration, 'CVAT_EXPORT')

        # Get user input
        username = input("CVAT Username: ")
        password = input("CVAT Password: ")
        project_name = input(f"CVAT Project Name (default: {self.flow_id}-project): ") or f"{self.flow_id}-project"

        # Export and upload to CVAT
        try:
            self._export_to_cvat(username, password, project_name)
            print("\nCVAT export completed successfully!")
            print("Please complete your annotations in CVAT.")
            print("The next cell will handle importing the corrected annotations and merging.")

            # Update status to indicate CVAT work is needed
            self.db.update_status(self.flow_id, self.current_iteration, 'CVAT_PENDING')

        except Exception as e:
            print(f"CVAT export failed: {e}")
            self.db.update_status(self.flow_id, self.current_iteration, 'CVAT_FAILED')

    def _export_to_cvat(self, username, password, project_name):
        """Internal method to handle CVAT export with full JSON preprocessing."""
        # Set up paths
        export_path = Path(self.local_path) / f"cvat_export_iter_{self.current_iteration}"

        # Delete existing folder if it exists to avoid conflicts
        if export_path.exists():
            import shutil
            shutil.rmtree(export_path)
            print(f"Deleted existing export folder: {export_path}")

        export_path.mkdir(parents=True, exist_ok=True)

        # Export dataset in COCO format
        json_filename = "annotations/instances_default.json"
        annotations_dir = export_path / "annotations"
        annotations_dir.mkdir(exist_ok=True)

        print(f"Preparing export to: {export_path}")

        # Load and export dataset
        raw_ds = self.client.datasets.load(self.predicted_dataset_name)
        raw_ds.predictions = [
            insts.filter_by_confidence(self.min_confidence)
            for insts in raw_ds.predictions
        ]
        raw_ds.targets = raw_ds.predictions
        raw_ds.export_coco(path=export_path, json_filename=json_filename)

        print(f"Initial COCO export complete (confidence >= {self.min_confidence})")

        # ========== JSON PREPROCESSING FOR CVAT COMPATIBILITY ==========
        json_path = export_path / json_filename
        with open(json_path, "r") as f:
            coco_data = json.load(f)

        print(
            f"Before fixes - Categories: {len(coco_data.get('categories', []))}, Images: {len(coco_data.get('images', []))}, Annotations: {len(coco_data.get('annotations', []))}")

        # Fix 1: Clean image file names
        for image_entry in coco_data.get("images", []):
            image_entry["file_name"] = Path(image_entry["file_name"]).name

        # Fix 2: Remove problematic fields from annotations
        for annotation in coco_data.get("annotations", []):
            # Remove score field - this causes import errors in CVAT
            if "score" in annotation:
                del annotation["score"]

            # Ensure required fields are present
            if "iscrowd" not in annotation:
                annotation["iscrowd"] = 0

            # Ensure area is calculated if missing
            if "area" not in annotation and "bbox" in annotation:
                bbox = annotation["bbox"]
                annotation["area"] = bbox[2] * bbox[3]  # width * height

        # Fix 3: Check category ID starting point and shift if needed
        categories = coco_data.get("categories", [])
        if categories:
            min_category_id = min(cat["id"] for cat in categories)
            if min_category_id == 0:
                print("Shifting category IDs from 0-based to 1-based...")
                for category in categories:
                    category["id"] += 1
                for annotation in coco_data.get("annotations", []):
                    annotation["category_id"] += 1
                print("Category IDs shifted by +1 for CVAT compatibility.")
            else:
                print(f"Category IDs already start from {min_category_id}, no shift needed.")

        # Fix 4: Validate image IDs (CVAT might need 1-based image IDs too)
        images = coco_data.get("images", [])
        if images:
            min_image_id = min(img["id"] for img in images)
            if min_image_id == 0:
                print("Converting image IDs from 0-based to 1-based...")
                # Create mapping for image ID conversion
                image_id_mapping = {}
                for i, image in enumerate(images):
                    old_id = image["id"]
                    new_id = i + 1  # Start from 1
                    image["id"] = new_id
                    image_id_mapping[old_id] = new_id

                # Update annotation image_id references
                for annotation in coco_data.get("annotations", []):
                    old_image_id = annotation["image_id"]
                    if old_image_id in image_id_mapping:
                        annotation["image_id"] = image_id_mapping[old_image_id]
                    else:
                        print(f"Warning: Annotation references non-existent image_id {old_image_id}")

                print("Image IDs converted to 1-based indexing.")
            else:
                print(f"Image IDs already start from {min_image_id}, no conversion needed.")

        # Fix 5: Ensure annotation IDs are sequential starting from 1
        annotations = coco_data.get("annotations", [])
        for i, annotation in enumerate(annotations):
            annotation["id"] = i + 1

        # Fix 6: Ensure we have valid info field
        if "info" not in coco_data:
            coco_data["info"] = {
                "description": "CVAT Pseudo-labeling Export",
                "version": "1.0",
                "year": 2025,
                "contributor": "Pseudo-labeling Pipeline",
                "date_created": "2025"
            }

        # Fix 7: Ensure we have licenses field
        if "licenses" not in coco_data:
            coco_data["licenses"] = []

        print(
            f"After fixes - Categories: {len(categories)}, Images: {len(coco_data.get('images', []))}, Annotations: {len(coco_data.get('annotations', []))}")

        # Save the corrected COCO file
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=2)

        print(f"COCO file cleaned and saved to {json_path}")

        # ========== UPLOAD TO CVAT ==========
        # Get dataset labels for CVAT project creation
        dataset_labels = self._get_dataset_labels()

        try:
            self.cvat.authenticate(username, password)
            project_id = self.cvat.get_or_create_project(project_name, dataset_labels)
            task_name = f"{self.flow_id}-iter{self.current_iteration}-corrections"
            task_id = self.cvat.create_task(project_id, task_name)

            # Upload images
            image_dir = export_path / "images"
            if image_dir.exists():
                self.cvat.upload_images(task_id, str(image_dir))

            # Upload annotations
            self.cvat.upload_annotations(task_id, str(json_path))

            task_url = f"{self.cvat.cvat_url}/tasks/{task_id}"
            print(f"CVAT task URL: {task_url}")

        except Exception as e:
            print(f"Error during CVAT upload: {e}")

    def merge_pseudo_labels(self):
        """
        Merge pseudo-labeled predictions into the training dataset with logging.
        """
        print("Starting merge process...")
        self.db.update_status(self.flow_id, self.current_iteration, 'MERGING')

        if self.manual_corrections_global:
            print("Manual corrections mode - importing from CVAT and merging...")
            self.import_cvat_corrections_and_merge()
        else:
            print("Automated mode - merging pseudo-labels directly...")
            self._merge_automated_pseudo_labels()

        # Update status after successful merge
        self.db.update_status(self.flow_id, self.current_iteration, 'MERGE_COMPLETE')

    def import_cvat_corrections_and_merge(self):
        """Import corrected annotations from CVAT and merge with training dataset."""
        if not self.manual_corrections_global:
            print("Manual corrections not enabled - using automated pseudo-labels")
            self._merge_automated_pseudo_labels()
            return

        # Ask user if they have annotations ready or need to download from CVAT
        print("\nANNOTATION SOURCE OPTIONS:")
        print("1. Download corrected annotations from CVAT automatically")
        print("2. Use existing annotation file (manual path)")

        choice = input("\nChoose option (1 or 2): ").strip()

        if choice == "1":
            # Original CVAT download workflow
            self._download_from_cvat_and_merge()
        elif choice == "2":
            # Manual file path workflow
            self._use_manual_annotation_file_and_merge()
        else:
            print("Invalid choice. Please run again and select 1 or 2.")
            return

    def _download_from_cvat_and_merge(self):
        """Download annotations from CVAT and merge with training dataset."""
        # Get user input for CVAT import
        username = input("CVAT Username: ")
        password = input("CVAT Password: ")
        project_name = input(f"CVAT Project Name (default: {self.flow_id}-project): ") or f"{self.flow_id}-project"
        task_name = input(
            f"CVAT Task Name (default: {self.flow_id}-iter{self.current_iteration}-corrections): ") or f"{self.flow_id}-iter{self.current_iteration}-corrections"

        try:
            # Set up paths
            export_path = Path(self.local_path) / f"cvat_export_iter_{self.current_iteration}"

            # Authenticate with CVAT
            self.cvat.authenticate(username, password)

            # Get project and task IDs
            dataset_labels = self._get_dataset_labels()
            project_id = self.cvat.get_or_create_project(project_name, dataset_labels)
            task_id = self.cvat.get_task_by_name(project_id, task_name)

            if task_id is None:
                raise RuntimeError(f"Task '{task_name}' not found in project '{project_name}'")

            print(f"Found CVAT task: {task_name} (ID: {task_id})")

            # Export corrected annotations from CVAT - this will replace the existing file
            corrected_annotation_file = self.cvat.export_annotations(task_id, export_path)

            # Continue with merge process
            self._import_and_merge_annotations(export_path)

        except Exception as e:
            print(f"CVAT import failed: {e}")
            print("Please check CVAT task exists and try again.")

    def _use_manual_annotation_file_and_merge(self):
        """Use manually provided annotation file and merge with training dataset."""
        print("\nMANUAL ANNOTATION FILE")
        print("Please provide the path to your corrected COCO annotation file.")
        print("Example: /path/to/your/annotations/instances_default.json")

        annotation_file_path = input("\nPath to annotation file: ").strip()

        if not annotation_file_path:
            print("No path provided. Cancelling operation.")
            return

        annotation_file = Path(annotation_file_path)

        # Validate the file exists
        if not annotation_file.exists():
            print(f"File not found: {annotation_file}")
            return

        # Validate it's a JSON file
        if not annotation_file.name.endswith('.json'):
            print(f"File must be a JSON file. Got: {annotation_file.name}")
            return

        try:
            # Validate it's valid JSON by trying to load it
            with open(annotation_file, 'r') as f:
                json.load(f)
            print(f"Valid JSON file found: {annotation_file}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON file: {e}")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        # Set up the export path (where our original export was)
        export_path = Path(self.local_path) / f"cvat_export_iter_{self.current_iteration}"
        target_annotation_file = export_path / "annotations" / "instances_default.json"

        # Create the annotations directory if it doesn't exist
        target_annotation_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Copy the user's annotation file to replace our existing one
            import shutil
            shutil.copy2(annotation_file, target_annotation_file)
            print(f"Copied annotation file to: {target_annotation_file}")

            # Continue with merge process
            self._import_and_merge_annotations(export_path)

        except Exception as e:
            print(f"Error copying annotation file: {e}")
            return

    def _import_and_merge_annotations(self, export_path):
        """Common method to import annotations and merge with training dataset."""
        try:
            # Import corrected annotations and create new dataset version
            print("Importing corrected annotations...")
            corrected_dataset = Dataset.import_coco(
                path=str(export_path),
                image_subdir="images",
                coco_filename="annotations/instances_default.json"
            )

            # Load the original predicted dataset and update it with corrections
            predicted_dataset = self.client.datasets.load(self.predicted_dataset_name)
            predicted_dataset.targets = corrected_dataset.targets

            # Save the corrected dataset as a new version
            corrected_dataset_name = f"{self.predicted_dataset_name}-corrected"
            self.client.datasets.save(corrected_dataset_name, predicted_dataset, exist="overwrite")
            self.client.datasets.push(corrected_dataset_name, push_policy='version')

            # Update the pipeline to use the corrected dataset
            self.predicted_dataset_name = corrected_dataset_name
            print(f"Updated predicted dataset with CVAT corrections: {corrected_dataset_name}")

            # Update database with corrected dataset name
            self.db.update_iteration_field(
                self.flow_id, self.current_iteration,
                pseudo_output_dataset_name=self.predicted_dataset_name
            )

            # Now merge with training dataset
            self._merge_corrected_labels()

        except Exception as e:
            print(f"Failed to import and merge annotations: {e}")
            print("Please check the annotation file format and try again.")

    def _merge_automated_pseudo_labels(self):
        """Merge automated pseudo-labels (no manual corrections)."""
        training_dataset = self.client.datasets.load(self.train_dataset_name, pull_policy="missing")
        pseudo_dataset = self.client.datasets.load(self.predicted_dataset_name)

        # Filter by confidence and set predictions as targets
        pseudo_dataset.predictions = [
            insts.filter_by_confidence(self.min_confidence)
            for insts in pseudo_dataset.predictions
        ]
        pseudo_dataset.targets = pseudo_dataset.predictions

        # Ensure label map consistency and merge
        self._ensure_label_consistency_and_merge(training_dataset, pseudo_dataset)

        print(f"Merged automated pseudo-labels (confidence >= {self.min_confidence})")
        print(f"Updated dataset: {self.train_dataset_name}")

    def _merge_corrected_labels(self):
        """Merge manually corrected labels from CVAT."""
        training_dataset = self.client.datasets.load(self.train_dataset_name, pull_policy="missing")
        pseudo_dataset = self.client.datasets.load(self.predicted_dataset_name)

        # For manual corrections, targets are already set correctly, no confidence filtering needed
        # Ensure label map consistency and merge
        self._ensure_label_consistency_and_merge(training_dataset, pseudo_dataset)

        print(f"Merged manually corrected labels from CVAT")
        print(f"Updated dataset: {self.train_dataset_name}")

    def _ensure_label_consistency_and_merge(self, training_dataset, pseudo_dataset):
        """
        Enhanced merging that combines:
        1. Initial annotations (shared global dataset)
        2. New pseudo-labeled dataset (with updated predictions from ALL past pseudo data)

        This ensures we always start from clean initial annotations plus current pseudo-labels.
        """
        print(f"\n=== ENHANCED MERGING FOR ITERATION {self.current_iteration} ===")

        # Load initial annotations (shared across all flows)
        initial_dataset = self.client.datasets.load(self.initial_annotated_dataset_name, pull_policy="missing")
        print(f"✓ Loaded initial annotations: {len(initial_dataset)} images")
        print(f"✓ Loaded new pseudo-labeled dataset: {len(pseudo_dataset)} images")

        # Ensure label map consistency
        if initial_dataset.targets.has_frozen_label_map():
            label_map = initial_dataset.targets.get_frozen_label_map()
        elif initial_dataset.targets.has_frozen_labels():
            label_map = LabelMap.from_labels(initial_dataset.targets.get_frozen_labels())
        else:
            label_map = initial_dataset.targets.generate_label_map()

        pseudo_dataset.targets.freeze_label_map(label_map)

        # Convert to ObjectIDColumn and merge
        initial_dataset.targets = ObjectIDColumn(initial_dataset.targets)
        pseudo_dataset.targets = ObjectIDColumn(pseudo_dataset.targets)

        # Merge: Initial annotations + Current pseudo-labels (which include updated past data)
        merged = initial_dataset + pseudo_dataset
        self.client.datasets.save(self.train_dataset_name, merged, exist="versions", skip_validation=True)
        self.client.datasets.push(self.train_dataset_name, push_policy="version")

        print(
            f"Merged datasets into '{self.train_dataset_name}': {len(initial_dataset)} initial + {len(pseudo_dataset)} pseudo = {len(merged)} total")

    def train_model(self):
        """Train model on current dataset with logging."""
        if self.train_cfg is None:
            raise ValueError("Training configuration not set. Run setup_training_config() first.")

        print("Starting model training...")
        self.db.update_status(self.flow_id, self.current_iteration, 'TRAINING')

        model_type = self.train_cfg['model_type']

        # Use initial_annotated_dataset for first training, train_dataset_name for iterations
        if self.current_iteration == 0:
            dataset_to_use = self.initial_annotated_dataset_name
        else:
            dataset_to_use = self.train_dataset_name

        # Base configuration parameters
        base_params = {
            'train_dataset_name': dataset_to_use,
            'val_dataset_name': self.validation_dataset,
            'num_epochs': self.train_cfg['epochs'],
            'batch_size': self.train_cfg['batch_size']
        }

        # Add backbone if selected
        if self.train_cfg.get('backbone') is not None:
            base_params['backbone'] = self.train_cfg['backbone']

        # Create model configuration based on type
        try:
            if model_type == "FasterRCNNConfig":
                config = FasterRCNNConfig(**base_params)
            elif model_type == "MaskRCNNConfig":
                config = MaskRCNNConfig(**base_params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Disable tracking if available
            if hasattr(config, 'tracking'):
                config.tracking.enabled = False

        except Exception as e:
            print(f"Error creating {model_type} config: {e}")
            print("Falling back to basic configuration without backbone...")

            # Fallback to basic config without backbone
            basic_params = {
                'train_dataset_name': dataset_to_use,
                'val_dataset_name': self.validation_dataset,
                'num_epochs': self.train_cfg['epochs'],
                'batch_size': self.train_cfg['batch_size']
            }

            if model_type == "FasterRCNNConfig":
                config = FasterRCNNConfig(**basic_params)
            else:
                config = MaskRCNNConfig(**basic_params)

            if hasattr(config, 'tracking'):
                config.tracking.enabled = False

        # Submit training job
        print(f"Training {model_type} on dataset: {dataset_to_use}")
        print(f"Configuration: {self.train_cfg['epochs']} epochs, batch size {self.train_cfg['batch_size']}")
        if self.train_cfg.get('backbone'):
            print(f"Backbone: {self.train_cfg['backbone']}")

        self.model_uid = self.client.jobs.submit(config)
        print(f"Training job submitted")
        print(f"Model UID: {self.model_uid}")

        # Update database with model UID
        self.db.update_iteration_field(
            self.flow_id, self.current_iteration,
            model_uid=self.model_uid,
            status='TRAINING_COMPLETE'
        )

    def evaluate_model(self):
        """Evaluate the current model on validation dataset with logging."""
        print(f'Evaluating {self.model_uid} on {self.validation_dataset}')
        self.db.update_status(self.flow_id, self.current_iteration, 'EVALUATING')

        config = EvaluationConfig(
            model_name=self.model_uid,
            dataset_name=self.validation_dataset,
            device=Device.CPU
        )

        self.evaluation_uid = self.client.jobs.submit(config)
        job_state = self.client.jobs.get_state(self.evaluation_uid)

        if job_state in ("DONE", "FAILED"):
            report_name = self.client.jobs.get_evaluation(self.evaluation_uid)
            report_info = self.client.evaluations.get_info(report_name)
            report_url = self.client.evaluations.get_uri(report_name)
            self.evaluation_info_str = json.dumps(report_info.get("metrics", {}))

            print("Evaluation complete")
            print(f"Report URL: {report_url}")
            print(f"Metrics: {self.evaluation_info_str}")

            # Update database with evaluation results
            self.db.update_iteration_field(
                self.flow_id, self.current_iteration,
                evaluation_uid=self.evaluation_uid,
                evaluation_info=self.evaluation_info_str,
                status='EVALUATION_COMPLETE'
            )
        else:
            self.evaluation_info_str = ""
            print("Evaluation incomplete or failed")
            self.db.update_status(self.flow_id, self.current_iteration, 'EVALUATION_FAILED')

    def complete_iteration(self):
        """
        Complete the current iteration and prepare for the next one.
        This should be called after all steps are finished.
        """
        # Verify that all required fields are filled
        required_fields = ['model_uid', 'evaluation_uid']

        # Check current iteration data
        self.db.cursor.execute(
            'SELECT model_uid, evaluation_uid FROM iteration_metadata WHERE flow_id = ? AND iteration = ?',
            (self.flow_id, self.current_iteration)
        )
        result = self.db.cursor.fetchone()

        if not result:
            raise RuntimeError(f"No data found for iteration {self.current_iteration}")

        model_uid, eval_uid = result

        # Check if essential fields are filled
        if not model_uid:
            raise RuntimeError("Cannot complete iteration: model_uid is missing. Please run train_model() first.")
        if not eval_uid:
            raise RuntimeError(
                "Cannot complete iteration: evaluation_uid is missing. Please run evaluate_model() first.")

        # Mark iteration as completed
        self.db.complete_iteration(self.flow_id, self.current_iteration)

        print("=" * 50)
        print("ITERATION COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Flow: {self.flow_id}")
        print(f"Iteration: {self.current_iteration}")
        print(f"Model UID: {model_uid}")
        print(f"Evaluation UID: {eval_uid}")
        print("=" * 50)
        print("Ready to start next iteration with setup_next_iteration()")
        print("=" * 50)

    # ========== LOGGING METHODS ==========

    def log_iteration_0(self):
        """Log iteration 0 to the database (legacy method for compatibility)."""
        self.db.log_iteration_0(
            flow_id=self.flow_id,
            iteration=0,
            num_gt_images=self.n_initial_samples,
            num_gt_images_added=self.n_initial_samples,
            num_pseudo_images=0,
            num_pseudo_images_added=0,
            total_train_size=self.n_initial_samples,
            train_dataset=self.train_dataset_name,
            pseudo_input_dataset_name="",
            pseudo_output_dataset_name="",
            inference_model_uid="",
            model_uid=self.model_uid,
            evaluation_uid=self.evaluation_uid,
            evaluation_info=self.evaluation_info_str,
            manual_correction=True,
            cvat_project_id=self.cvat_project_id,
            main_dataset=self.main_dataset_name,
            validation_dataset=self.validation_dataset,
            train_cfg=self.train_cfg
        )

        print(f"Iteration 0 logged for {self.flow_id}")

    def get_pipeline_status(self):
        """Display current pipeline status and state information."""
        print(f"\nPIPELINE STATUS REPORT")
        print(f"=" * 50)
        print(f"Flow ID: {self.flow_id}")
        print(f"Current Iteration: {self.current_iteration}")

        # Get current iteration status from database
        current_status = self.db.get_iteration_status(self.flow_id, self.current_iteration)
        if current_status:
            print(f"Current Status: {current_status}")

        print(f"Training Dataset: {self.train_dataset_name}")
        print(f"Current Model UID: {self.model_uid}")
        print(f"Training Configuration: {self.train_cfg}")
        print(f"Database Path: {self.db.db_path}")

        if hasattr(self, 'num_gt_images_after_iter') and self.num_gt_images_after_iter is not None:
            print(f"Ground Truth Images: {self.num_gt_images_after_iter}")
            print(f"Pseudo-labeled Images: {self.num_pseudo_images_after_iter}")
            total_images = self.num_gt_images_after_iter + self.num_pseudo_images_after_iter
            print(f"Total Training Images: {total_images}")

        print(f"Sample Size Per Iteration: {self.sample_size_per_iter}")
        print(f"Minimum Confidence Threshold: {self.min_confidence}")

        # Show recent iterations status
        print(f"\nRECENT ITERATIONS:")
        self.db.cursor.execute('''
            SELECT iteration, status, completed_timestamp 
            FROM iteration_metadata 
            WHERE flow_id = ? 
            ORDER BY iteration DESC 
            LIMIT 5
        ''', (self.flow_id,))

        for iteration, status, completed_time in self.db.cursor.fetchall():
            completion_info = f" (completed: {completed_time})" if completed_time else ""
            print(f"  Iteration {iteration}: {status}{completion_info}")

        print(f"=" * 50)