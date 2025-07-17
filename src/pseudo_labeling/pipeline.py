import random
import json
import os
import glob
import time
from pathlib import Path

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

from .database_pseudo import DatabaseManager
from .cvat_manager import CVATManager


class PseudoLabelingPipeline:
    """
    Main pseudo-labeling pipeline for object detection and instance segmentation.
    Handles complete workflow from data preparation to model training and evaluation with continuous logging.
    """

    def __init__(self, project_name, main_dataset_name, initial_annotated_dataset_name,
                 validation_dataset, sample_size_per_iter, current_flow, min_confidence=0.8,
                 local_path=None, db_path="pseudo_labeling_metadata.db"):
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
                print(f"Resuming incomplete iteration {self.current_iteration} (status: {current_status})")

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

        # Auto-recovery for kernel restarts
        if self.current_iteration > 0:
            print("\nAttempting auto-recovery...")
            self.recover_current_iteration_state()

        print("=" * 60)

    def cvat_connect(self, username, password, organization=None, project_id=None, project_name=None):
        """
        CVAT connection and configure project settings.

        Args:
            username (str): CVAT username
            password (str): CVAT password
            organization (str, optional): Organization name or slug to work within
            project_id (int, optional): Use existing project ID (leave None to create/find by name)
            project_name (str, optional): Project name to find/create (leave None to use default naming)

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            success = self.cvat.authenticate(username, password, organization)
            if success:
                self.cvat_authenticated = True

                # Store project configuration for later use
                self.cvat_project_config = {
                    'project_id': project_id,
                    'project_name': project_name
                }

                if organization:
                    print(f"✓ CVAT session established in organization: {organization}")
                else:
                    print("✓ CVAT session established in personal workspace")

                # Show project configuration
                if project_id:
                    print(f"✓ Will use existing project ID: {project_id}")
                elif project_name:
                    print(f"✓ Will find/create project: {project_name}")
                else:
                    default_name = f"{self.flow_id}-project"
                    print(f"✓ Will use default project naming: {default_name}")

            return success

        except Exception as e:
            print(f"✗ CVAT connection failed: {e}")
            self.cvat_authenticated = False
            return False

    def recover_current_iteration_state(self):
        """
        Recover current iteration state from database after kernel restart.
        Only recovers essential fields needed for resuming work.
        """
        print(f"Recovering state for {self.flow_id} iteration {self.current_iteration}...")

        self.db.cursor.execute('''
            SELECT model_uid, evaluation_uid, pseudo_output_dataset_name, 
                   manual_correction, inference_model_uid, pseudo_input_dataset_name
            FROM iteration_metadata 
            WHERE flow_id = ? AND iteration = ?
        ''', (self.flow_id, self.current_iteration))

        result = self.db.cursor.fetchone()

        if result:
            (model_uid, eval_uid, pseudo_output, manual_corr,
             inference_model_uid, pseudo_input) = result

            # Recover essential state
            if model_uid:
                self.model_uid = model_uid
                print(f"✓ Recovered model_uid: {model_uid}")

            if eval_uid:
                self.evaluation_uid = eval_uid
                print(f"✓ Recovered evaluation_uid: {eval_uid}")

            if pseudo_output:
                self.predicted_dataset_name = pseudo_output
                print(f"✓ Recovered predicted_dataset_name: {pseudo_output}")

            if manual_corr is not None:
                self.manual_corrections_global = bool(manual_corr)
                print(f"✓ Recovered manual_corrections_mode: {self.manual_corrections_global}")

            if inference_model_uid:
                self.inference_model_uid = inference_model_uid
                print(f"✓ Recovered inference_model_uid: {inference_model_uid}")

            # NEW: Recover pseudo_input_dataset_name
            if pseudo_input:
                self.pseudo_input_dataset_name = pseudo_input
                print(f"✓ Recovered pseudo_input_dataset_name: {pseudo_input}")

                # For manual corrections, also set the persistent pseudo dataset name
                if self.manual_corrections_global:
                    # In manual mode, pseudo_input is the temp dataset
                    print(f"✓ Manual corrections mode - using temp dataset: {pseudo_input}")
                else:
                    # In auto mode, pseudo_input is the persistent dataset
                    self.persistent_pseudo_dataset_name = pseudo_input
                    print(f"✓ Auto mode - set persistent pseudo dataset: {pseudo_input}")

            print("✓ Recovery complete - ready to resume")
            return True

        else:
            print(f"No database record found - this appears to be a new iteration")
            return False

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
        """Set up the next iteration with persistent dataset architecture."""

        # If current_flow is provided, set up flow variables (for kernel restart scenarios)
        if current_flow is not None:
            self.current_flow = current_flow
            self.flow_id = f'f{current_flow}'
            self.train_dataset_name = f"train-{self.flow_id}"

        # Ensure flow_id is set
        if self.flow_id is None:
            raise RuntimeError("Flow ID not set. Please provide current_flow parameter.")

        # FIXED: Check if current iteration is complete before proceeding
        current_status = self.db.get_iteration_status(self.flow_id, self.current_iteration)
        if current_status == 'COMPLETED':
            self.current_iteration += 1
            print(
                f"Previous iteration {self.current_iteration - 1} completed. Moving to iteration {self.current_iteration}")
        elif current_status is None:
            print(f"No status found for iteration {self.current_iteration}. Proceeding...")
        else:
            # FIXED: Just show status and exit - the existing auto-recovery handles everything else
            print(f"Current iteration {self.current_iteration} status: {current_status}")
            print("Complete the current iteration first, or re-initialize the pipeline to auto-recover state.")
            print("Use pipeline.get_pipeline_status() to see what step to do next.")
            return

        # [Rest of the original setup_next_iteration code remains exactly the same...]
        self.manual_corrections_global = manual_corrections

        # Get the last completed iteration for the specified flow
        last_completed = self.db.get_last_completed_iteration(self.flow_id)

        # Handle iteration progression properly
        if self.current_iteration == 0 and last_completed is None:
            # This is a brand new flow, no completed iterations yet
            previous_gt_total = 0
            previous_pseudo_total = 0
            self.inference_model_uid = ""
        elif self.current_iteration > 0:
            # For iterations 1+, get model data from the previous iteration
            result = self.db.get_previous_model_data(self.flow_id, self.current_iteration - 1)
            if result is None:
                raise RuntimeError(
                    f"No model metadata found for {self.flow_id} @ iteration {self.current_iteration - 1}")
            self.inference_model_uid, previous_gt_total, previous_pseudo_total = result
        else:
            # Iteration 0 exists and is completed, we're setting up iteration 1
            result = self.db.get_previous_model_data(self.flow_id, 0)
            if result is None:
                raise RuntimeError(f"No model metadata found for {self.flow_id} @ iteration 0")
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

        # Define dataset names - NEW PERSISTENT ARCHITECTURE
        self.persistent_pseudo_dataset_name = f"pseudo-{self.flow_id}"  # Persistent pseudo dataset
        self.manual_corrections_dataset_name = f"manual-corrections-{self.flow_id}"  # Persistent manual corrections

        if manual_corrections:
            # For manual corrections: temp folder -> manual corrections dataset
            self.pseudo_input_dataset_name = f"temp-cvat-iter{self.current_iteration}-{self.flow_id}"
            self.predicted_dataset_name = self.manual_corrections_dataset_name  # Will be updated/created
        else:
            # For auto pseudo-labeling: persistent pseudo -> inference output
            self.pseudo_input_dataset_name = self.persistent_pseudo_dataset_name
            self.predicted_dataset_name = None  # Will be set by inference

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
            train_cfg=self.train_cfg
        )

        # Enhanced output formatting
        print("-" * 40)
        print("ITERATION INITIALIZED - PERSISTENT ARCHITECTURE")
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
        print(f"Persistent pseudo dataset: {self.persistent_pseudo_dataset_name}")
        print(f"Manual corrections dataset: {self.manual_corrections_dataset_name}")
        print(f"Pseudo input dataset: {self.pseudo_input_dataset_name}")
        print(f"Initial annotations: {self.initial_annotated_dataset_name}")
        print(f"Inference model UID: {self.inference_model_uid}")
        print("-" * 40)

    def set_pseudo_dataset_for_inference(self, dataset_name, model_uid=None):
        """
        Set an existing dataset as the pseudo input dataset for inference, bypassing sampling.
        This allows you to use pre-existing datasets for inference without sampling new data.

        Args:
            dataset_name (str): Name of the existing dataset to use for inference
            model_uid (str, optional): Model UID to use for inference. If not provided,
                                     uses the current inference_model_uid

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Verify the dataset exists
            test_dataset = self.client.datasets.load(dataset_name, pull_policy="missing")
            print(f"✓ Dataset '{dataset_name}' found with {len(test_dataset)} images")

            # Set the pseudo input dataset name
            if self.manual_corrections_global:
                # For manual corrections mode
                self.pseudo_input_dataset_name = dataset_name
                print(f"✓ Set pseudo input dataset for manual corrections: {dataset_name}")
            else:
                # For auto pseudo-labeling mode
                self.persistent_pseudo_dataset_name = dataset_name
                self.pseudo_input_dataset_name = dataset_name
                print(f"✓ Set persistent pseudo dataset for inference: {dataset_name}")

            # Set model UID if provided
            if model_uid is not None:
                self.set_inference_model_uid(model_uid)

            # Update database with the dataset information
            self.db.update_iteration_field(
                self.flow_id, self.current_iteration,
                pseudo_input_dataset_name=self.pseudo_input_dataset_name,
                status='PRE_INFERENCE'  # New status to indicate ready for inference
            )

            print(f"✓ Ready to run inference on dataset: {dataset_name}")
            if self.inference_model_uid:
                print(f"✓ Using model: {self.inference_model_uid}")
            else:
                print("⚠ Warning: No inference model UID set. Use set_inference_model_uid() before running inference.")

            return True

        except Exception as e:
            print(f"✗ Error setting pseudo dataset: {e}")
            return False

    def setup_iteration_with_existing_data(self, manual_corrections, dataset_name, model_uid, current_flow=None):
        """
        Convenience function to set up a new iteration with existing dataset and model.
        This combines setup_next_iteration() with set_pseudo_dataset_for_inference().

        Args:
            manual_corrections (bool): Whether this iteration uses manual corrections
            dataset_name (str): Name of existing dataset to use
            model_uid (str): Model UID to use for inference
            current_flow (int, optional): Flow number if needed for kernel restart scenarios

        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # First set up the iteration
            self.setup_next_iteration(manual_corrections, current_flow)

            # Then set the existing dataset and model
            success = self.set_pseudo_dataset_for_inference(dataset_name, model_uid)

            if success:
                print("\n" + "=" * 50)
                print("ITERATION SETUP COMPLETE WITH EXISTING DATA")
                print("=" * 50)
                print(f"Flow: {self.flow_id}")
                print(f"Iteration: {self.current_iteration}")
                print(f"Mode: {'Manual Corrections' if manual_corrections else 'Auto Pseudo-labeling'}")
                print(f"Dataset: {dataset_name}")
                print(f"Model: {model_uid}")
                print(f"Status: Ready for inference")
                print("=" * 50)

            return success

        except Exception as e:
            print(f"✗ Error in iteration setup: {e}")
            return False

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
        Sample new images and prepare ALL pseudo data for re-inference.
        """
        print(f"\n=== SAMPLING FOR RE-INFERENCE (ITERATION {self.current_iteration}) ===")

        # Check if sampling already completed - FIXED: Added missing CVAT statuses
        current_status = self.db.get_iteration_status(self.flow_id, self.current_iteration)
        if current_status in ['SAMPLING_COMPLETE', 'INFERENCE', 'INFERENCE_COMPLETE', 'CVAT_EXPORT', 'CVAT_PENDING',
                              'CVAT_FAILED',
                              'MERGING', 'MERGE_COMPLETE', 'TRAINING', 'TRAINING_COMPLETE', 'EVALUATING',
                              'EVALUATION_COMPLETE', 'COMPLETED']:
            print(f"✓ Sampling already completed for iteration {self.current_iteration} (status: {current_status})")
            return

        self.db.update_status(self.flow_id, self.current_iteration, 'SAMPLING')

        if self.current_iteration > 0 and (self.inference_model_uid is None or self.inference_model_uid == ""):
            raise ValueError(f"Inference model UID not set for iteration {self.current_iteration}.")

        # Load training dataset to see what's already used
        try:
            current_training = self.client.datasets.load(self.train_dataset_name, pull_policy="missing")
            training_hashes = set(current_training.inputs.hash_iterator())
            print(f"✓ Current training dataset has {len(current_training)} images")
        except Exception as e:
            print(f"✗ Could not load training dataset {self.train_dataset_name}: {e}")
            raise

        # Find unused images
        full_hashes = list(self.full_dataset.inputs.hash_iterator())
        unused_hashes = list(set(full_hashes) - training_hashes)

        print(f"Total images in full dataset: {len(full_hashes)}")
        print(f"Images already in training: {len(training_hashes)}")
        print(f"Images NOT in training (available): {len(unused_hashes)}")

        if len(unused_hashes) < self.sample_size_per_iter:
            raise ValueError(
                f"Only {len(unused_hashes)} unused images available, but {self.sample_size_per_iter} requested.")

        # Sample new images
        sampled_hashes = random.sample(unused_hashes, self.sample_size_per_iter)
        index_map = {h: i for i, h in enumerate(full_hashes)}
        sampled_indices = [index_map[h] for h in sampled_hashes]
        new_sampled_data = self.full_dataset[sampled_indices]

        print(f"✓ Sampled {self.sample_size_per_iter} new images from unused set")

        if self.manual_corrections_global:
            # For manual corrections - create temp dataset (inputs only)
            temp_dataset = Dataset(inputs=new_sampled_data.inputs)
            self.client.datasets.save(self.pseudo_input_dataset_name, temp_dataset, exist="overwrite")
            self.client.datasets.push(self.pseudo_input_dataset_name, push_policy="version")
            print(f"✓ Created temp dataset for CVAT: {self.pseudo_input_dataset_name}")
        else:
            # FIXED: For pseudo-labeling - combine all inputs and save as inputs-only dataset
            persistent_pseudo_name = f"pseudo-f{self.current_flow}"

            try:
                # Load existing pseudo dataset and extract only inputs
                existing_pseudo = self.client.datasets.load(persistent_pseudo_name, pull_policy="missing")
                print(f"✓ Found existing pseudo dataset: {len(existing_pseudo)} images")

                # FIXED: Combine datasets properly, then extract inputs
                existing_dataset = Dataset(inputs=existing_pseudo.inputs)
                new_dataset = Dataset(inputs=new_sampled_data.inputs)
                combined_dataset = existing_dataset + new_dataset
                all_inputs = combined_dataset.inputs

                print(
                    f"✓ Combined: {len(existing_pseudo)} existing + {len(new_sampled_data)} new = {len(all_inputs)} total")

            except Exception as e:
                print(f"No existing pseudo dataset found: {e}")
                # First iteration - only new data
                all_inputs = new_sampled_data.inputs
                print(f"✓ First iteration: {len(all_inputs)} new inputs")

            # FIXED: Create dataset with ONLY inputs (no targets, no predictions)
            inputs_only_dataset = Dataset(inputs=all_inputs)

            # Save as the persistent pseudo dataset
            self.client.datasets.save(persistent_pseudo_name, inputs_only_dataset, exist="version")
            self.client.datasets.push(persistent_pseudo_name, push_policy="version")

            # Set this as our inference input
            self.pseudo_input_dataset_name = persistent_pseudo_name

            print(f"✓ Saved inputs-only dataset: {persistent_pseudo_name}")
            print(f"✓ Ready for re-inference on {len(all_inputs)} images with evolved model")

        # Update database
        self.db.update_iteration_field(
            self.flow_id, self.current_iteration,
            pseudo_input_dataset_name=self.pseudo_input_dataset_name,
            status='SAMPLING_COMPLETE'
        )

    def run_inference(self):
        """Run inference on the appropriate dataset based on mode."""
        if self.inference_model_uid is None:
            raise ValueError("Inference model UID not set. Use set_inference_model_uid() first.")

        # Check if inference already completed
        current_status = self.db.get_iteration_status(self.flow_id, self.current_iteration)
        if current_status in ['INFERENCE_COMPLETE', 'CVAT_EXPORT', 'CVAT_PENDING', 'CVAT_FAILED',
                              'MERGING', 'MERGE_COMPLETE', 'TRAINING', 'TRAINING_COMPLETE',
                              'EVALUATING', 'EVALUATION_COMPLETE', 'COMPLETED']:
            print(f"✓ Inference already completed for iteration {self.current_iteration} (status: {current_status})")
            return

        print(f"Running inference with model: {self.inference_model_uid}")
        self.db.update_status(self.flow_id, self.current_iteration, 'INFERENCE')

        if self.manual_corrections_global:
            # Manual corrections mode - run inference on temp dataset AND update existing pseudo dataset
            temp_dataset = self.pseudo_input_dataset_name  # temp-cvat-iter{N}-{flow_id}
            print(f"Running inference on temp dataset for manual corrections: {temp_dataset}")

            # Run inference on temp dataset for CVAT
            config = EvaluationConfig(
                model_name=self.inference_model_uid,
                dataset_name=temp_dataset,
                report_template=REPORT_TEMPLATE.EMPTY,
                batch_size=1
            )
            job = self.client.jobs.submit(config)
            self.predicted_dataset_name = self.client.jobs.get_dataset(job)
            print(f"✓ Temp dataset predictions saved as: {self.predicted_dataset_name}")

            # ALSO run inference on existing pseudo dataset to update old labels
            persistent_pseudo_name = f"pseudo-f{self.current_flow}"
            try:
                existing_pseudo = self.client.datasets.load(persistent_pseudo_name, pull_policy="missing")
                print(f"✓ Found existing pseudo dataset with {len(existing_pseudo)} images")
                print(f"✓ Re-running inference on existing pseudo dataset with evolved model...")

                config_pseudo = EvaluationConfig(
                    model_name=self.inference_model_uid,
                    dataset_name=persistent_pseudo_name,
                    report_template=REPORT_TEMPLATE.EMPTY,
                    batch_size=1
                )
                job_pseudo = self.client.jobs.submit(config_pseudo)
                updated_pseudo_predictions = self.client.jobs.get_dataset(job_pseudo)

                # Replace the persistent pseudo dataset with updated predictions
                predicted_pseudo_dataset = self.client.datasets.load(updated_pseudo_predictions)

                # Filter predictions by confidence and set as targets
                filtered_predictions = []
                for pred in predicted_pseudo_dataset.predictions:
                    if hasattr(pred, 'filter_by_confidence'):
                        filtered_pred = pred.filter_by_confidence(self.min_confidence)
                        filtered_predictions.append(filtered_pred)
                    else:
                        filtered_predictions.append(pred)

                predicted_pseudo_dataset.predictions = filtered_predictions
                predicted_pseudo_dataset.targets = predicted_pseudo_dataset.predictions

                # Replace the persistent pseudo dataset
                self.client.datasets.save(persistent_pseudo_name, predicted_pseudo_dataset, exist="version")
                self.client.datasets.push(persistent_pseudo_name, push_policy="version")

                print(f"✓ Updated persistent pseudo dataset: {persistent_pseudo_name}")
                print(
                    f"✓ Dataset now contains {len(predicted_pseudo_dataset)} images with updated predictions as targets")

            except Exception as e:
                print(f"No existing pseudo dataset found: {e}")
                print("✓ No existing pseudo labels to update")

        else:
            # Auto pseudo-labeling mode - run inference on persistent pseudo dataset
            inference_dataset = self.persistent_pseudo_dataset_name  # pseudo-{flow_id}
            print(f"Running inference on persistent pseudo dataset: {inference_dataset}")

            # Run inference on the appropriate dataset
            config = EvaluationConfig(
                model_name=self.inference_model_uid,
                dataset_name=inference_dataset,
                report_template=REPORT_TEMPLATE.EMPTY,
                batch_size=1
            )

            job = self.client.jobs.submit(config)
            self.predicted_dataset_name = self.client.jobs.get_dataset(job)

            print(f"Inference complete")
            print(f"Predictions saved as: {self.predicted_dataset_name}")

            # Auto mode: Replace the persistent pseudo dataset with filtered predictions
            self._replace_persistent_pseudo_dataset_with_predictions()
            print("✓ Persistent pseudo dataset updated with new predictions")

        # Update database
        self.db.update_iteration_field(
            self.flow_id, self.current_iteration,
            pseudo_output_dataset_name=self.predicted_dataset_name,
            status='INFERENCE_COMPLETE'
        )

    def _replace_persistent_pseudo_dataset_with_predictions(self):
        """
        CORRECTED: After inference, replace the persistent pseudo-f{flow} dataset with filtered predictions.
        This avoids duplicates by replacing the entire dataset instead of adding to it.
        """
        print(f"\n=== REPLACING PERSISTENT PSEUDO DATASET WITH PREDICTIONS ===")

        if self.predicted_dataset_name is None:
            raise ValueError("No predicted dataset to update from")

        # Load the predicted dataset and filter by confidence
        predicted_dataset = self.client.datasets.load(self.predicted_dataset_name)

        # Filter predictions by confidence and set as targets
        filtered_predictions = []
        for pred in predicted_dataset.predictions:
            if hasattr(pred, 'filter_by_confidence'):
                filtered_pred = pred.filter_by_confidence(self.min_confidence)
                filtered_predictions.append(filtered_pred)
            else:
                filtered_predictions.append(pred)

        predicted_dataset.predictions = filtered_predictions
        predicted_dataset.targets = predicted_dataset.predictions

        print(f"✓ Filtered predictions by confidence >= {self.min_confidence}")

        # Create the persistent pseudo dataset name for this flow
        persistent_pseudo_name = f"pseudo-f{self.current_flow}"

        # CORRECTED: Replace the entire dataset instead of adding to it
        self.client.datasets.save(persistent_pseudo_name, predicted_dataset, exist="version")
        self.client.datasets.push(persistent_pseudo_name, push_policy="version")

        print(f"✓ Replaced persistent pseudo dataset: {persistent_pseudo_name}")
        print(f"✓ Dataset now contains {len(predicted_dataset)} images with predictions as targets")
        print("=" * 50)

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
        """Export predictions to CVAT for manual correction."""
        if not self.manual_corrections_global:
            print("Manual corrections not enabled for this iteration.")
            return

        # Check if CVAT export already completed or in progress
        current_status = self.db.get_iteration_status(self.flow_id, self.current_iteration)
        if current_status in ['MERGING', 'MERGE_COMPLETE', 'TRAINING', 'TRAINING_COMPLETE',
                              'EVALUATING', 'EVALUATION_COMPLETE', 'COMPLETED']:
            print(f"✓ CVAT export already completed (status: {current_status})")
            return

        if not hasattr(self, 'cvat_authenticated') or not self.cvat_authenticated:
            raise RuntimeError("CVAT not connected. Please run pipeline.cvat_connect() first.")

        print("Exporting to CVAT...")
        self.db.update_status(self.flow_id, self.current_iteration, 'CVAT_EXPORT')

        try:
            self._export_to_cvat()
            print("✓ CVAT export completed successfully!")
            print("Complete your annotations in CVAT, then run merge with manual_annotation_path.")
            self.db.update_status(self.flow_id, self.current_iteration, 'CVAT_PENDING')

        except Exception as e:
            print(f"✗ CVAT export failed: {e}")
            self.db.update_status(self.flow_id, self.current_iteration, 'CVAT_FAILED')
            raise

    def _export_to_cvat(self):
        """Internal method to handle CVAT export with minimal output."""
        # Set up paths
        export_path = Path(self.local_path) / f"cvat_export_iter_{self.current_iteration}"

        if export_path.exists():
            import shutil
            shutil.rmtree(export_path)

        export_path.mkdir(parents=True, exist_ok=True)

        # Export dataset in COCO format
        json_filename = "annotations/instances_default.json"
        annotations_dir = export_path / "annotations"
        annotations_dir.mkdir(exist_ok=True)

        # Load and export dataset
        raw_ds = self.client.datasets.load(self.predicted_dataset_name)
        raw_ds.predictions = [
            insts.filter_by_confidence(self.min_confidence)
            for insts in raw_ds.predictions
        ]
        raw_ds.targets = raw_ds.predictions
        raw_ds.export_coco(path=export_path, json_filename=json_filename)

        print(f"✓ Exported {len(raw_ds)} images with {sum(len(pred) for pred in raw_ds.predictions)} annotations")

        # ========== JSON PREPROCESSING FOR CVAT COMPATIBILITY ==========
        json_path = export_path / json_filename
        with open(json_path, "r") as f:
            coco_data = json.load(f)

        # Fix image file names
        for image_entry in coco_data.get("images", []):
            image_entry["file_name"] = Path(image_entry["file_name"]).name

        # Remove problematic fields from annotations
        for annotation in coco_data.get("annotations", []):
            if "score" in annotation:
                del annotation["score"]
            if "iscrowd" not in annotation:
                annotation["iscrowd"] = 0
            if "area" not in annotation and "bbox" in annotation:
                bbox = annotation["bbox"]
                annotation["area"] = bbox[2] * bbox[3]

        # Fix category ID starting point
        categories = coco_data.get("categories", [])
        if categories and min(cat["id"] for cat in categories) == 0:
            for category in categories:
                category["id"] += 1
            for annotation in coco_data.get("annotations", []):
                annotation["category_id"] += 1

        # Fix image IDs
        images = coco_data.get("images", [])
        if images and min(img["id"] for img in images) == 0:
            image_id_mapping = {}
            for i, image in enumerate(images):
                old_id = image["id"]
                new_id = i + 1
                image["id"] = new_id
                image_id_mapping[old_id] = new_id

            for annotation in coco_data.get("annotations", []):
                old_image_id = annotation["image_id"]
                if old_image_id in image_id_mapping:
                    annotation["image_id"] = image_id_mapping[old_image_id]

        # Ensure annotation IDs are sequential
        for i, annotation in enumerate(coco_data.get("annotations", [])):
            annotation["id"] = i + 1

        # Ensure required fields exist
        if "info" not in coco_data:
            coco_data["info"] = {"description": "Pseudo-labeling Export", "version": "1.0"}
        if "licenses" not in coco_data:
            coco_data["licenses"] = []

        # Save the corrected COCO file
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=2)

        print(f"✓ COCO annotations cleaned and saved")

        # ========== UPLOAD TO CVAT ==========
        dataset_labels = self._get_dataset_labels()

        try:
            config = getattr(self, 'cvat_project_config', {})
            project_name_to_use = config.get('project_name') or f"{self.flow_id}-project"

            # Create/find project
            project_id = self.cvat.get_or_create_project(
                project_name_to_use,
                dataset_labels,
                project_id=config.get('project_id')
            )

            # Create task
            task_name = f"{self.flow_id}-iter{self.current_iteration}"
            task_id = self.cvat.create_task(project_id, task_name)

            # Upload images
            image_dir = export_path / "images"
            if image_dir.exists():
                self.cvat.upload_images(task_id, str(image_dir))
                print(f"✓ Uploaded images to CVAT")

            # Upload annotations
            self.cvat.upload_annotations(task_id, str(json_path))
            print(f"✓ Uploaded annotations to CVAT")

            task_url = f"{self.cvat.cvat_url}/tasks/{task_id}"
            print(f"✓ CVAT task ready: {task_url}")

        except Exception as e:
            print(f"✗ CVAT upload failed: {e}")
            raise

    def merge_pseudo_labels(self, manual_annotation_path=None, pseudo_only=False):
        """
        SIMPLIFIED merge logic with clear separation of manual vs auto modes.
        """
        # Check if merge already completed
        current_status = self.db.get_iteration_status(self.flow_id, self.current_iteration)
        if current_status in ['MERGE_COMPLETE', 'TRAINING', 'TRAINING_COMPLETE',
                              'EVALUATING', 'EVALUATION_COMPLETE', 'COMPLETED']:
            print(f"Merge already completed for iteration {self.current_iteration} (status: {current_status})")
            return

        if self.manual_corrections_global and current_status == 'CVAT_PENDING':
            if manual_annotation_path is None:
                print("CVAT annotations are pending. Please complete your annotations in CVAT first.")
                print("Then provide the manual_annotation_path parameter of the new annotation json file (in COCO) to proceed with merge.")
                return
            else:
                print("✓ Manual annotation path provided, proceeding with merge...")

        print("Starting simplified merge process...")
        self.db.update_status(self.flow_id, self.current_iteration, 'MERGING')

        if self.manual_corrections_global:
            print("\n=== MANUAL CORRECTION MODE ===")

            if manual_annotation_path is None:
                raise ValueError(
                    "Manual corrections mode requires manual_annotation_path parameter.\n"
                    "Please provide the path to your corrected COCO annotation file."
                )

            # Process manual corrections and add to manual corrections dataset
            self._process_and_accumulate_manual_corrections(manual_annotation_path)

        else:
            print("\n=== AUTO PSEUDO-LABELING MODE ===")
            # In auto mode, the persistent pseudo dataset was already replaced in run_inference()
            print("✓ Pseudo dataset already updated after inference")

        # Now rebuild training dataset with all components
        self._rebuild_training_dataset_simplified(pseudo_only=pseudo_only)

        # Update status after successful merge
        self.db.update_status(self.flow_id, self.current_iteration, 'MERGE_COMPLETE')

    def _process_and_accumulate_manual_corrections(self, annotation_file_path):
        """
        Process new manual corrections and accumulate them in the manual corrections dataset.
        """
        print("Processing manual corrections...")

        annotation_file = Path(annotation_file_path)

        # Validate file
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        if not annotation_file.name.endswith('.json'):
            raise ValueError(f"File must be a JSON file. Got: {annotation_file.name}")

        try:
            with open(annotation_file, 'r') as f:
                json.load(f)
            print(f"✓ Valid JSON file: {annotation_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")

        # Set up temp processing path
        export_path = Path(self.local_path) / f"cvat_export_iter_{self.current_iteration}"
        target_annotation_file = export_path / "annotations" / "instances_default.json"
        target_annotation_file.parent.mkdir(parents=True, exist_ok=True)

        # Copy annotation file to expected location
        import shutil
        shutil.copy2(annotation_file, target_annotation_file)
        print(f"✓ Copied annotation file to: {target_annotation_file}")

        # Import the corrected annotations
        new_corrected_dataset = Dataset.import_coco(
            path=str(export_path),
            image_subdir="images",
            coco_filename="annotations/instances_default.json"
        )
        print(f"✓ Imported {len(new_corrected_dataset)} manually corrected images")

        # Accumulate in manual corrections dataset
        manual_corrections_name = f"manual-corrections-f{self.current_flow}"

        try:
            existing_manual = self.client.datasets.load(manual_corrections_name, pull_policy="missing")
            print(f"✓ Found existing manual corrections: {len(existing_manual)} images")

            updated_manual = existing_manual + new_corrected_dataset
            print(
                f"✓ Accumulated: {len(existing_manual)} existing + {len(new_corrected_dataset)} new = {len(updated_manual)} total")

        except Exception as e:
            print(f"No existing manual corrections found: {e}")
            updated_manual = new_corrected_dataset
            print(f"✓ Created new manual corrections dataset with {len(new_corrected_dataset)} images")

        # Save the accumulated manual corrections
        self.client.datasets.save(manual_corrections_name, updated_manual, exist="version")
        self.client.datasets.push(manual_corrections_name, push_policy='version')

        print(f"✓ Saved manual corrections dataset: {manual_corrections_name}")

        # Update database with manual corrections dataset name
        self.db.update_iteration_field(
            self.flow_id, self.current_iteration,
            pseudo_output_dataset_name=manual_corrections_name
        )

    def _rebuild_training_dataset_simplified(self, pseudo_only=False):
        """
        SIMPLIFIED rebuilding logic:
        1. Initial dataset (unless pseudo_only=True)
        2. Manual corrections dataset for this flow (if exists)
        3. Pseudo dataset for this flow (if exists)
        """
        print(f"\n=== REBUILDING TRAINING DATASET (SIMPLIFIED) ===")
        if pseudo_only:
            print("PSEUDO-ONLY MODE: Skipping initial dataset")

        # Start with initial dataset (unless pseudo_only)
        if not pseudo_only:
            merged_dataset = self.client.datasets.load(self.initial_annotated_dataset_name, pull_policy="missing")
            component_info = [f"Initial GT: {len(merged_dataset)} images"]
            print(f"✓ Started with initial dataset: {len(merged_dataset)} images")
        else:
            merged_dataset = None
            component_info = []
            print("✓ Skipped initial dataset (pseudo-only mode)")

        # Add manual corrections for this flow (if exists)
        manual_corrections_name = f"manual-corrections-f{self.current_flow}"
        try:
            manual_dataset = self.client.datasets.load(manual_corrections_name, pull_policy="missing")
            if merged_dataset is None:
                merged_dataset = manual_dataset
            else:
                merged_dataset = merged_dataset + manual_dataset
            component_info.append(f"Manual corrections: {len(manual_dataset)} images")
            print(f"✓ Added manual corrections: {len(manual_dataset)} images, total: {len(merged_dataset)}")
        except Exception as e:
            print(f"✓ No manual corrections found for this flow: {e}")

        # Add pseudo dataset for this flow (if exists)
        pseudo_name = f"pseudo-f{self.current_flow}"
        try:
            pseudo_dataset = self.client.datasets.load(pseudo_name, pull_policy="missing")
            if merged_dataset is None:
                merged_dataset = pseudo_dataset
            else:
                merged_dataset = merged_dataset + pseudo_dataset
            component_info.append(f"Pseudo labels: {len(pseudo_dataset)} images")
            print(f"✓ Added pseudo dataset: {len(pseudo_dataset)} images, total: {len(merged_dataset)}")
        except Exception as e:
            print(f"✓ No pseudo dataset found for this flow: {e}")

        # Validate we have at least one component
        if merged_dataset is None:
            raise RuntimeError("No datasets available for training. Check your configuration.")

        # Ensure label map consistency (use initial dataset as reference if available)
        if not pseudo_only:
            reference_dataset = self.client.datasets.load(self.initial_annotated_dataset_name, pull_policy="missing")
        else:
            # In pseudo_only mode, use the first available dataset as reference
            if any("Manual corrections" in info for info in component_info):
                reference_dataset = self.client.datasets.load(manual_corrections_name, pull_policy="missing")
            else:
                reference_dataset = self.client.datasets.load(pseudo_name, pull_policy="missing")

        if reference_dataset.targets.has_frozen_label_map():
            label_map = reference_dataset.targets.get_frozen_label_map()
        elif reference_dataset.targets.has_frozen_labels():
            from onedl.core import LabelMap
            label_map = LabelMap.from_labels(reference_dataset.targets.get_frozen_labels())
        else:
            label_map = reference_dataset.targets.generate_label_map()

        # Apply label map and convert to ObjectIDColumn
        merged_dataset.targets.freeze_label_map(label_map)
        from onedl.datasets.columns import ObjectIDColumn
        merged_dataset.targets = ObjectIDColumn(merged_dataset.targets)

        # Save final training dataset
        self.client.datasets.save(self.train_dataset_name, merged_dataset, exist='version', skip_validation=True)
        self.client.datasets.push(self.train_dataset_name, push_policy="version")

        # Final summary
        print(f"\n✓ TRAINING DATASET REBUILT:")
        for info in component_info:
            print(f"  - {info}")
        print(f"  - Total: {len(merged_dataset)} images")
        print(f"  - Saved as: {self.train_dataset_name}")
        print(f"  - Label map: {label_map}")
        print("=" * 50)

    def train_model(self):
        """Train model on current dataset with logging. Can be run multiple times without restrictions."""
        if self.train_cfg is None:
            raise ValueError(
                "Training configuration not set. Run setup_training_config() or set train_cfg dictionary first.")

        current_status = self.db.get_iteration_status(self.flow_id, self.current_iteration)
        if current_status in ['TRAINING_COMPLETE', 'EVALUATING', 'EVALUATION_COMPLETE', 'COMPLETED']:
            print(f"✓ Training already completed for iteration {self.current_iteration} (status: {current_status})")
            return

        print("Starting model training...")
        self.db.update_status(self.flow_id, self.current_iteration, 'TRAINING')

        # Use initial_annotated_dataset for first training, train_dataset_name for iterations
        if self.current_iteration == 0:
            dataset_to_use = self.initial_annotated_dataset_name
        else:
            dataset_to_use = self.train_dataset_name

        # Use the widget/dictionary config method
        model_type = self.train_cfg['model_type']

        # Base configuration parameters
        base_params = {
            'train_dataset_name': dataset_to_use,
            'val_dataset_name': self.validation_dataset,
            'num_epochs': self.train_cfg['epochs'],
            'batch_size': self.train_cfg['batch_size']
        }

        # Add input_size if specified (for semantic segmentation)
        if self.train_cfg.get('input_size'):
            base_params['input_size'] = self.train_cfg['input_size']

        # Add backbone if selected
        if self.train_cfg.get('backbone') is not None:
            if model_type == "UPerNetConfig":
                from onedl.zoo.semantic_segmentation import UPerNetBackbone
                base_params['backbone'] = UPerNetBackbone(self.train_cfg['backbone'])
            else:
                base_params['backbone'] = self.train_cfg['backbone']

        # Create model configuration based on type
        try:
            if model_type == "FasterRCNNConfig":
                from onedl.zoo.object_detection.mmdetection import FasterRCNNConfig
                config = FasterRCNNConfig(**base_params)
            elif model_type == "MaskRCNNConfig":
                from onedl.zoo.instance_segmentation.mmdetection import MaskRCNNConfig
                config = MaskRCNNConfig(**base_params)
            elif model_type == "UPerNetConfig":
                from onedl.zoo.semantic_segmentation import UPerNetConfig
                config = UPerNetConfig(**base_params)
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

            # Add input_size for fallback too
            if self.train_cfg.get('input_size'):
                basic_params['input_size'] = self.train_cfg['input_size']

            if model_type == "FasterRCNNConfig":
                from onedl.zoo.object_detection.mmdetection import FasterRCNNConfig
                config = FasterRCNNConfig(**basic_params)
            elif model_type == "MaskRCNNConfig":
                from onedl.zoo.instance_segmentation.mmdetection import MaskRCNNConfig
                config = MaskRCNNConfig(**basic_params)
            elif model_type == "UPerNetConfig":
                from onedl.zoo.semantic_segmentation import UPerNetConfig
                config = UPerNetConfig(**basic_params)

            if hasattr(config, 'tracking'):
                config.tracking.enabled = False

        print(f"Training {model_type} on dataset: {dataset_to_use}")
        print(f"Configuration: {self.train_cfg['epochs']} epochs, batch size {self.train_cfg['batch_size']}")
        if self.train_cfg.get('backbone'):
            print(f"Backbone: {self.train_cfg['backbone']}")
        if self.train_cfg.get('input_size'):
            print(f"Input size: {self.train_cfg['input_size']}")

        # Submit training job
        self.model_uid = self.client.jobs.submit(config)
        print(f"Training job submitted")
        print(f"Model UID: {self.model_uid}")

        # Check job state and update database accordingly
        job_state = self.client.jobs.get_state(self.model_uid)
        print(f"Training job state: {job_state}")

        # Map job state to database status
        if job_state == "DONE":
            db_status = 'TRAINING_COMPLETE'
        elif job_state == "FAILED":
            db_status = 'TRAINING_FAILED'
        elif job_state == "CANCELLED":
            db_status = 'TRAINING_CANCELLED'
        elif job_state in ["RUNNING", "WAITING", "UNALLOCABLE"]:
            db_status = 'TRAINING'
        else:
            db_status = 'TRAINING'  # Default fallback

        # Update database with model UID and appropriate status
        self.db.update_iteration_field(
            self.flow_id, self.current_iteration,
            model_uid=self.model_uid,
            status=db_status
        )

    def evaluate_model(self):
        """Evaluate the current model on validation dataset. Can be run multiple times without restrictions."""
        # Check if evaluation already completed
        current_status = self.db.get_iteration_status(self.flow_id, self.current_iteration)
        if current_status in ['EVALUATION_COMPLETE', 'COMPLETED']:
            print(f"✓ Evaluation already completed for iteration {self.current_iteration} (status: {current_status})")
            return

        # Ensure we have a model to evaluate
        if not self.model_uid:
            # Try to recover from database
            self.db.cursor.execute(
                'SELECT model_uid FROM iteration_metadata WHERE flow_id = ? AND iteration = ?',
                (self.flow_id, self.current_iteration)
            )
            result = self.db.cursor.fetchone()
            if result and result[0]:
                self.model_uid = result[0]
                print(f"✓ Using model UID from database: {self.model_uid}")
            else:
                raise RuntimeError("No model UID available for evaluation. Please run train_model() first.")

        print(f'Evaluating {self.model_uid} on {self.validation_dataset}')
        self.db.update_status(self.flow_id, self.current_iteration, 'EVALUATING')

        from onedl.zoo.eval import EvaluationConfig, Device
        config = EvaluationConfig(
            model_name=self.model_uid,
            dataset_name=self.validation_dataset,
            device=Device.CPU
        )

        self.evaluation_uid = self.client.jobs.submit(config)
        print(f"Evaluation job submitted")
        print(f"Evaluation UID: {self.evaluation_uid}")

        # Check job state and update database accordingly
        job_state = self.client.jobs.get_state(self.evaluation_uid)
        print(f"Evaluation job state: {job_state}")

        # Map job state to database status for evaluation
        if job_state == "DONE":
            # Get evaluation results if job is done
            try:
                report_name = self.client.jobs.get_evaluation(self.evaluation_uid)
                report_info = self.client.evaluations.get_info(report_name)
                report_url = self.client.evaluations.get_uri(report_name)
                import json
                self.evaluation_info_str = json.dumps(report_info.get("metrics", {}))

                print("Evaluation complete")
                print(f"Report URL: {report_url}")
                print(f"Metrics: {self.evaluation_info_str}")

                # FIXED: Automatically complete the iteration when evaluation is done
                self.db.update_iteration_field(
                    self.flow_id, self.current_iteration,
                    evaluation_uid=self.evaluation_uid,
                    evaluation_info=self.evaluation_info_str,
                    status='COMPLETED'
                )
                self.db.complete_iteration(self.flow_id, self.current_iteration)
                print("✓ Iteration automatically marked as COMPLETED")
                return

            except Exception as e:
                print(f"Error retrieving evaluation results: {e}")
                self.evaluation_info_str = ""
                db_status = 'EVALUATION_COMPLETE'

        elif job_state == "FAILED":
            self.evaluation_info_str = ""
            db_status = 'EVALUATION_FAILED'
        elif job_state == "CANCELLED":
            self.evaluation_info_str = ""
            db_status = 'EVALUATION_CANCELLED'
        elif job_state in ["RUNNING", "WAITING", "UNALLOCABLE"]:
            self.evaluation_info_str = ""
            db_status = 'EVALUATING'
        else:
            self.evaluation_info_str = ""
            db_status = 'EVALUATING'  # Default fallback

        # Update database with evaluation results (for non-DONE cases)
        self.db.update_iteration_field(
            self.flow_id, self.current_iteration,
            evaluation_uid=self.evaluation_uid,
            evaluation_info=self.evaluation_info_str,
            status=db_status
        )

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
            main_dataset=self.main_dataset_name,
            validation_dataset=self.validation_dataset,
            train_cfg=self.train_cfg
        )

        print(f"Iteration 0 logged for {self.flow_id}")

    def log_iteration_0_external_model(self, external_model_uid):
        """Log iteration 0 using an external model."""
        # Set the model UID
        self.model_uid = external_model_uid
        self.evaluation_uid = ""  # No evaluation for external model
        self.evaluation_info_str = ""  # No evaluation info

        print("=== LOGGING ITERATION 0 WITH EXTERNAL MODEL ===")
        print(f"External model UID: {external_model_uid}")
        print(f"Flow: {self.flow_id}")

        # Use the database method
        self.db.log_iteration_0_external_model(
            flow_id=self.flow_id,
            iteration=0,
            num_gt_images=self.n_initial_samples,
            num_gt_images_added=self.n_initial_samples,
            num_pseudo_images=0,
            num_pseudo_images_added=0,
            total_train_size=self.n_initial_samples,
            main_dataset=self.main_dataset_name,
            validation_dataset=self.validation_dataset,
            train_dataset=self.train_dataset_name,
            pseudo_input_dataset_name="",
            pseudo_output_dataset_name="",
            inference_model_uid="",
            model_uid=self.model_uid,
            evaluation_uid=self.evaluation_uid,
            evaluation_info=self.evaluation_info_str,
            manual_correction=False,
            train_cfg=self.train_cfg
        )

        print(f"Iteration 0 logged with external model: {external_model_uid}")
        print("Ready to proceed with pseudo-labeling iterations")

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

    def clear_database(self, confirm_phrase="DELETE ALL DATA"):
        """
        DANGER: Completely clears the entire database.

        This function deletes ALL data from the metadata database, including all flows,
        iterations, and tracking information. This is intended for testing purposes only.

        Args:
            confirm_phrase (str): Must type "DELETE ALL DATA" exactly to confirm deletion

        Raises:
            ValueError: If confirmation phrase doesn't match exactly
        """
        print("=" * 50)
        print("DANGER: DATABASE DELETION REQUESTED")
        print("=" * 50)
        print("This will permanently delete ALL pipeline metadata including:")
        print("- All flow data")
        print("- All iteration records")
        print("- All model and evaluation tracking")
        print("- All CVAT project associations")
        print("\nThis action CANNOT be undone!")
        print("=" * 50)

        # Get user confirmation
        user_input = input(f'Type "{confirm_phrase}" exactly to confirm deletion: ')

        if user_input != confirm_phrase:
            print("CANCELLED: Confirmation phrase incorrect. Database deletion cancelled.")
            print("Database remains intact.")
            return False

        try:
            # Drop the entire table
            self.db.cursor.execute('DROP TABLE IF EXISTS iteration_metadata')
            self.db.conn.commit()

            # Recreate the empty table structure
            self.db._initialize_metadata_db()

            print("SUCCESS: Database successfully cleared!")
            print("SUCCESS: Empty database structure recreated.")
            print("Pipeline is now reset and ready for fresh use.")

            # Reset pipeline state
            self.current_iteration = 0
            self.model_uid = None
            self.evaluation_uid = None
            self.evaluation_info_str = None
            self.inference_model_uid = None

            print("SUCCESS: Pipeline state reset to initial values.")
            return True

        except Exception as e:
            print(f"ERROR: Error clearing database: {e}")
            print("Database may be in an inconsistent state. Check manually.")
            return False

    def get_all_flows_summary(self):
        """
        Display a summary of all flows and their status in the database.
        Useful before clearing database to see what will be lost.
        """
        print("=" * 60)
        print("DATABASE CONTENTS SUMMARY")
        print("=" * 60)

        # Get all flows
        self.db.cursor.execute('''
            SELECT DISTINCT flow_id 
            FROM iteration_metadata 
            ORDER BY flow_id
        ''')
        flows = [row[0] for row in self.db.cursor.fetchall()]

        if not flows:
            print("Database is empty - no flows found.")
            return

        total_iterations = 0
        total_completed = 0

        for flow_id in flows:
            print(f"\nFlow: {flow_id}")
            print("-" * 30)

            # Get iterations for this flow
            self.db.cursor.execute('''
                SELECT iteration, status, num_gt_images, num_pseudo_images, 
                       model_uid, completed_timestamp
                FROM iteration_metadata 
                WHERE flow_id = ? 
                ORDER BY iteration
            ''', (flow_id,))

            iterations = self.db.cursor.fetchall()
            flow_iterations = len(iterations)
            flow_completed = len([i for i in iterations if i[1] == 'COMPLETED'])

            total_iterations += flow_iterations
            total_completed += flow_completed

            print(f"Iterations: {flow_iterations} total, {flow_completed} completed")

            for iteration, status, gt_imgs, pseudo_imgs, model_uid, completed in iterations:
                status_icon = "✓" if status == "COMPLETED" else "○"
                total_imgs = (gt_imgs or 0) + (pseudo_imgs or 0)
                model_short = model_uid[:8] + "..." if model_uid else "None"
                print(f"  {status_icon} Iter {iteration}: {status} | {total_imgs} images | Model: {model_short}")

        print("=" * 60)
        print(f"TOTAL: {len(flows)} flows, {total_iterations} iterations, {total_completed} completed")
        print("=" * 60)