import random
import json
from onedl.datasets import Dataset
from onedl.zoo.eval import EvaluationConfig, REPORT_TEMPLATE, Device
from onedl.core import LabelMap
from onedl.datasets.columns import ObjectIDColumn
from onedl.zoo.instance_segmentation.mmdetection import MaskRCNNConfig
from onedl.zoo.mmlabs.mmdetection.rtmdet import RTMDetConfig  # add at some point. We need to keep updating models
from onedl.zoo.object_detection.mmdetection import FasterRCNNConfig

# Try to import interactive packages
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

try:
    import inquirer

    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False


# --------- STEP 1: Sampling from Unlabeled Pool ---------

def sample_unseen_inputs_only(client, train_dataset_name, full_dataset, sample_size_per_iter,
                              pseudo_input_dataset_name):
    """
    Sample unseen inputs from the full dataset and save as pseudo input dataset.

    Args:
        client: OneDL client
        train_dataset_name (str): Current training dataset name
        full_dataset: Full dataset object
        sample_size_per_iter (int): Number of samples per iteration
        pseudo_input_dataset_name (str): Name for the pseudo input dataset

    Returns:
        str: The pseudo input dataset name
    """
    training_dataset = client.datasets.load(train_dataset_name, pull_policy="missing")
    full_hashes = list(full_dataset.inputs.hash_iterator())
    train_hashes = set(training_dataset.inputs.hash_iterator())
    remaining_hashes = list(set(full_hashes) - train_hashes)

    if len(remaining_hashes) < sample_size_per_iter:
        raise ValueError(f"Only {len(remaining_hashes)} unseen images left, but {sample_size_per_iter} requested.")

    sampled_hashes = random.sample(remaining_hashes, sample_size_per_iter)
    index_map = {h: i for i, h in enumerate(full_hashes)}
    sampled_indices = [index_map[h] for h in sampled_hashes]
    sampled_data = full_dataset[sampled_indices]

    input_only_dataset = Dataset(inputs=sampled_data.inputs)
    client.datasets.save(pseudo_input_dataset_name, dataset=input_only_dataset, exist="overwrite")
    client.datasets.push(pseudo_input_dataset_name)

    print(f"Sampled {sample_size_per_iter} unseen → saved as '{pseudo_input_dataset_name}'")
    return pseudo_input_dataset_name


# --------- STEP 2: Inference ---------

def run_pseudo_label_inference(inference_model_uid, pseudo_input_dataset_name, client):
    """
    Run inference on pseudo input dataset to generate predictions.

    Args:
        inference_model_uid (str): Model UID to use for inference
        pseudo_input_dataset_name (str): Input dataset name
        client: OneDL client

    Returns:
        str: The predicted dataset name
    """
    config = EvaluationConfig(
        model_name=inference_model_uid,
        dataset_name=pseudo_input_dataset_name,
        report_template=REPORT_TEMPLATE.EMPTY,
        batch_size=1
    )

    job = client.jobs.submit(config)
    predicted_dataset_name = client.jobs.get_dataset(job)
    print(f"Inference complete. Predictions saved as '{predicted_dataset_name}'")
    return predicted_dataset_name


# --------- STEP 3: Merge Predictions into Training Set ---------

def merge_pseudo_labels_into_training(client, train_dataset_name, predicted_dataset_name, min_confidence):
    """
    Merge pseudo-labeled predictions into the training dataset.

    Args:
        client: OneDL client
        train_dataset_name (str): Training dataset name
        predicted_dataset_name (str): Predicted dataset name
        min_confidence (float): Minimum confidence threshold

    Returns:
        Dataset: The merged dataset
    """
    training_dataset = client.datasets.load(train_dataset_name, pull_policy="missing")
    pseudo_dataset = client.datasets.load(predicted_dataset_name)

    pseudo_dataset.predictions = [
        insts.filter_by_confidence(min_confidence)
        for insts in pseudo_dataset.predictions
    ]
    pseudo_dataset.targets = pseudo_dataset.predictions

    if training_dataset.targets.has_frozen_label_map():
        label_map = training_dataset.targets.get_frozen_label_map()
    elif training_dataset.targets.has_frozen_labels():
        label_map = LabelMap.from_labels(training_dataset.targets.get_frozen_labels())
    else:
        label_map = training_dataset.targets.generate_label_map()

    pseudo_dataset.targets.freeze_label_map(label_map)

    training_dataset.targets = ObjectIDColumn(training_dataset.targets)
    pseudo_dataset.targets = ObjectIDColumn(pseudo_dataset.targets)

    merged = training_dataset + pseudo_dataset
    client.datasets.save(train_dataset_name, merged, exist="versions", skip_validation=True)
    client.datasets.push(train_dataset_name, push_policy="version")

    print(f"Merged {predicted_dataset_name} into '{train_dataset_name}'")
    return merged


# --------- STEP 4: Train Model ---------

def train_model_on_current_dataset(train_cfg, current_iteration, train_dataset_name,
                                   initial_annotated_dataset_name, validation_dataset, client):
    """
    Enhanced training function that uses training configuration.

    Args:
        train_cfg (dict): Training configuration dictionary
        current_iteration (int): Current iteration number (0 for initial training)
        train_dataset_name (str): Current training dataset name
        initial_annotated_dataset_name (str): Initial annotated dataset name
        validation_dataset (str): Validation dataset name
        client: OneDL client

    Returns:
        str: The UID of the newly trained model
    """
    if train_cfg is None:
        raise ValueError("Training configuration not set. Run setup_training_config() in global initializer first.")

    model_type = train_cfg['model_type']

    # Use initial_annotated_dataset for first training, train_dataset_name for iterations
    if current_iteration == 0:
        dataset_to_use = initial_annotated_dataset_name
    else:
        dataset_to_use = train_dataset_name

    # Base configuration parameters
    base_params = {
        'train_dataset_name': dataset_to_use,
        'val_dataset_name': validation_dataset,
        'num_epochs': train_cfg['epochs'],
        'batch_size': train_cfg['batch_size']
    }

    # Add backbone if selected
    if train_cfg.get('backbone') is not None:
        base_params['backbone'] = train_cfg['backbone']

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
            'val_dataset_name': validation_dataset,
            'num_epochs': train_cfg['epochs'],
            'batch_size': train_cfg['batch_size']
        }

        if model_type == "FasterRCNNConfig":
            config = FasterRCNNConfig(**basic_params)
        else:
            config = MaskRCNNConfig(**basic_params)

        if hasattr(config, 'tracking'):
            config.tracking.enabled = False

    # Submit training job
    print(f"Training {model_type} on dataset '{dataset_to_use}'")
    print(f"Config: {train_cfg['epochs']} epochs, batch size {train_cfg['batch_size']}")
    if train_cfg.get('backbone'):
        print(f"Backbone: {train_cfg['backbone']}")

    model_uid = client.jobs.submit(config)
    print(f"Training job submitted. UID: {model_uid}")

    return model_uid


# --------- STEP 5: Evaluate Model ---------

def evaluate_current_model(model_uid, validation_dataset, client):
    """
    Evaluate the current model on validation dataset.

    Args:
        model_uid (str): Model UID to evaluate
        validation_dataset (str): Validation dataset name
        client: OneDL client

    Returns:
        tuple: (evaluation_uid, evaluation_info_str)
    """
    config = EvaluationConfig(
        model_name=model_uid,
        dataset_name=validation_dataset,
        device=Device.CPU
    )

    evaluation_uid = client.jobs.submit(config)
    job_state = client.jobs.get_state(evaluation_uid)

    if job_state in ("DONE", "FAILED"):
        report_name = client.jobs.get_evaluation(evaluation_uid)
        report_info = client.evaluations.get_info(report_name)
        report_url = client.evaluations.get_uri(report_name)
        evaluation_info_str = json.dumps(report_info.get("metrics", {}))

        print("Evaluation complete.")
        print("Report:", report_url)
        print("Metrics:", evaluation_info_str)
    else:
        evaluation_info_str = ""
        print("Evaluation incomplete or failed.")

    return evaluation_uid, evaluation_info_str


# --------- Training Configuration Functions ---------

def get_available_backbones(model_type):
    """
    Get available backbones for the specified model type
    """
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


def setup_training_config():
    """
    Interactive training configuration setup that works in both Jupyter and terminal.
    Call this function in your global initializer section.

    Returns:
    - config (dict): Training configuration dictionary
    """
    print("=== TRAINING CONFIGURATION SETUP ===")

    if JUPYTER_AVAILABLE:
        return _setup_config_jupyter()
    elif INQUIRER_AVAILABLE:
        return _setup_config_terminal()
    else:
        raise RuntimeError("Neither Jupyter widgets nor inquirer available. Please install ipywidgets or inquirer.")


def _setup_config_jupyter():
    """Jupyter notebook interface using ipywidgets"""
    from IPython.display import display, clear_output
    import ipywidgets as widgets

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
        available_backbones = get_available_backbones(model_type)
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

    print("Please use the widgets above to configure your training settings.")
    return config


def _setup_config_terminal():
    """Terminal interface (for command line use) NOT USED BUT JUST IN CASE"""
    model_choices = [
        ('FasterRCNNConfig (Object Detection)', 'FasterRCNNConfig'),
        ('MaskRCNNConfig (Instance Segmentation)', 'MaskRCNNConfig')
    ]

    questions = [
        inquirer.List('model_type',
                      message="Select model type",
                      choices=[choice[0] for choice in model_choices],
                      ),
    ]

    answers = inquirer.prompt(questions)

    model_type = None
    task_type = None
    for display_name, actual_type in model_choices:
        if answers['model_type'] == display_name:
            model_type = actual_type
            task_type = 'object_detection' if actual_type == 'FasterRCNNConfig' else 'instance_segmentation'
            break

    print(f"Selected: {model_type} for {task_type}")

    available_backbones = get_available_backbones(model_type)
    selected_backbone = None

    if available_backbones:
        backbone_questions = [
            inquirer.List('backbone',
                          message=f"Select backbone for {model_type}",
                          choices=[str(backbone) for backbone in available_backbones],
                          ),
        ]

        backbone_answers = inquirer.prompt(backbone_questions)

        for backbone in available_backbones:
            if str(backbone) == backbone_answers['backbone']:
                selected_backbone = backbone
                break
    else:
        print("No backbones available. Using default configuration.")

    epochs = int(input("Epochs (default 50): ") or "50")
    batch_size = int(input("Batch size (default 6): ") or "6")

    config = {
        'model_type': model_type,
        'task_type': task_type,
        'backbone': selected_backbone,
        'epochs': epochs,
        'batch_size': batch_size
    }

    print(f"\n=== CONFIGURATION COMPLETE ===")
    print(f"Model: {model_type}")
    print(f"Task: {task_type}")
    print(f"Backbone: {selected_backbone}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    return config