# Pseudo-Labeling Pipeline for Object Detection & Instance Segmentation

![Pseudo-Labeling Workflow](/uploads/27032eedef10aae81e822e0d9e35c123/workflow.gif)

An automated pseudo-labeling pipeline designed to streamline the annotation process for computer vision tasks. This tool iteratively improves model performance by using an initial model trained on a small set of manually annotated data to generate labels on new images, which can then be refined and used to retrain progressively better models.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Key Concepts](#key-concepts)
- [Pipeline Workflow](#pipeline-workflow)
- [Configuration Guide](#configuration-guide)
- [Manual Correction Setup (CVAT)](#manual-correction-setup-cvat)
- [Database Management](#database-management)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Overview

### What is Pseudo-Labeling?

Pseudo-labeling is a semi-supervised learning technique where:
1. **Start Small**: Begin with a small set of manually annotated images
2. **Train Initial Model**: Create a baseline model from your initial annotations
3. **Generate Predictions**: Use the model to predict labels on new, unlabeled images
4. **Refine Labels**: Either automatically accept high-confidence predictions or manually correct them
5. **Expand Training Set**: Add the new labeled data to your training set
6. **Iterate**: Retrain the model on the expanded dataset and repeat

### Key Features

- **Automated Pipeline**: Complete workflow from data preparation to model training
- **Database Tracking**: SQLite database tracks all iterations, models, and metrics
- **CVAT Integration**: Optional manual correction workflow through CVAT annotation tool
- **Flexible Configuration**: Support for FasterRCNN and MaskRCNN architectures
- **Progress Monitoring**: Real-time status tracking and performance evaluation
- **Flow Management**: Organize multiple experiments with different configurations

### What You'll Need

- **OneDL Platform Access**: For dataset management and model training
- **Initial Annotated Dataset**: Small set of manually labeled images (recommended: 50-200 images)
- **Unlabeled Dataset**: Large collection of images to progressively label
- **Validation Dataset**: Separate dataset for model evaluation
- **CVAT Account** (Optional): For manual correction workflow

## Installation

### 1. Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

### 2. Download Pipeline Files

Download the following files to your project directory:
- `pseudo_labeling.py` - Main pipeline class
- `pseudo_labeling_notebook.ipynb` - Jupyter notebook interface

### 3. OneDL Setup

Ensure you have:
- OneDL account and project access
- Datasets uploaded to OneDL platform
- Proper authentication configured

### 4. CVAT Setup (Optional - for Manual Corrections)

If you plan to use manual corrections:
1. **CVAT Account**: Create account at your CVAT instance
2. **Access Credentials**: Obtain username and password
3. **Network Access**: Ensure connectivity to CVAT server

## Quick Start

### Step 1: Configure Global Settings

Open the Jupyter notebook and modify the **Global Initializers** section with your project details:

```python
pipeline = PseudoLabelingPipeline(
    project_name="your-project-name/your-subproject",           # Your OneDL project
    main_dataset_name="unlabeled-dataset:0",                   # Full unlabeled dataset
    initial_annotated_dataset_name="initial-annotations:0",    # Small annotated dataset
    validation_dataset="validation-set:0",                     # Validation dataset
    sample_size_per_iter=150,                                  # Images per iteration
    current_flow=0,                                            # Flow number (start with 0)
    min_confidence=0.5,                                        # Confidence threshold
    local_path='/path/to/your/local/storage',                  # Local storage path
    cvat_project_id=88,                                        # CVAT project ID (optional)
    db_path="pseudo_labeling_metadata.db"                      # Database file path (No need to change)
)
```

### Step 2: Set Training Configuration

Configure your model training parameters:

```python
train_cfg = pipeline.setup_training_config()
```

This opens an interactive widget where you can select:
- **Model Type**: FasterRCNN (Object Detection) or MaskRCNN (Instance Segmentation)
- **Backbone**: Neural network architecture
- **Training Parameters**: Epochs, batch size, etc.

### Step 3: Follow Notebook Steps to Train Initial Model (Section 1)
- **This includes steps**
  - 1.1 Training and Evaluation
  - 1.2 Logging for Initial Metadata

### Step 4: Choose Correction Strategy (Section 2)

After training initial model, for each iteration, decide your correction strategy:

```python
# For manual corrections via CVAT
manual_corrections = True

# For fully automated pseudo-labeling
manual_corrections = False

pipeline.setup_next_iteration(manual_corrections)
```

### Step 4: Run the Pipeline

Follow the notebook cells to execute each step of the pipeline automatically.

## Key Concepts

### Flows vs Iterations

- **Flow**: A complete pseudo-labeling experiment with specific configuration
  - Example: Flow 0 might use FasterRCNN with manual corrections and 50 initial annotated images.
  - Example: Flow 1 might use MaskRCNN with automated labeling and 150 initial annotated images.
  
- **Iteration**: Individual training cycles within a flow
  - Iteration 0: Train on initial annotated data only
  - Iteration 1+: Add new pseudo-labeled data and retrain

### Data Accumulation Strategy

The pipeline implements intelligent data accumulation:
- **New Samples**: Each iteration samples fresh unlabeled images
- **Past Pseudo-Labels**: All previously used pseudo-labeled images are re-processed after each iteration.
- **Updated Predictions**: Newer, better models generate improved labels on old data
- **Progressive Growth**: Training set expands while improving label quality

### Database Schema

The SQLite database tracks comprehensive metadata:
- Flow and iteration information
- Dataset names and sizes
- Number of ground truths and pseudo labeled images
- Model and evaluation UIDs
- Training configurations
- Evaluation metrics
- Status tracking
- Timestamps and completion records

## Pipeline Workflow

### Phase 1: Initial Setup (Iteration 0)

```
Initial Annotated Data → Train Baseline Model → Evaluate → Log Results
```

1. **Load Initial Data**: Small manually annotated dataset
2. **Train Model**: Create baseline model (FasterRCNN/MaskRCNN)
3. **Evaluate**: Test performance on validation set
4. **Log**: Record iteration 0 metadata

### Phase 2: Pseudo-Labeling Iterations (1, 2, 3...)

```
Sample New Data → Combine with Past → Generate Predictions → Correct → Merge → Train → Evaluate → Log
```

1. **Sample Unseen Inputs**: Select new unlabeled images from main dataset
2. **Accumulate Past Data**: Combine new samples with all previously used pseudo-labeled images
3. **Run Inference**: Generate predictions using the latest trained model
4. **Handle Corrections**:
   - **Manual**: Export to CVAT for human review and correction
   - **Automated**: Apply confidence filtering and use predictions directly
5. **Merge Datasets**: Combine initial annotations with new pseudo-labels
6. **Train Updated Model**: Retrain on expanded dataset
7. **Evaluate Performance**: Test on validation set
8. **Log Results**: Record iteration metadata and metrics

### Correction Workflows

#### Manual Correction Workflow (CVAT)
```
Predictions → Export to CVAT → Human Review → Import Corrections → Merge
```

- Exports predictions in COCO format
- Creates CVAT project with appropriate labels
- Uploads images and annotations
- Allows human review and correction
- Downloads corrected annotations
- Integrates back into pipeline

#### Automated Workflow
```
Predictions → Confidence Filtering → Direct Merge
```

- Applies minimum confidence threshold
- Automatically accepts high-confidence predictions
- Faster iteration but potentially lower quality

## Configuration Guide

### Essential Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `project_name` | OneDL project identifier | `"team/detection-project"` |
| `main_dataset_name` | Full unlabeled dataset | `"unlabeled-images:latest"` |
| `initial_annotated_dataset_name` | Small annotated dataset | `"initial-labels:v1"` |
| `validation_dataset` | Validation set | `"validation:v1"` |
| `sample_size_per_iter` | Images per iteration | `100-300` |
| `current_flow` | Flow number | `0, 1, 2...` |
| `min_confidence` | Confidence threshold | `0.3-0.8` |

### Advanced Parameters

| Parameter | Description | Default                              |
|-----------|-------------|--------------------------------------|
| `local_path` | Local storage directory | Required if `manual_correction = True` |
| `cvat_project_id` | CVAT project ID | `None`                               |
| `db_path` | Database file path | `"pseudo_labeling_metadata.db"`      |

### Training Configuration Options

#### Model Types
- **FasterRCNN**: Object detection (bounding boxes)
- **MaskRCNN**: Instance segmentation (masks + boxes)

#### Backbone Options
- ResNet variants
- RegNet variants
- ConvNeXt variants
- Custom architectures

#### Training Parameters
- **Epochs**: Number of training epochs (typically 20-100)
- **Batch Size**: Training batch size (adjust based on GPU memory)
- **Learning Rate**: Automatically configured by OneDL
- **Data Augmentation**: Handled by model configuration

## Manual Correction Setup (CVAT)

### Prerequisites

1. **CVAT Instance**: Access to a CVAT server
   - VBTI Server URL: `https:cvat2.vbti.nl`

2. **Account Credentials**: Username and password for CVAT

3. **Project Setup**: CVAT project for your annotation work

### Workflow Steps

#### 1. Export to CVAT
```python
# Enable manual corrections
manual_corrections = True
pipeline.setup_next_iteration(manual_corrections)

# Run sampling and inference
pipeline.sample_unseen_inputs()
pipeline.run_inference()

# Export to CVAT
pipeline.manually_correct_cvat()
```

#### 2. CVAT Annotation Process
- Pipeline creates CVAT project with appropriate labels
- Uploads images and initial predictions
- Provides CVAT task URL for annotation
- Human annotators review and correct predictions

#### 3. Import Corrections
```python
# Import corrected annotations and merge
pipeline.merge_pseudo_labels()
```


## Database Management

### Database Structure

The pipeline uses SQLite to track all metadata:

```sql
CREATE TABLE iteration_metadata (
    flow_id TEXT,
    iteration INTEGER,
    status TEXT,
    num_gt_images INTEGER,
    num_pseudo_images INTEGER,
    total_train_size INTEGER,
    model_uid TEXT,
    evaluation_uid TEXT,
    -- ... additional fields
    PRIMARY KEY (flow_id, iteration)
)
```

# Status Tracking

## Pipeline Status Tracking

The pipeline tracks detailed status updates throughout the lifecycle of each iteration. This includes sampling, inference, manual correction, training, evaluation, and completion.



## All Possible Status States

### Starting Iteration and Sampling Flow States

| status              | description                                              |
|---------------------|----------------------------------------------------------|
| `INITIALIZED`       | iteration has been set up but no work started            |
| `SAMPLING`          | currently sampling new images                            |
| `SAMPLING_COMPLETE` | sampling finished successfully                           |
| `PRE_INFERENCE`     | ready for inference (when using an existing dataset)     |

### Inference States

| status               | description                                |
|----------------------|--------------------------------------------|
| `INFERENCE`          | running inference on images                |
| `INFERENCE_COMPLETE` | inference completed successfully           |

### CVAT States

| status         | description                                  |
|----------------|----------------------------------------------|
| `CVAT_EXPORT`  | exporting data to cvat for manual correction |
| `CVAT_PENDING` | waiting for manual corrections in cvat       |
| .....          | .......                                      |
| `CVAT_FAILED`  | cvat export failed                           |

### Merging States

| status           | description                               |
|------------------|-------------------------------------------|
| `MERGING`        | currently merging pseudo-labels           |
| `MERGE_COMPLETE` | merging completed successfully            |

### Training States

| status              | description                          |
|---------------------|--------------------------------------|
| `TRAINING`          | training job is running/waiting      |
| `TRAINING_COMPLETE` | training completed successfully      |
| `TRAINING_FAILED`   | training job failed                  |
| `TRAINING_CANCELLED`| training job was cancelled           |

### Evaluation States

| status                 | description                          |
|------------------------|--------------------------------------|
| `EVALUATING`           | evaluation job is running/waiting    |
| `EVALUATION_COMPLETE`  | evaluation completed successfully    |
| `EVALUATION_FAILED`    | evaluation job failed                |
| `EVALUATION_CANCELLED` | evaluation job was cancelled         |

### Final State 

| status      | description                |
|-------------|----------------------------|
| `COMPLETED` | entire iteration completed |

---

##  OneDL job states to database states

### Training 

| job state     | database status       |
|---------------|------------------------|
| `DONE`        | `TRAINING_COMPLETE`    |
| `FAILED`      | `TRAINING_FAILED`      |
| `CANCELLED`   | `TRAINING_CANCELLED`   |
| `RUNNING`     | `TRAINING`             |
| `WAITING`     | `TRAINING`             |
| `UNALLOCABLE` | `TRAINING`             |

### Evaluation 

| job state     | database status         |
|---------------|--------------------------|
| `DONE`        | `EVALUATION_COMPLETE`    |
| `FAILED`      | `EVALUATION_FAILED`      |
| `CANCELLED`   | `EVALUATION_CANCELLED`   |
| `RUNNING`     | `EVALUATING`             |
| `WAITING`     | `EVALUATING`             |
| `UNALLOCABLE` | `EVALUATING`             |

## Database Operations

#### View Pipeline Status
```python
pipeline.get_pipeline_status()
```

#### View All Flows
```python
pipeline.get_all_flows_summary()
```

#### Clear Database (Testing)
```python
# DANGER: Clears all data
pipeline.clear_database("DELETE ALL DATA")
```

### Backup and Recovery

**Important**: Always backup your database before major operations:

```bash
# Backup database
cp pseudo_labeling_metadata.db pseudo_labeling_metadata_backup.db

# Restore if needed
cp pseudo_labeling_metadata_backup.db pseudo_labeling_metadata.db
```

## Troubleshooting

### Common Issues

#### 1. Model UID Not Set
```
Error: Inference model UID not set for iteration X
```
**Solution**: Ensure previous iteration completed successfully or manually set:
```python
pipeline.set_inference_model_uid("your-model-uid")
```

#### 2. Dataset Not Found
```
Error: Could not load dataset 'dataset-name'
```
**Solution**: 
- Verify dataset exists in OneDL project
- Check dataset name spelling and version
- Ensure proper OneDL authentication

#### 3. CVAT Connection Issues
```
Error: CVAT authentication failed
```
**Solution**:
- Verify CVAT server URL
- Check username/password
- Ensure network connectivity
- Try different CVAT instance

#### 4. Insufficient Unlabeled Data
```
Error: Only X unseen images left, but Y requested
```
**Solution**:
- Reduce `sample_size_per_iter`
- Add more unlabeled images to main dataset
- Complete current flow and start new one

#### 5. Training Failures
```
Error: Training job failed
```
**Solution**:
- Check OneDL job logs
- Verify dataset format and labels
- Adjust training parameters (batch size, epochs)
- Ensure sufficient computational resources

### Performance Optimization

#### 1. Confidence Threshold
- Higher thresholds (0.7-0.9): Fewer but higher quality pseudo-labels (Recommended)
- Lower thresholds (0.3-0.5): More pseudo-labels but potentially noisier

#### 2. Sample Size Strategy
- Smaller iterations (50-100): More control, slower progress (Recommended)
- Larger iterations (200-500): Faster progress, less control

#### 3. Database 
- It is highly recommended to review the SQLite database to ensure all processes are functioning correctly and no issues have occurred.


## Advanced Usage and Notes

### Multiple Flow Management

Run different experimental configurations:

```python
# Flow 0: FasterRCNN with manual corrections
pipeline_flow0 = PseudoLabelingPipeline(
    current_flow=0,
    # ... other parameters
)

# Flow 1: MaskRCNN with automated labeling
pipeline_flow1 = PseudoLabelingPipeline(
    current_flow=1,
    # ... other parameters
)
```

### Custom Training Configurations

Advanced model configuration:

```python
# Custom training parameters
train_cfg = {
    'model_type': 'MaskRCNNConfig',
    'backbone': MaskRCNNBackbone.REGNETX_4GF,
    'epochs': 75,
    'batch_size': 8
    ('add as much as you want...')
}
pipeline.train_cfg = train_cfg
```

### Batch Processing (Automatic Iterations)

Process multiple iterations at once:

```python
for iteration in range(1, 6):  # Run 5 iterations
    pipeline.setup_next_iteration(manual_corrections=False)
    pipeline.sample_unseen_inputs()
    pipeline.run_inference()
    pipeline.merge_pseudo_labels()
    pipeline.train_model()
    pipeline.evaluate_model()
    pipeline.complete_iteration()
    print(f"Iteration {iteration} completed")
```
Please note that to run this loop, `manual_corrections` must be `False`


## Best Practices

### 1. Data Quality
- **Start Strong**: Ensure high-quality initial annotations
- **Consistent Standards**: Maintain annotation consistency throughout
- **Regular Review**: Review pseudo-labels for quality
- **Validation Monitoring**: Watch validation metrics 

### 2. Iteration Strategy
- **Conservative Start**: Begin with higher confidence thresholds
- **Gradual Expansion**: Increase sample sizes as model improves
- **Strategic Corrections**: Use manual corrections for challenging cases
- **Regular Evaluation**: Monitor performance trends across iterations

### 3. Resource Management
- **Storage Planning**: Make sure you keep an eye on disk usage for exporting data
- **Database Review**: Regularly review the database

### 4. Experimental Design
- **Control Groups**: Compare different flows/strategies
- **Baseline Comparison**: Compare against traditional annotation

## Changelog and Versioning

### Current Features
- Automated pseudo-labeling pipeline
- CVAT integration for manual corrections
- SQLite database for metadata tracking
- Support for FasterRCNN and MaskRCNN
- Flexible configuration system
- Status monitoring and logging
- Data accumulation across iterations

### Planned Enhancements
- Specifying CVAT organization project and folder
- Fixing CVAT Authentication

---

## Acknowledgments

- OneDL access required for dataset management and model training
- CVAT access required for annotation capabilities

---
