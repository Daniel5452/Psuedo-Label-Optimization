import sqlite3
import json


def connect_db(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    return conn, cursor


def initialize_metadata_db(cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS iteration_metadata (
        flow_id TEXT,                  -- The current flow we are on
        iteration INTEGER,             -- Iteration that we are on within this flow
        num_gt_images INTEGER,         -- Cumulative GT images after this iteration
        num_gt_images_added INTEGER,   -- Number of GT images added in this iteration
        num_pseudo_images INTEGER,     -- Cumulative pseudo-labeled images after this iteration
        num_pseudo_images_added INTEGER, -- Number of pseudo-labeled images added this iteration
        total_train_size INTEGER,      -- Training set size at this flow/iteration
        main_dataset TEXT,             -- Name of the main dataset with "unlabeled images"
        validation_set TEXT,           -- Name of the validation dataset.
        train_dataset TEXT,            -- Name of the training set for this flow
        pseudo_input_dataset_name TEXT,     --Dataset name of the current iteration sample, this is because we create one each time. it will be called something like pseudo-iter1-f1
        pseudo_output_dataset_name TEXT,    --Dataset name of the previous iteration sample after being evaluated to get predictions. Will be called something like pseudo-iter1-f1-0-gpu-198408
        inference_model_uid TEXT,      -- Model used for inference (previous)
        model_uid TEXT,                -- Model trained at this iteration
        evaluation_uid TEXT,           -- Evaluation job ID
        evaluation_info TEXT,          -- Dictionary containing evaluation info for model trained at this iteration
        manual_correction BOOLEAN,     -- Whether manual correction was done in this iteration
        cvat_project_id INTEGER,          -- CVAT project ID used for annotation or review
        train_cfg TEXT,                -- Training configuration dictionary as string
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (flow_id, iteration)
    )
    ''')


def log_iteration_metadata(
        cursor,
        flow_id,
        iteration,
        num_gt_images,
        num_gt_images_added,
        num_pseudo_images,
        num_pseudo_images_added,
        total_train_size,
        train_dataset,
        pseudo_input_dataset_name,
        pseudo_output_dataset_name,
        inference_model_uid,
        model_uid,
        evaluation_uid,
        evaluation_info,
        manual_correction,
        cvat_project_id,
        main_dataset,
        validation_dataset,
        train_cfg=None
):
    # Convert train_cfg to string if provided (preserves the exact dictionary structure)
    train_cfg_str = str(train_cfg) if train_cfg is not None else None

    cursor.execute('''
        INSERT INTO iteration_metadata (
            flow_id, iteration,
            num_gt_images, num_gt_images_added,
            num_pseudo_images, num_pseudo_images_added,
            total_train_size,
            main_dataset, validation_set, train_dataset,
            pseudo_input_dataset_name, pseudo_output_dataset_name,
            inference_model_uid, model_uid, evaluation_uid, evaluation_info,
            manual_correction, cvat_project_id, train_cfg
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        flow_id,
        iteration,
        num_gt_images,
        num_gt_images_added,
        num_pseudo_images,
        num_pseudo_images_added,
        total_train_size,
        main_dataset,
        validation_dataset,
        train_dataset,
        pseudo_input_dataset_name,
        pseudo_output_dataset_name,
        inference_model_uid,
        model_uid,
        evaluation_uid,
        evaluation_info,
        manual_correction,
        cvat_project_id,
        train_cfg_str
    ))


def get_train_cfg_from_db(cursor, flow_id, iteration):
    """
    Retrieve training configuration from database for a specific flow/iteration

    Args:
        cursor: Database cursor
        flow_id (str): Flow identifier
        iteration (int): Iteration number

    Returns:
        dict: Training configuration dictionary (or None if not found)
    """
    cursor.execute('''
        SELECT train_cfg
        FROM iteration_metadata
        WHERE flow_id = ? AND iteration = ?
    ''', (flow_id, iteration))

    result = cursor.fetchone()
    if result and result[0]:
        # Convert string back to dictionary using eval (safe since we control the input)
        return eval(result[0])
    return None