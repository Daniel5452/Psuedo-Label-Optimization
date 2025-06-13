# --------- LOCAL INITIALIZERS (local_initializers.py) ---------

def setup_initial_flow(current_flow, client, cursor, initial_annotated_dataset_name):
    """
    Set up the initial flow by copying initial_annotated_dataset to train-f{current_flow}.
    Call this once after global initializers to prepare for iteration 0.

    Args:
        current_flow (int): Flow number
        client: OneDL client
        cursor: Database cursor
        initial_annotated_dataset_name (str): Initial dataset to copy from

    Returns:
        dict: Dictionary containing:
            - current_iteration: Set to 0
            - flow_id: Flow identifier
            - train_dataset_name: Training dataset for this flow
            - n_initial_samples: Number of samples in initial dataset
    """
    current_iteration = 0
    flow_id = f'f{current_flow}'
    train_dataset_name = f"train-{flow_id}"

    # Check if this flow already exists
    cursor.execute('''
        SELECT COUNT(*)
        FROM iteration_metadata
        WHERE flow_id = ?
    ''', (flow_id,))
    result = cursor.fetchone()

    if result[0] > 0:
        print(f"Flow {flow_id} already exists. Skipping setup.")
        return {
            'current_iteration': current_iteration,
            'flow_id': flow_id,
            'train_dataset_name': train_dataset_name,
            'n_initial_samples': 0  # We don't know the count if already exists
        }

    # Copy initial annotated dataset to new flow training dataset
    initial_dataset = client.datasets.load(initial_annotated_dataset_name)
    n_initial_samples = len(initial_dataset)
    # client.datasets.save(train_dataset_name, dataset=initial_dataset, exist="overwrite")
    # client.datasets.push(train_dataset_name)

    print(f"Flow {flow_id} initialized")
    print(f"Created training dataset: {train_dataset_name}")
    print(f"Copied from: {initial_annotated_dataset_name}")
    print(f"Initial samples: {n_initial_samples}")
    print(f"Ready for iteration {current_iteration}")

    return {
        'current_iteration': current_iteration,
        'flow_id': flow_id,
        'train_dataset_name': train_dataset_name,
        'n_initial_samples': n_initial_samples
    }


def setup_next_iteration(current_flow, manual_corrections, sample_size_per_iter, cursor):
    """
    Set up the next iteration for the current flow.
    Call this before running the pipeline for iterations 1, 2, 3, etc.

    Args:
        current_flow (int): Flow number
        manual_corrections (bool): Whether this iteration uses manual corrections
        sample_size_per_iter (int): Number of samples per iteration
        cursor: Database cursor

    Returns:
        dict: Dictionary containing:
            - current_iteration: Next iteration number
            - flow_id: Flow identifier
            - train_dataset_name: Training dataset name
            - pseudo_input_dataset_name: Pseudo input dataset name
            - inference_model_uid: Model from previous iteration
            - manual_corrections_global: Correction mode for this iteration
            - num_gt_images_after_iter: Total GT images after this iteration
            - num_gt_images_added: GT images added this iteration
            - num_pseudo_images_after_iter: Total pseudo images after this iteration
            - num_pseudo_images_added: Pseudo images added this iteration
    """
    flow_id = f"f{current_flow}"

    # Get the last iteration for the specified flow
    cursor.execute('''
        SELECT MAX(iteration)
        FROM iteration_metadata
        WHERE flow_id = ?
    ''', (flow_id,))
    result = cursor.fetchone()

    if result is None or result[0] is None:
        raise RuntimeError(f"No iteration history found for flow: {flow_id}")

    last_iteration = result[0]
    current_iteration = last_iteration + 1

    # Get previous model UID and image counts for this flow
    cursor.execute('''
        SELECT model_uid, num_gt_images, num_pseudo_images
        FROM iteration_metadata
        WHERE flow_id = ? AND iteration = ?
    ''', (flow_id, current_iteration - 1))

    result = cursor.fetchone()
    if result is None or result[0] is None:
        raise RuntimeError(f"No model metadata found for {flow_id} @ iteration {current_iteration - 1}")

    inference_model_uid, previous_gt_total, previous_pseudo_total = result

    # Define image additions based on correction mode
    if manual_corrections:
        num_gt_images_added = sample_size_per_iter
        num_pseudo_images_added = 0
    else:
        num_gt_images_added = 0
        num_pseudo_images_added = sample_size_per_iter

    num_gt_images_after_iter = previous_gt_total + num_gt_images_added
    num_pseudo_images_after_iter = previous_pseudo_total + num_pseudo_images_added
    total_train_size = num_gt_images_after_iter + num_pseudo_images_after_iter

    # Define dataset names
    train_dataset_name = f"train-{flow_id}"
    pseudo_input_dataset_name = f"pseudo-iter{current_iteration}-{flow_id}"

    # Output for debug
    correction_type = "Manual" if manual_corrections else "Pseudo"
    print(f"{flow_id} Iteration {current_iteration} ({correction_type})")
    print(f"Adding {sample_size_per_iter} {correction_type.lower()} samples")
    print(
        f"Total size after: {total_train_size} (GT: {num_gt_images_after_iter}, Pseudo: {num_pseudo_images_after_iter})")
    print(f"Using model: {inference_model_uid[:8]}...")

    return {
        'current_iteration': current_iteration,
        'flow_id': flow_id,
        'train_dataset_name': train_dataset_name,
        'pseudo_input_dataset_name': pseudo_input_dataset_name,
        'inference_model_uid': inference_model_uid,
        'manual_corrections_global': manual_corrections,
        'num_gt_images_after_iter': num_gt_images_after_iter,
        'num_gt_images_added': num_gt_images_added,
        'num_pseudo_images_after_iter': num_pseudo_images_after_iter,
        'num_pseudo_images_added': num_pseudo_images_added
    }


# Import the log_iteration_metadata function from db_functions
