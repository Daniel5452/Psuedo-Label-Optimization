# --------- LOGGING FUNCTIONS (logging_functions.py) ---------
# Import the log_iteration_metadata function from db_functions
from scrap.db.db_functions import log_iteration_metadata


def log_iteration_0(cursor, conn, flow_id, model_uid, evaluation_uid, evaluation_info_str,
                    train_dataset_name, validation_dataset,
                    n_initial_samples, main_dataset_name, train_cfg):
    """
    Log iteration 0 to the database.
    Call this after training and evaluating the initial model.

    Args:
        cursor: Database cursor
        conn: Database connection
        flow_id (str): Flow identifier
        model_uid (str): Model UID from training
        evaluation_uid (str): Evaluation UID
        evaluation_info_str (str): Evaluation metrics as string
        initial_annotated_dataset_name (str): Initial annotated dataset
        validation_dataset (str): Validation dataset name
        n_initial_samples (int): Number of initial samples
        main_dataset_name (str): Main dataset name
        train_cfg (dict): Training configuration
    """

    log_iteration_metadata(
        cursor=cursor,
        flow_id=flow_id,
        iteration=0,
        num_gt_images=n_initial_samples,
        num_gt_images_added=n_initial_samples,
        num_pseudo_images=0,
        num_pseudo_images_added=0,
        total_train_size=n_initial_samples,
        train_dataset=train_dataset_name,
        pseudo_input_dataset_name="",
        pseudo_output_dataset_name="",
        inference_model_uid="",
        model_uid=model_uid,
        evaluation_uid=evaluation_uid,
        evaluation_info=evaluation_info_str,
        manual_correction=True,
        cvat_project_id=None,
        main_dataset=main_dataset_name,
        validation_dataset=validation_dataset,
        train_cfg=train_cfg
    )

    conn.commit()
    print(f"Iteration 0 logged for {flow_id}")


def logging(client, train_dataset_name, cursor, current_flow, current_iteration,
            num_gt_images_after_iter, num_gt_images_added, num_pseudo_images_after_iter,
            num_pseudo_images_added, pseudo_input_dataset_name, predicted_dataset_name,
            inference_model_uid, model_uid, evaluation_uid, evaluation_info_str,
            manual_corrections_global, main_dataset_name, validation_dataset, train_cfg, conn):
    """
    Log regular iterations to the database.
    Call this after running the full pipeline for iterations 1+.

    Args:
        client: OneDL client
        train_dataset_name (str): Training dataset name
        cursor: Database cursor
        current_flow (int): Current flow number
        current_iteration (int): Current iteration number
        num_gt_images_after_iter (int): GT images after this iteration
        num_gt_images_added (int): GT images added this iteration
        num_pseudo_images_after_iter (int): Pseudo images after this iteration
        num_pseudo_images_added (int): Pseudo images added this iteration
        pseudo_input_dataset_name (str): Pseudo input dataset name
        predicted_dataset_name (str): Predicted dataset name
        inference_model_uid (str): Inference model UID
        model_uid (str): Trained model UID
        evaluation_uid (str): Evaluation UID
        evaluation_info_str (str): Evaluation info
        manual_corrections_global (bool): Manual corrections flag
        main_dataset_name (str): Main dataset name
        validation_dataset (str): Validation dataset name
        train_cfg (dict): Training configuration
        conn: Database connection
    """

    dataset_init = client.datasets.load(train_dataset_name)
    log_iteration_metadata(
        cursor=cursor,
        flow_id=f"f{current_flow}",
        iteration=current_iteration,
        num_gt_images=num_gt_images_after_iter,
        num_gt_images_added=num_gt_images_added,
        num_pseudo_images=num_pseudo_images_after_iter,
        num_pseudo_images_added=num_pseudo_images_added,
        total_train_size=len(dataset_init),
        train_dataset=train_dataset_name,
        pseudo_input_dataset_name=pseudo_input_dataset_name,
        pseudo_output_dataset_name=predicted_dataset_name,
        inference_model_uid=inference_model_uid,
        model_uid=model_uid,
        evaluation_uid=evaluation_uid,
        evaluation_info=evaluation_info_str,
        manual_correction=manual_corrections_global,
        cvat_project_id=None,
        main_dataset=main_dataset_name,
        validation_dataset=validation_dataset,
        train_cfg=train_cfg
    )
    conn.commit()
    print("Metadata logged.")


