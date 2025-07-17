import sqlite3
import time


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
                inference_model_uid, manual_correction, train_cfg
            ) VALUES (?, ?, 'INITIALIZED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            kwargs['flow_id'], kwargs['iteration'],
            kwargs['num_gt_images'], kwargs['num_gt_images_added'],
            kwargs['num_pseudo_images'], kwargs['num_pseudo_images_added'],
            kwargs['total_train_size'],
            kwargs['main_dataset'], kwargs['validation_dataset'], kwargs['train_dataset'],
            kwargs['inference_model_uid'], kwargs['manual_correction'], train_cfg_str
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
        """Log iteration 0 to the database."""
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
                manual_correction, train_cfg,
                timestamp, completed_timestamp
            ) VALUES (?, ?, 'COMPLETED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            kwargs['flow_id'], kwargs['iteration'],
            kwargs['num_gt_images'], kwargs['num_gt_images_added'],
            kwargs['num_pseudo_images'], kwargs['num_pseudo_images_added'],
            kwargs['total_train_size'],
            kwargs['main_dataset'], kwargs['validation_dataset'], kwargs['train_dataset'],
            kwargs['pseudo_input_dataset_name'], kwargs['pseudo_output_dataset_name'],
            kwargs['inference_model_uid'], kwargs['model_uid'], kwargs['evaluation_uid'], kwargs['evaluation_info'],
            kwargs['manual_correction'], train_cfg_str,
            time.strftime('%Y-%m-%d %H:%M:%S'),  # timestamp
            time.strftime('%Y-%m-%d %H:%M:%S')  # completed_timestamp
        ))
        self.conn.commit()

    def log_iteration_0_external_model(self, **kwargs):
        """Log iteration 0 with external model to the database."""
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
                manual_correction, train_cfg,
                timestamp, completed_timestamp
            ) VALUES (?, ?, 'COMPLETED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            kwargs['flow_id'], kwargs['iteration'],
            kwargs['num_gt_images'], kwargs['num_gt_images_added'],
            kwargs['num_pseudo_images'], kwargs['num_pseudo_images_added'],
            kwargs['total_train_size'],
            kwargs['main_dataset'], kwargs['validation_dataset'], kwargs['train_dataset'],
            kwargs['pseudo_input_dataset_name'], kwargs['pseudo_output_dataset_name'],
            kwargs['inference_model_uid'], kwargs['model_uid'], kwargs['evaluation_uid'], kwargs['evaluation_info'],
            kwargs['manual_correction'], train_cfg_str,
            time.strftime('%Y-%m-%d %H:%M:%S'),  # timestamp
            time.strftime('%Y-%m-%d %H:%M:%S')  # completed_timestamp
        ))
        self.conn.commit()