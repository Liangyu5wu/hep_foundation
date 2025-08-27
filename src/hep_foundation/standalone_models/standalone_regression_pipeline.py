"""
Standalone Regression Pipeline for HEP Foundation Project.

This module provides a complete pipeline for training and evaluating standalone DNN models
for regression tasks without requiring foundation model pretraining.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow as tf

from hep_foundation.config.logging_config import get_logger
from hep_foundation.data.dataset_manager import DatasetManager
from hep_foundation.standalone_models.standalone_config import (
    StandaloneEvaluationConfig,
    StandaloneTrainingConfig,
    load_standalone_config,
)
from hep_foundation.standalone_models.standalone_dnn_regressor import (
    StandaloneDNNConfig,
    StandaloneDNNRegressor,
)
from hep_foundation.standalone_models.standalone_plot_manager import (
    StandalonePlotManager,
)
from hep_foundation.standalone_models.standalone_trainer import StandaloneTrainer


class StandaloneRegressionPipeline:
    """
    Complete pipeline for standalone DNN regression experiments.

    Handles data loading, model creation, training, evaluation, and visualization
    for standalone regression tasks without foundation model dependencies.
    """

    def __init__(
        self,
        processed_datasets_dir: Path = Path("_processed_datasets"),
        experiments_output_dir: Path = Path("_standalone_experiments"),
    ):
        """
        Initialize StandaloneRegressionPipeline.

        Args:
            processed_datasets_dir: Directory for processed datasets
            experiments_output_dir: Directory for experiment outputs
        """
        self.logger = get_logger(__name__)
        self.processed_datasets_dir = processed_datasets_dir
        self.experiments_output_dir = experiments_output_dir
        self.plot_manager = StandalonePlotManager()
        self.norm_params = None  # Store normalization parameters

        # Create output directory
        self.experiments_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("StandaloneRegressionPipeline initialized")
        self.logger.info(f"Processed datasets directory: {self.processed_datasets_dir}")
        self.logger.info(f"Experiments output directory: {self.experiments_output_dir}")

    def _make_json_serializable(self, obj):
        """Convert numpy/tensorflow objects to JSON serializable types."""
        import numpy as np
        import tensorflow as tf

        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tf.TensorShape):
            return obj.as_list()
        elif hasattr(obj, "numpy"):  # TensorFlow tensor
            return obj.numpy().tolist()
        else:
            return obj

    def _load_normalization_params(self, dataset_manager: DatasetManager) -> dict:
        """Load normalization parameters from dataset."""
        try:
            dataset_path = dataset_manager.get_current_dataset_path()
            with h5py.File(dataset_path, "r") as f:
                norm_params = json.loads(f.attrs["normalization_params"])
                self.logger.info("Successfully loaded normalization parameters")
                return norm_params
        except Exception as e:
            self.logger.warning(f"Failed to load normalization parameters: {e}")
            return None

    def _denormalize_labels(
        self,
        normalized_labels: np.ndarray,
        norm_params: dict,
        label_config_index: int = 0
    ) -> np.ndarray:
        """
        Denormalize labels using stored normalization parameters.
        
        Args:
            normalized_labels: Array of normalized label values
            norm_params: Normalization parameters dictionary
            label_config_index: Index of the label configuration to use
            
        Returns:
            Denormalized labels array
        """
        if norm_params is None:
            self.logger.warning("No normalization parameters available, returning original labels")
            return normalized_labels
            
        try:
            label_norm_params = norm_params["labels"][label_config_index]
            denormalized = normalized_labels.copy()
            
            # Handle aggregated features (arrays)
            if "aggregated" in label_norm_params:
                for agg_name, params in label_norm_params["aggregated"].items():
                    means = np.array(params["means"])
                    stds = np.array(params["stds"])
                    
                    # Apply denormalization: denormalized = normalized * std + mean
                    denormalized = denormalized * stds + means
                    break  # Assuming single aggregator for labels
                    
            # Handle scalar features
            elif "scalar" in label_norm_params:
                for scalar_name, params in label_norm_params["scalar"].items():
                    mean = params["mean"] 
                    std = params["std"]
                    # Apply denormalization for scalar features
                    denormalized = denormalized * std + mean
                    break  # Assuming single scalar feature for labels
                    
            return denormalized
            
        except Exception as e:
            self.logger.warning(f"Failed to denormalize labels: {e}, using normalized values")
            return normalized_labels

    def _average_training_histories(self, fold_histories: list[dict]) -> dict:
        """
        Average training histories from multiple k-fold runs.
        
        Args:
            fold_histories: List of training history dictionaries from different folds
            
        Returns:
            Averaged training history dictionary
        """
        if not fold_histories:
            return {}
            
        # Find common metrics across all folds
        common_metrics = set(fold_histories[0].keys())
        for history in fold_histories[1:]:
            common_metrics = common_metrics.intersection(set(history.keys()))
        
        averaged_history = {}
        
        for metric in common_metrics:
            # Get all metric values for this metric across folds
            metric_values = []
            min_length = float('inf')
            
            # Find minimum length across all folds for this metric
            for history in fold_histories:
                if metric in history and history[metric]:
                    min_length = min(min_length, len(history[metric]))
            
            if min_length == float('inf') or min_length == 0:
                continue
                
            # Collect values for averaging (truncate to min_length)
            for history in fold_histories:
                if metric in history and history[metric]:
                    metric_values.append(history[metric][:min_length])
            
            if metric_values:
                # Calculate mean across folds for each epoch
                averaged_values = []
                for epoch_idx in range(min_length):
                    epoch_values = [fold_values[epoch_idx] for fold_values in metric_values if epoch_idx < len(fold_values)]
                    if epoch_values:
                        averaged_values.append(np.mean(epoch_values))
                
                if averaged_values:
                    averaged_history[metric] = averaged_values
        
        return averaged_history

    def run_complete_pipeline(
        self,
        config_dict: dict[str, Any],
        delete_catalogs: bool = False,
    ) -> bool:
        """
        Run the complete standalone regression pipeline.

        Args:
            config_dict: Configuration dictionary from YAML
            delete_catalogs: Whether to delete data catalogs after processing

        Returns:
            True if pipeline succeeded, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING STANDALONE REGRESSION PIPELINE")
        self.logger.info("=" * 80)

        try:
            # Load configuration
            config = load_standalone_config(config_dict)
            training_config = config["training_config"]
            evaluation_config = config["evaluation_config"]
            model_config_dict = config["model_config_dict"]

            # Create experiment directory
            experiment_id = self._generate_experiment_id(config["metadata"]["name"])
            experiment_dir = self.experiments_output_dir / experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Experiment ID: {experiment_id}")
            self.logger.info(f"Experiment directory: {experiment_dir}")

            # Save configuration for reproducibility
            self._save_experiment_config(experiment_dir, config_dict, config)

            # Step 1: Load or create dataset
            dataset_manager = DatasetManager(base_dir=self.processed_datasets_dir)
            dataset_config, task_config = self._extract_dataset_configs(config)

            train_dataset, val_dataset, test_dataset = self._load_datasets(
                dataset_manager, dataset_config, task_config, delete_catalogs
            )

            # Load normalization parameters for denormalization
            self.norm_params = self._load_normalization_params(dataset_manager)

            # Step 2: Create and configure model
            model = self._create_model(model_config_dict, train_dataset)

            # Step 3: Run regression evaluation across different data sizes
            success = self._run_regression_evaluation(
                model,
                training_config,
                evaluation_config,
                train_dataset,
                val_dataset,
                test_dataset,
                experiment_dir,
            )

            if success:
                self.logger.info("=" * 80)
                self.logger.info(
                    "STANDALONE REGRESSION PIPELINE COMPLETED SUCCESSFULLY"
                )
                self.logger.info("=" * 80)
                self.logger.info(f"Results saved to: {experiment_dir}")
            else:
                self.logger.error("STANDALONE REGRESSION PIPELINE FAILED")

            return success

        except Exception as e:
            self.logger.error(f"Pipeline failed: {type(e).__name__}: {str(e)}")
            self.logger.exception("Detailed traceback:")
            return False

    def _generate_experiment_id(self, experiment_name: str) -> str:
        """Generate unique experiment ID."""
        # Find next available number
        existing_dirs = [d for d in self.experiments_output_dir.iterdir() if d.is_dir()]
        existing_numbers = []

        for d in existing_dirs:
            parts = d.name.split("_")
            if parts and parts[0].isdigit():
                existing_numbers.append(int(parts[0]))

        next_number = max(existing_numbers, default=0) + 1
        return f"{next_number:03d}_Standalone_{experiment_name.replace(' ', '_')}"

    def _save_experiment_config(
        self,
        experiment_dir: Path,
        original_config: dict[str, Any],
        processed_config: dict[str, Any],
    ) -> None:
        """Save experiment configuration for reproducibility."""
        try:
            # Save original config
            config_path = experiment_dir / "_experiment_config.yaml"
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(original_config, f, default_flow_style=False, indent=2)

            # Save experiment info
            info_path = experiment_dir / "_experiment_info.json"
            experiment_info = {
                "experiment_type": "standalone_regression",
                "timestamp": str(datetime.now()),
                "metadata": processed_config["metadata"],
                "training_config": processed_config["training_config"].to_dict(),
                "evaluation_config": processed_config["evaluation_config"].to_dict(),
                "model_config": processed_config["model_config_dict"],
            }

            with open(info_path, "w") as f:
                json.dump(self._make_json_serializable(experiment_info), f, indent=2)

            self.logger.info(f"Experiment configuration saved to: {config_path}")
            self.logger.info(f"Experiment info saved to: {info_path}")

        except Exception as e:
            self.logger.error(f"Failed to save experiment configuration: {e}")

    def _extract_dataset_configs(self, config: dict[str, Any]) -> tuple[Any, Any]:
        """Extract dataset and task configurations."""
        # Import here to avoid circular imports
        from hep_foundation.config.dataset_config import DatasetConfig
        from hep_foundation.config.task_config import TaskConfig

        try:
            # Create task config object FIRST
            task_dict = config["task_settings"]
            task_config = TaskConfig.create_from_branch_names(
                event_filter_dict=task_dict.get("event_filters", {}),
                input_features=task_dict.get("input_features", []),
                input_array_aggregators=task_dict.get("input_array_aggregators", []),
                label_features=task_dict.get("label_features", []),
                label_array_aggregators=task_dict.get("label_array_aggregators", []),
            )

            # Create dataset config object with task_config
            dataset_dict = config["dataset_settings"]
            dataset_config = DatasetConfig(
                run_numbers=dataset_dict["run_numbers"],
                signal_keys=dataset_dict["signal_keys"],
                catalog_limit=dataset_dict.get("catalog_limit", 10),
                validation_fraction=dataset_dict.get("validation_fraction", 0.15),
                test_fraction=dataset_dict.get("test_fraction", 0.15),
                shuffle_buffer=dataset_dict.get("shuffle_buffer", 10000),
                plot_distributions=dataset_dict.get("plot_distributions", True),
                include_labels=dataset_dict.get("include_labels", True),
                task_config=task_config,
            )

            # Validate configs
            dataset_config.validate()

            self.logger.info("Dataset and task configurations created successfully")
            return dataset_config, task_config

        except Exception as e:
            self.logger.error(f"Failed to create dataset/task configs: {e}")
            raise

    def _load_datasets(
        self,
        dataset_manager: DatasetManager,
        dataset_config: Any,
        task_config: Any,
        delete_catalogs: bool,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load or create datasets."""
        self.logger.info("Loading datasets...")

        train_dataset, val_dataset, test_dataset = dataset_manager.load_atlas_datasets(
            dataset_config=dataset_config,
            validation_fraction=dataset_config.validation_fraction,
            test_fraction=dataset_config.test_fraction,
            batch_size=1024,  # Will be rebatched during training
            shuffle_buffer=dataset_config.shuffle_buffer,
            include_labels=True,
            delete_catalogs=delete_catalogs,
        )

        self.logger.info("Datasets loaded successfully")
        return train_dataset, val_dataset, test_dataset

    def _create_model(
        self,
        model_config_dict: dict[str, Any],
        train_dataset: tf.data.Dataset,
    ) -> StandaloneDNNRegressor:
        """Create and build standalone DNN model."""
        self.logger.info("Creating standalone DNN model...")

        # Determine input and output shapes from dataset
        for batch in train_dataset:
            if isinstance(batch, tuple):
                features, labels = batch
                input_shape = features.shape[1:]
                if isinstance(labels, (list, tuple)):
                    # Multiple label sets - use the first one
                    output_shape = labels[0].shape[1:]
                else:
                    output_shape = labels.shape[1:]
                break
        else:
            raise ValueError("Unable to determine input/output shapes from dataset")

        self.logger.info(f"Determined input shape: {input_shape}")
        self.logger.info(f"Determined output shape: {output_shape}")

        # Create model config
        model_config = StandaloneDNNConfig(
            model_type="standalone_dnn_regressor",
            architecture={
                "input_shape": input_shape,
                "output_shape": output_shape,
                "hidden_layers": model_config_dict["architecture"]["hidden_layers"],
                "activation": model_config_dict["architecture"].get(
                    "activation", "relu"
                ),
                "output_activation": model_config_dict["architecture"].get(
                    "output_activation", "linear"
                ),
                "name": model_config_dict["architecture"].get("name", "standalone_dnn"),
            },
            hyperparameters={
                "dropout_rate": model_config_dict["hyperparameters"].get(
                    "dropout_rate", 0.0
                ),
                "l2_regularization": model_config_dict["hyperparameters"].get(
                    "l2_regularization", 0.0
                ),
                "batch_normalization": model_config_dict["hyperparameters"].get(
                    "batch_normalization", False
                ),
            },
        )

        # Create and build model
        model = StandaloneDNNRegressor(config=model_config)
        model.build(input_shape)

        self.logger.info("Standalone DNN model created successfully")
        self.logger.info(f"Total parameters: {model.model.count_params():,}")

        return model

    def _run_regression_evaluation(
        self,
        model: StandaloneDNNRegressor,
        training_config: StandaloneTrainingConfig,
        evaluation_config: StandaloneEvaluationConfig,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        experiment_dir: Path,
    ) -> bool:
        """Run two-stage regression evaluation: main model + data efficiency study."""
        self.logger.info("Starting two-stage regression evaluation...")

        # Create directories
        eval_dir = experiment_dir / "testing" / "regression_evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        main_model_dir = experiment_dir / "models" / "main_model"
        main_model_dir.mkdir(parents=True, exist_ok=True)

        # Count total training events
        total_train_events = sum(1 for _ in train_dataset.unbatch())
        self.logger.info(f"Total training events available: {total_train_events}")

        # Stage 1: Train main model with full data
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: TRAINING MAIN MODEL WITH FULL DATA")
        self.logger.info("=" * 60)

        main_model_success, main_results, main_predictions = self._train_main_model(
            model,
            training_config,
            train_dataset,
            val_dataset,
            test_dataset,
            total_train_events,
            main_model_dir,
            eval_dir,
        )

        if not main_model_success:
            self.logger.error("Main model training failed")
            return False

        # Stage 2: Data efficiency study
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: DATA EFFICIENCY STUDY")
        self.logger.info("=" * 60)

        _efficiency_success, all_results, all_predictions = (
            self._run_data_efficiency_study(
                model,
                training_config,
                evaluation_config,
                train_dataset,
                val_dataset,
                test_dataset,
                total_train_events,
                eval_dir,
            )
        )

        # Create comprehensive plots (including main model)
        self._create_comprehensive_plots(
            main_results,
            main_predictions,
            all_results,
            all_predictions,
            eval_dir,
            evaluation_config,
            total_train_events,
        )

        # Save all results
        self._save_comprehensive_results(main_results, all_results, eval_dir)

        self.logger.info("Two-stage regression evaluation completed successfully")
        return True

    def _train_main_model(
        self,
        model: StandaloneDNNRegressor,
        training_config: StandaloneTrainingConfig,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        total_train_events: int,
        main_model_dir: Path,
        eval_dir: Path,
    ) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
        """Train main model with full training data."""
        try:
            # Create fresh model instance using the same config as the original model
            config_dict = model.get_config()
            fresh_config = StandaloneDNNConfig(
                model_type="standalone_dnn_regressor",
                architecture={
                    "input_shape": config_dict["input_shape"],
                    "output_shape": config_dict["output_shape"],
                    "hidden_layers": config_dict["hidden_layers"],
                    "activation": config_dict["activation"],
                    "output_activation": config_dict["output_activation"],
                    "name": config_dict["name"],
                },
                hyperparameters={
                    "dropout_rate": config_dict["dropout_rate"],
                    "l2_regularization": config_dict["l2_regularization"],
                    "batch_normalization": config_dict["batch_normalization"],
                },
            )
            main_model = StandaloneDNNRegressor(fresh_config)
            main_model.build(model.input_shape)

            # Create trainer for main model
            trainer = StandaloneTrainer(main_model.model, training_config)

            # Train with full dataset
            self.logger.info(f"Training main model with {total_train_events} events...")
            success = trainer.train(
                dataset=train_dataset,
                validation_data=val_dataset,
                training_history_dir=eval_dir / "training_histories",
                model_name="main_model",
                dataset_id=f"full_{total_train_events}_events",
                experiment_id="standalone_regression_main",
                save_individual_history=True,
            )

            if not success:
                return False, {}, {}

            # Save main model
            model_save_path = main_model_dir / "model_weights.h5"
            trainer.save_model(model_save_path)

            # Save model config
            config_path = main_model_dir / "model_config.json"
            with open(config_path, "w") as f:
                json.dump(
                    self._make_json_serializable(main_model.get_config()), f, indent=2
                )

            # Evaluate on test set
            test_results = trainer.evaluate(test_dataset)

            # Generate predictions for analysis
            predictions = trainer.predict(test_dataset)

            # Extract true labels from test dataset
            true_labels = []
            for batch in test_dataset:
                if isinstance(batch, tuple):
                    _, labels = batch
                    if isinstance(labels, (list, tuple)):
                        true_labels.append(labels[0].numpy())
                    else:
                        true_labels.append(labels.numpy())

            true_labels = np.concatenate(true_labels, axis=0)

            # Denormalize predictions and labels for analysis
            if self.norm_params is not None:
                self.logger.info("Denormalizing predictions and labels for main model analysis...")
                try:
                    predictions_denorm = self._denormalize_labels(predictions, self.norm_params)
                    true_labels_denorm = self._denormalize_labels(true_labels, self.norm_params)
                    self.logger.info("Successfully denormalized main model predictions and labels")
                except Exception as e:
                    self.logger.warning(f"Failed to denormalize main model data: {e}, using normalized values")
                    predictions_denorm = predictions
                    true_labels_denorm = true_labels
            else:
                predictions_denorm = predictions
                true_labels_denorm = true_labels

            # Prepare results - ensure test_metrics has consistent format
            test_metrics = test_results.copy()
            test_metrics.update({
                "test_loss": test_results.get("loss", 0),
                "test_mse": test_results.get("mse", 0),
            })
            
            results = {
                "model_type": "main_model",
                "data_size": total_train_events,
                "test_metrics": test_metrics,
                "final_training_metrics": test_results,  # Keep original for compatibility
                "training_history": trainer.get_training_history(),
                "total_parameters": main_model.model.count_params(),
            }

            predictions_dict = {
                "predictions": predictions_denorm,  # Use denormalized values for plotting
                "targets": true_labels_denorm,     # Use denormalized values for plotting
            }

            self.logger.info("Main model training completed successfully")
            return True, results, predictions_dict

        except Exception as e:
            self.logger.error(f"Main model training failed: {e}")
            return False, {}, {}

    def _run_data_efficiency_study(
        self,
        model: StandaloneDNNRegressor,
        training_config: StandaloneTrainingConfig,
        evaluation_config: StandaloneEvaluationConfig,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        total_train_events: int,
        eval_dir: Path,
    ) -> tuple[bool, dict[int, dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
        """Run data efficiency study across different training data sizes."""
        data_sizes = evaluation_config.regression_data_sizes
        k_folds = getattr(evaluation_config, 'k_fold', 3)  # Default 3-fold CV
        use_k_fold = getattr(evaluation_config, 'use_k_fold_cv', True)
        
        self.logger.info(f"Running data efficiency study for sizes: {data_sizes}")
        if use_k_fold:
            self.logger.info(f"Using {k_folds}-fold cross-validation")

        all_results = {}
        all_predictions = {}
        k_fold_results = {}

        for data_size in data_sizes:
            self.logger.info("=" * 50)
            self.logger.info(f"Training with {data_size} events")
            self.logger.info("=" * 50)

            if use_k_fold and data_size > 1000:  # Only use k-fold for larger datasets
                success, results, predictions, kfold_stats = self._train_and_evaluate_with_kfold(
                    model,
                    training_config,
                    evaluation_config,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    data_size,
                    k_folds,
                    eval_dir,
                )
                if success:
                    k_fold_results[data_size] = kfold_stats
            else:
                # Single run for small datasets
                success, results, predictions = self._train_and_evaluate_single_size(
                    model,
                    training_config,
                    evaluation_config,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    data_size,
                    eval_dir,
                )

            if success:
                all_results[data_size] = results
                all_predictions[data_size] = predictions
            else:
                self.logger.warning(f"Failed to train model with {data_size} events")

        # Save k-fold results for plotting
        if k_fold_results:
            kfold_results_file = eval_dir / "k_fold_statistics.json"
            with open(kfold_results_file, "w") as f:
                json.dump(self._make_json_serializable(k_fold_results), f, indent=2)
            self.logger.info(f"K-fold statistics saved to: {kfold_results_file}")

        self.logger.info("Data efficiency study completed")
        return True, all_results, all_predictions

    def _train_and_evaluate_single_size(
        self,
        model: StandaloneDNNRegressor,
        training_config: StandaloneTrainingConfig,
        evaluation_config: StandaloneEvaluationConfig,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        data_size: int,
        eval_dir: Path,
    ) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
        """Train and evaluate model for a single data size."""
        try:
            # Create fresh model with proper config
            config_dict = model.get_config()
            fresh_config = StandaloneDNNConfig(
                model_type="standalone_dnn_regressor",
                architecture={
                    "input_shape": config_dict["input_shape"],
                    "output_shape": config_dict["output_shape"],
                    "hidden_layers": config_dict["hidden_layers"],
                    "activation": config_dict["activation"],
                    "output_activation": config_dict["output_activation"],
                    "name": config_dict["name"],
                },
                hyperparameters={
                    "dropout_rate": config_dict["dropout_rate"],
                    "l2_regularization": config_dict["l2_regularization"],
                    "batch_normalization": config_dict["batch_normalization"],
                },
            )
            fresh_model = StandaloneDNNRegressor(fresh_config)
            fresh_model.build(model.input_shape)

            # Create trainer with fixed epochs
            eval_training_config = StandaloneTrainingConfig(
                batch_size=training_config.batch_size,
                learning_rate=training_config.learning_rate,
                epochs=evaluation_config.fixed_epochs,
                early_stopping_patience=evaluation_config.fixed_epochs
                + 1,  # Disable early stopping
                early_stopping_min_delta=0,
                plot_training=training_config.plot_training,
                gradient_clip_norm=training_config.gradient_clip_norm,
                lr_scheduler=training_config.lr_scheduler,
            )

            trainer = StandaloneTrainer(fresh_model.model, eval_training_config)

            # Create data subset
            train_subset = (
                train_dataset.unbatch()
                .take(data_size)
                .batch(training_config.batch_size)
            )

            # Train model
            model_name = f"standalone_dnn_{data_size}"
            success = trainer.train(
                dataset=train_subset,
                validation_data=val_dataset,
                training_history_dir=eval_dir / "training_histories",
                model_name=model_name,
                dataset_id=f"{data_size}_events",
                experiment_id="standalone_regression",
                save_individual_history=True,
            )

            if not success:
                return False, {}, {}

            # Evaluate on test set
            test_results = trainer.evaluate(test_dataset)

            # Generate predictions for analysis
            predictions = trainer.predict(test_dataset)

            # Extract true labels from test dataset
            true_labels = []
            for batch in test_dataset:
                if isinstance(batch, tuple):
                    _, labels = batch
                    if isinstance(labels, (list, tuple)):
                        true_labels.append(labels[0].numpy())
                    else:
                        true_labels.append(labels.numpy())

            true_labels = np.concatenate(true_labels, axis=0)

            # Denormalize predictions and labels for analysis
            if self.norm_params is not None:
                try:
                    predictions_denorm = self._denormalize_labels(predictions, self.norm_params)
                    true_labels_denorm = self._denormalize_labels(true_labels, self.norm_params)
                except Exception as e:
                    self.logger.warning(f"Failed to denormalize data size {data_size}: {e}, using normalized values")
                    predictions_denorm = predictions
                    true_labels_denorm = true_labels
            else:
                predictions_denorm = predictions
                true_labels_denorm = true_labels

            # Prepare results - ensure test_metrics has consistent format
            test_metrics = test_results.copy()
            test_metrics.update({
                "test_loss": test_results.get("loss", 0),
                "test_mse": test_results.get("mse", 0),
            })
            
            results = {
                "data_size": data_size,
                "test_metrics": test_metrics,
                "final_training_metrics": test_results,  # Keep original for compatibility
                "training_history": trainer.get_training_history(),
            }

            predictions_dict = {
                "predictions": predictions_denorm,  # Use denormalized values for plotting
                "targets": true_labels_denorm,     # Use denormalized values for plotting
            }

            return True, results, predictions_dict

        except Exception as e:
            self.logger.error(f"Training failed for data size {data_size}: {e}")
            return False, {}, {}

    def _train_and_evaluate_with_kfold(
        self,
        model: StandaloneDNNRegressor,
        training_config: StandaloneTrainingConfig,
        evaluation_config: StandaloneEvaluationConfig,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        data_size: int,
        k_folds: int,
        eval_dir: Path,
    ) -> tuple[bool, dict[str, Any], dict[str, np.ndarray], dict[str, list[float]]]:
        """Train and evaluate model using k-fold cross-validation."""
        try:
            self.logger.info(f"Starting {k_folds}-fold cross-validation for {data_size} events")
            
            # Convert train dataset to list of examples for shuffling and splitting
            train_examples = []
            for batch in train_dataset.unbatch().take(data_size):
                train_examples.append(batch)
            
            if len(train_examples) < data_size:
                self.logger.warning(f"Only {len(train_examples)} events available, less than requested {data_size}")
                data_size = len(train_examples)
            
            # Shuffle examples
            np.random.shuffle(train_examples)
            
            # Split into k folds
            fold_size = data_size // k_folds
            fold_results = []
            fold_predictions_list = []
            
            for fold in range(k_folds):
                self.logger.info(f"Training fold {fold + 1}/{k_folds}")
                
                # Split data for this fold
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < k_folds - 1 else data_size
                
                # Create validation set for this fold
                fold_val_examples = train_examples[start_idx:end_idx]
                
                # Create training set (all other folds)
                fold_train_examples = train_examples[:start_idx] + train_examples[end_idx:]
                
                # Convert back to datasets
                def create_dataset_from_examples(examples, batch_size):
                    if not examples:
                        return None
                    
                    # Extract features and labels
                    features = []
                    labels = []
                    
                    for example in examples:
                        if isinstance(example, tuple):
                            feat, lab = example
                            features.append(feat)
                            labels.append(lab)
                        else:
                            features.append(example)
                    
                    # Create dataset
                    if labels:
                        dataset = tf.data.Dataset.from_tensor_slices((
                            tf.stack(features), 
                            tf.stack(labels)
                        ))
                    else:
                        dataset = tf.data.Dataset.from_tensor_slices(tf.stack(features))
                    
                    return dataset.batch(batch_size)
                
                fold_train_ds = create_dataset_from_examples(fold_train_examples, training_config.batch_size)
                fold_val_ds = create_dataset_from_examples(fold_val_examples, training_config.batch_size)
                
                # Create fresh model for this fold
                config_dict = model.get_config()
                fresh_config = StandaloneDNNConfig(
                    model_type="standalone_dnn_regressor",
                    architecture={
                        "input_shape": config_dict["input_shape"],
                        "output_shape": config_dict["output_shape"],
                        "hidden_layers": config_dict["hidden_layers"],
                        "activation": config_dict["activation"],
                        "output_activation": config_dict["output_activation"],
                        "name": config_dict["name"],
                    },
                    hyperparameters={
                        "dropout_rate": config_dict["dropout_rate"],
                        "l2_regularization": config_dict["l2_regularization"],
                        "batch_normalization": config_dict["batch_normalization"],
                    },
                )
                fold_model = StandaloneDNNRegressor(fresh_config)
                fold_model.build(model.input_shape)
                
                # Create trainer with fixed epochs
                eval_training_config = StandaloneTrainingConfig(
                    batch_size=training_config.batch_size,
                    learning_rate=training_config.learning_rate,
                    epochs=evaluation_config.fixed_epochs,
                    early_stopping_patience=evaluation_config.fixed_epochs + 1,
                    early_stopping_min_delta=0,
                    plot_training=False,  # Don't create plots for each fold
                    gradient_clip_norm=training_config.gradient_clip_norm,
                    lr_scheduler=training_config.lr_scheduler,
                )
                
                trainer = StandaloneTrainer(fold_model.model, eval_training_config)
                
                # Train this fold
                fold_success = trainer.train(
                    dataset=fold_train_ds,
                    validation_data=fold_val_ds,
                    training_history_dir=eval_dir / "kfold_histories",
                    model_name=f"fold_{fold+1}_of_{k_folds}_size_{data_size}",
                    dataset_id=f"{data_size}_events_fold_{fold+1}",
                    experiment_id="kfold_cv",
                    save_individual_history=False,  # Don't save individual fold histories
                )
                
                if not fold_success:
                    self.logger.warning(f"Fold {fold+1} training failed")
                    continue
                
                # Evaluate this fold on test set
                fold_test_results = trainer.evaluate(test_dataset)
                fold_predictions = trainer.predict(test_dataset)
                
                # Store training history for averaging
                fold_test_results['training_history'] = trainer.get_training_history()
                fold_results.append(fold_test_results)
                fold_predictions_list.append(fold_predictions)
                
                test_loss = fold_test_results.get('test_loss', fold_test_results.get('test_mse', 0.0))
                if isinstance(test_loss, (int, float)):
                    self.logger.info(f"Fold {fold+1} completed - Test Loss: {test_loss:.4f}")
                else:
                    self.logger.info(f"Fold {fold+1} completed - Test Loss: {test_loss}")
            
            if not fold_results:
                self.logger.error("All folds failed")
                return False, {}, {}, {}
            
            # Calculate k-fold statistics
            kfold_stats = self._calculate_kfold_statistics(fold_results)
            test_loss_mean = kfold_stats.get('test_loss_mean', 0.0)
            test_loss_std = kfold_stats.get('test_loss_std', 0.0)
            if isinstance(test_loss_mean, (int, float)) and isinstance(test_loss_std, (int, float)):
                self.logger.info(f"K-fold CV completed - Mean Test Loss: {test_loss_mean:.4f} ± {test_loss_std:.4f}")
            else:
                self.logger.info(f"K-fold CV completed - Mean Test Loss: {test_loss_mean} ± {test_loss_std}")
            
            # Extract true labels from test dataset (same for all folds)
            true_labels = []
            for batch in test_dataset:
                if isinstance(batch, tuple):
                    _, labels = batch
                    if isinstance(labels, (list, tuple)):
                        true_labels.append(labels[0].numpy())
                    else:
                        true_labels.append(labels.numpy())
            true_labels = np.concatenate(true_labels, axis=0)
            
            # Use the best fold's predictions (lowest test loss)
            best_fold_idx = np.argmin([result.get('test_loss', result.get('test_mse', float('inf'))) for result in fold_results])
            best_predictions = fold_predictions_list[best_fold_idx]

            # Denormalize predictions and labels for analysis
            if self.norm_params is not None:
                try:
                    best_predictions_denorm = self._denormalize_labels(best_predictions, self.norm_params)
                    true_labels_denorm = self._denormalize_labels(true_labels, self.norm_params)
                except Exception as e:
                    self.logger.warning(f"Failed to denormalize k-fold data size {data_size}: {e}, using normalized values")
                    best_predictions_denorm = best_predictions
                    true_labels_denorm = true_labels
            else:
                best_predictions_denorm = best_predictions
                true_labels_denorm = true_labels
            
            # Prepare results using k-fold averages
            results = {
                "data_size": data_size,
                "test_metrics": {
                    "loss": kfold_stats.get('loss_mean', kfold_stats.get('test_loss_mean', 0)),
                    "test_loss": kfold_stats.get('loss_mean', kfold_stats.get('test_loss_mean', 0)),
                    "mse": kfold_stats.get('mse_mean', kfold_stats.get('test_mse_mean', 0)),
                    "test_mse": kfold_stats.get('mse_mean', kfold_stats.get('test_mse_mean', 0)),
                    "mae": kfold_stats.get('mae_mean', 0),
                    "r2": kfold_stats.get('r2_mean', 0),
                    "correlation": kfold_stats.get('correlation_mean', 0),
                },
                "kfold_statistics": kfold_stats,
                "n_successful_folds": len(fold_results),
                "training_history": fold_results[best_fold_idx].get('training_history', {}),  # Use best fold's training history
            }
            
            predictions_dict = {
                "predictions": best_predictions_denorm,  # Use denormalized values for plotting
                "targets": true_labels_denorm,          # Use denormalized values for plotting
            }
            
            # Convert kfold_stats for returning  
            kfold_return_stats = {
                'test_loss': [r.get('loss', r.get('test_loss', 0)) for r in fold_results],
                'mae': [r.get('mae', 0) for r in fold_results],
                'mse': [r.get('mse', r.get('test_mse', 0)) for r in fold_results],
                'r2': [r.get('r2', 0) for r in fold_results],
                'correlation': [r.get('correlation', 0) for r in fold_results],
            }
            
            return True, results, predictions_dict, kfold_return_stats
            
        except Exception as e:
            self.logger.error(f"K-fold cross-validation failed for data size {data_size}: {e}")
            return False, {}, {}, {}

    def _calculate_kfold_statistics(self, fold_results: list[dict]) -> dict[str, float]:
        """Calculate statistics across k-fold results."""
        if not fold_results:
            return {}
        
        # Extract metrics from all folds
        metrics_lists = {}
        for result in fold_results:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    if key not in metrics_lists:
                        metrics_lists[key] = []
                    metrics_lists[key].append(value)
        
        # Calculate mean and std for each metric
        stats = {}
        for metric, values in metrics_lists.items():
            stats[f"{metric}_mean"] = np.mean(values)
            stats[f"{metric}_std"] = np.std(values)
            stats[f"{metric}_min"] = np.min(values)
            stats[f"{metric}_max"] = np.max(values)
        
        return stats

    def _create_comprehensive_plots(
        self,
        main_results: dict[str, Any],
        main_predictions: dict[str, np.ndarray],
        all_results: dict[int, dict[str, Any]],
        all_predictions: dict[int, dict[str, np.ndarray]],
        eval_dir: Path,
        evaluation_config: StandaloneEvaluationConfig,
        total_train_events: int,
    ) -> None:
        """Create comprehensive plots for standalone regression analysis."""
        try:
            plots_dir = eval_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # 1. Main model analysis (simplified: pred vs true + residuals histogram)
            if main_predictions and "predictions" in main_predictions:
                self.plot_manager.create_prediction_quality_plot(
                    main_predictions["predictions"],
                    main_predictions["targets"],
                    plots_dir / "main_model_analysis.png",
                    f"Main Model Analysis ({total_train_events} events)",
                    evaluation_config.prediction_sample_size,
                )

            # 2. Main model training history
            if "training_history" in main_results:
                self.plot_manager.create_training_history_plot(
                    main_results["training_history"],
                    plots_dir / "main_model_training_history.png",
                    f"Training History ({total_train_events} events)",
                )

            # 3. Training history summary for different data sizes - use k-fold averaged histories when available
            if all_results and len(all_results) > 1:
                histories_by_size = {}
                for data_size, results in all_results.items():
                    # Use single training history if available
                    # Note: K-fold averaging of training histories would require additional implementation
                    
                    # Fall back to single training history if no k-fold data
                    if "training_history" in results:
                        histories_by_size[data_size] = results["training_history"]
                
                if histories_by_size:
                    self.plot_manager.create_training_history_summary(
                        histories_by_size,
                        plots_dir / "training_history_summary.png",
                        "Training History Summary (K-Fold Averaged where available)",
                    )

            # 4. Metrics vs data size with k-fold error bars
            if all_results and len(all_results) > 1:
                # Prepare metrics data
                data_size_metrics = {}
                for data_size, results in all_results.items():
                    test_metrics = results.get("test_metrics", {})
                    if test_metrics:
                        data_size_metrics[data_size] = test_metrics
                
                # Add main model
                main_test_metrics = main_results.get("test_metrics", {})
                if main_test_metrics:
                    data_size_metrics[total_train_events] = main_test_metrics
                
                # Load k-fold results if available
                kfold_results_file = eval_dir / "k_fold_statistics.json"
                k_fold_data = None
                if kfold_results_file.exists():
                    try:
                        with open(kfold_results_file, 'r') as f:
                            k_fold_data = json.load(f)
                        # Convert string keys to int
                        if k_fold_data:
                            k_fold_data = {int(k): v for k, v in k_fold_data.items()}
                    except Exception as e:
                        self.logger.warning(f"Failed to load k-fold results: {e}")
                        k_fold_data = None
                
                if data_size_metrics:
                    self.plot_manager.create_metrics_vs_datasize_plot(
                        data_size_metrics,
                        k_fold_data,
                        plots_dir / "metrics_vs_datasize.png",
                        "Evaluation Metrics vs Data Size",
                    )

            # 5. Pred vs true subplots for different data sizes
            if (evaluation_config.create_detailed_plots and all_predictions and 
                len(all_predictions) > 0):
                
                self.plot_manager.create_pred_vs_true_subplots(
                    all_predictions,
                    plots_dir / "pred_vs_true_by_datasize.png",
                    "Predictions vs True Values by Data Size",
                    evaluation_config.prediction_sample_size,
                )

            # 6. Additional comprehensive plots for main model
            if main_predictions and "predictions" in main_predictions:
                # Relative error histogram for main model
                self.plot_manager.create_relative_error_histogram(
                    main_predictions["predictions"],
                    main_predictions["targets"],
                    plots_dir / "main_model_relative_errors.png",
                    f"Main Model Relative Errors ({total_train_events} events)",
                    n_bins=evaluation_config.error_analysis_bins,
                    max_relative_error=2.0,
                )

            # 7. Data efficiency comparison summary
            if all_results and len(all_results) > 1:
                # Extract metrics for summary
                data_size_metrics = {}
                for data_size, results in all_results.items():
                    test_metrics = results.get("test_metrics", {})
                    if test_metrics:
                        data_size_metrics[data_size] = test_metrics

                # Add main model
                main_test_metrics = main_results.get("test_metrics", {})
                if main_test_metrics:
                    data_size_metrics[total_train_events] = main_test_metrics

                # Load k-fold results if available for error bars
                kfold_results_file = eval_dir / "k_fold_statistics.json"
                k_fold_data = None
                if kfold_results_file.exists():
                    try:
                        with open(kfold_results_file, 'r') as f:
                            k_fold_data = json.load(f)
                        # Convert string keys to int
                        if k_fold_data:
                            k_fold_data = {int(k): v for k, v in k_fold_data.items()}
                    except Exception as e:
                        self.logger.warning(f"Failed to load k-fold results: {e}")
                        k_fold_data = None

                if data_size_metrics:
                    self.plot_manager.create_data_size_comparison_summary(
                        data_size_metrics,
                        plots_dir / "data_size_comparison_summary.png", 
                        "Comprehensive Data Size Effect Analysis",
                        k_fold_data,
                    )


        except Exception as e:
            self.logger.error(f"Failed to create plots: {e}")
            self.logger.exception("Detailed error:")

    def _save_comprehensive_results(
        self,
        main_results: dict[str, Any],
        all_results: dict[int, dict[str, Any]],
        eval_dir: Path,
    ) -> None:
        """Save comprehensive results including main model and efficiency study."""
        try:
            # Save main model results
            main_results_file = eval_dir / "main_model_results.json"
            main_serializable = {
                "model_type": main_results.get("model_type", "main_model"),
                "data_size": main_results.get("data_size", "unknown"),
                "test_metrics": main_results.get("test_metrics", {}),
                "total_parameters": main_results.get("total_parameters", 0),
                "final_training_metrics": {
                    metric: values[-1] if values else None
                    for metric, values in main_results.get(
                        "training_history", {}
                    ).items()
                },
            }

            with open(main_results_file, "w") as f:
                json.dump(self._make_json_serializable(main_serializable), f, indent=2)

            # Save efficiency study results
            if all_results:
                efficiency_results_file = eval_dir / "data_efficiency_results.json"
                efficiency_serializable = {}
                for data_size, results in all_results.items():
                    efficiency_serializable[str(data_size)] = {
                        "data_size": results["data_size"],
                        "test_metrics": results["test_metrics"],
                        "final_training_metrics": {
                            metric: values[-1] if values else None
                            for metric, values in results["training_history"].items()
                        },
                    }

                with open(efficiency_results_file, "w") as f:
                    json.dump(
                        self._make_json_serializable(efficiency_serializable),
                        f,
                        indent=2,
                    )

            # Save combined summary
            summary_file = eval_dir / "evaluation_summary.json"
            summary = {
                "experiment_type": "two_stage_standalone_regression",
                "main_model": main_serializable,
                "data_efficiency_study": efficiency_serializable if all_results else {},
                "summary_statistics": {
                    "main_model_performance": main_results.get("test_metrics", {}),
                    "efficiency_study_sizes": list(all_results.keys())
                    if all_results
                    else [],
                    "total_models_trained": 1 + len(all_results),
                },
            }

            with open(summary_file, "w") as f:
                json.dump(self._make_json_serializable(summary), f, indent=2)

            self.logger.info(f"Main model results saved to: {main_results_file}")
            if all_results:
                self.logger.info(
                    f"Efficiency study results saved to: {efficiency_results_file}"
                )
            self.logger.info(f"Evaluation summary saved to: {summary_file}")

        except Exception as e:
            self.logger.error(f"Failed to save comprehensive results: {e}")
