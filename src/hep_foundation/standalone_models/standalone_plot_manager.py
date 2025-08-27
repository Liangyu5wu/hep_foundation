"""
Standalone Plot Manager for HEP Foundation Pipeline.

This module provides comprehensive plotting functionality for standalone DNN regression tasks,
including training history, prediction analysis, and error analysis visualizations.
"""

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from hep_foundation.config.logging_config import get_logger

# Import plot utilities (assuming they exist in the project)
try:
    from hep_foundation.plots.plot_utils import (
        FONT_SIZES,
        LINE_WIDTHS,
        MARKER_SIZES,
        get_color_cycle,
        get_figure_size,
        set_science_style,
    )
except ImportError:
    # Fallback constants if plot_utils not available
    FONT_SIZES = {
        "tiny": 8,
        "small": 10,
        "normal": 12,
        "large": 14,
        "xlarge": 16,
    }
    LINE_WIDTHS = {"thin": 1, "thick": 2, "very_thick": 3}
    MARKER_SIZES = {"tiny": 2, "small": 4, "normal": 6, "large": 8}

    def get_color_cycle(style="default", n_colors=8):
        return plt.cm.tab10(np.linspace(0, 1, n_colors))

    def get_figure_size(size_type="single", ratio=1.0):
        base_width = 8 if size_type == "single" else 12
        return (base_width, base_width * ratio)

    def set_science_style(use_tex=False):
        plt.style.use(
            "seaborn-v0_8"
            if hasattr(plt.style, "available") and "seaborn-v0_8" in plt.style.available
            else "default"
        )


class StandalonePlotManager:
    """
    Plot manager for standalone DNN regression experiments.

    Provides comprehensive visualization capabilities for training monitoring,
    prediction analysis, and error analysis specific to standalone regression tasks.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        set_science_style(use_tex=False)

    def create_training_history_plot(
        self,
        history_data: dict[str, list[float]],
        output_path: Path,
        title_prefix: str = "Standalone DNN Training",
    ) -> None:
        """
        Create comprehensive training history plot with loss, metrics, and learning rate.

        Args:
            history_data: Dictionary containing training history
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
        """
        try:
            # Extract data
            epochs = list(range(1, len(history_data.get("loss", [])) + 1))
            if not epochs:
                self.logger.warning("No training history data available")
                return

            # Determine number of subplots based on available data
            has_lr_data = "lr" in history_data
            n_subplots = 2 + (1 if has_lr_data else 0)

            # Create figure
            fig, axes = plt.subplots(
                n_subplots, 1, figsize=get_figure_size("single", ratio=1.2), sharex=True
            )
            if n_subplots == 1:
                axes = [axes]

            colors = get_color_cycle("high_contrast", 4)

            # Plot 1: Loss
            ax_idx = 0
            if "loss" in history_data:
                axes[ax_idx].plot(
                    epochs,
                    history_data["loss"],
                    color=colors[0],
                    linewidth=LINE_WIDTHS["thick"],
                    label="Training Loss",
                    linestyle="-",
                )
                if "val_loss" in history_data:
                    axes[ax_idx].plot(
                        epochs,
                        history_data["val_loss"],
                        color=colors[1],
                        linewidth=LINE_WIDTHS["thick"],
                        label="Validation Loss",
                        linestyle="--",
                    )
                axes[ax_idx].set_ylabel("Loss", fontsize=FONT_SIZES["normal"])
                axes[ax_idx].legend(fontsize=FONT_SIZES["small"])
                axes[ax_idx].grid(True, alpha=0.3)
                axes[ax_idx].set_yscale("log")
                ax_idx += 1

            # Plot 2: Metrics (MSE, MAE)
            metric_names = [
                key
                for key in history_data.keys()
                if key in ["mse", "mae"]
                or key.startswith("val_")
                and key[4:] in ["mse", "mae"]
            ]
            if metric_names:
                for i, metric in enumerate(["mse", "mae"]):
                    if metric in history_data:
                        axes[ax_idx].plot(
                            epochs,
                            history_data[metric],
                            color=colors[i % len(colors)],
                            linewidth=LINE_WIDTHS["thick"],
                            label=f"Training {metric.upper()}",
                            linestyle="-",
                        )
                    val_metric = f"val_{metric}"
                    if val_metric in history_data:
                        axes[ax_idx].plot(
                            epochs,
                            history_data[val_metric],
                            color=colors[i % len(colors)],
                            linewidth=LINE_WIDTHS["thick"],
                            label=f"Validation {metric.upper()}",
                            linestyle="--",
                        )

                axes[ax_idx].set_ylabel("Metrics", fontsize=FONT_SIZES["normal"])
                axes[ax_idx].legend(fontsize=FONT_SIZES["small"])
                axes[ax_idx].grid(True, alpha=0.3)
                axes[ax_idx].set_yscale("log")
                ax_idx += 1

            # Plot 3: Learning Rate (if available)
            if has_lr_data and "lr" in history_data:
                axes[ax_idx].plot(
                    epochs,
                    history_data["lr"],
                    color=colors[3],
                    linewidth=LINE_WIDTHS["thick"],
                    label="Learning Rate",
                    linestyle="-",
                )
                axes[ax_idx].set_ylabel("Learning Rate", fontsize=FONT_SIZES["normal"])
                axes[ax_idx].set_yscale("log")
                axes[ax_idx].legend(fontsize=FONT_SIZES["small"])
                axes[ax_idx].grid(True, alpha=0.3)

            # Set common x-axis label
            axes[-1].set_xlabel("Epoch", fontsize=FONT_SIZES["normal"])

            # Set title
            fig.suptitle(
                f"{title_prefix} - Training History", fontsize=FONT_SIZES["large"]
            )

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Training history plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create training history plot: {e}")

    def create_data_efficiency_plot(
        self,
        results_data: dict[str, Any],
        output_path: Path,
        title_prefix: str = "Standalone DNN Data Efficiency",
    ) -> None:
        """
        Create data efficiency plot showing model performance vs training data size.

        Args:
            results_data: Dictionary containing evaluation results for different data sizes
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
        """
        try:
            # Extract data sizes and corresponding metrics
            data_sizes = []
            test_losses = []

            for data_size, metrics in results_data.items():
                if isinstance(data_size, (int, str)) and str(data_size).isdigit():
                    data_sizes.append(int(data_size))
                    test_losses.append(
                        metrics.get("test_loss", metrics.get("test_mse", 0.0))
                    )

            if not data_sizes:
                self.logger.warning("No data efficiency results available")
                return

            # Filter out invalid test losses (non-positive values)
            valid_data = [(size, loss) for size, loss in zip(data_sizes, test_losses) 
                         if isinstance(loss, (int, float)) and loss > 0]
            
            if not valid_data:
                self.logger.warning("No valid data efficiency results with positive test losses")
                return

            # Sort by data size
            data_sizes, test_losses = zip(*sorted(valid_data))

            # Create plot
            fig, ax = plt.subplots(figsize=get_figure_size("single", ratio=0.8))

            colors = get_color_cycle("high_contrast", 1)
            ax.plot(
                data_sizes,
                test_losses,
                color=colors[0],
                linewidth=LINE_WIDTHS["thick"],
                marker="o",
                markersize=MARKER_SIZES["normal"],
                label="Standalone DNN",
            )

            # Formatting
            ax.set_xlabel("Training Data Size", fontsize=FONT_SIZES["normal"])
            ax.set_ylabel("Test Loss (MSE)", fontsize=FONT_SIZES["normal"])
            ax.set_title(f"{title_prefix}", fontsize=FONT_SIZES["large"])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=FONT_SIZES["small"])

            # Log scale for better visualization
            ax.set_xscale("log")
            ax.set_yscale("log")

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Data efficiency plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create data efficiency plot: {e}")

    def create_prediction_quality_plot(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_path: Path,
        title_prefix: str = "Prediction Quality Analysis",
        sample_size: int = 1000,
    ) -> None:
        """
        Create comprehensive prediction quality analysis plot.

        Args:
            predictions: Model predictions
            targets: True target values
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
            sample_size: Number of samples to plot (for performance)
        """
        try:
            # Sample data if too large
            if len(predictions) > sample_size:
                indices = np.random.choice(len(predictions), sample_size, replace=False)
                pred_sample = predictions[indices]
                target_sample = targets[indices]
            else:
                pred_sample = predictions
                target_sample = targets

            # Create 2x2 subplot for better proportions and more comprehensive analysis
            fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", ratio=0.8))
            axes = axes.flatten()
            colors = get_color_cycle("high_contrast", 4)

            # Plot 1: Predictions vs Targets scatter
            axes[0].scatter(
                target_sample,
                pred_sample,
                alpha=0.6,
                s=MARKER_SIZES["small"],
                color=colors[0],
            )
            # Perfect prediction line
            min_val = min(np.min(target_sample), np.min(pred_sample))
            max_val = max(np.max(target_sample), np.max(pred_sample))
            axes[0].plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=LINE_WIDTHS["thick"],
                label="Perfect Prediction",
            )

            axes[0].set_xlabel("True Values", fontsize=FONT_SIZES["small"])
            axes[0].set_ylabel("Predictions", fontsize=FONT_SIZES["small"])
            axes[0].set_title(
                "Predictions vs True Values", fontsize=FONT_SIZES["normal"]
            )
            axes[0].legend(fontsize=FONT_SIZES["tiny"])
            axes[0].grid(True, alpha=0.3)

            # Calculate R²
            r2 = stats.pearsonr(target_sample.flatten(), pred_sample.flatten())[0] ** 2
            axes[0].text(
                0.05,
                0.95,
                f"R² = {r2:.3f}",
                transform=axes[0].transAxes,
                fontsize=FONT_SIZES["small"],
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Plot 2: Residuals histogram
            residuals = pred_sample - target_sample
            axes[1].hist(
                residuals.flatten(), bins=50, alpha=0.7, color=colors[1], density=True
            )
            axes[1].axvline(
                0, color="r", linestyle="--", linewidth=LINE_WIDTHS["thick"]
            )
            axes[1].set_xlabel(
                "Residuals (Pred - True)", fontsize=FONT_SIZES["small"]
            )
            axes[1].set_ylabel("Density", fontsize=FONT_SIZES["small"])
            axes[1].set_title(
                "Residuals Distribution", fontsize=FONT_SIZES["normal"]
            )
            axes[1].grid(True, alpha=0.3)

            # Add residual statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            axes[1].text(
                0.05,
                0.95,
                f"μ = {mean_residual:.3e}\nσ = {std_residual:.3e}",
                transform=axes[1].transAxes,
                fontsize=FONT_SIZES["tiny"],
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Plot 3: Relative errors (pred-true)/true * 100%
            relative_errors = (pred_sample - target_sample) / (np.abs(target_sample) + 1e-8) * 100.0
            relative_errors_clipped = np.clip(relative_errors.flatten(), -200, 200)  # Clip extreme outliers
            axes[2].hist(
                relative_errors_clipped, bins=50, alpha=0.7, color=colors[2], density=True
            )
            axes[2].axvline(
                0, color="r", linestyle="--", linewidth=LINE_WIDTHS["thick"]
            )
            axes[2].set_xlabel(
                "Relative Error (%)", fontsize=FONT_SIZES["small"]
            )
            axes[2].set_ylabel("Density", fontsize=FONT_SIZES["small"])
            axes[2].set_title(
                "Relative Error Distribution", fontsize=FONT_SIZES["normal"]
            )
            axes[2].grid(True, alpha=0.3)

            # Add relative error statistics
            mean_rel_error = np.mean(relative_errors_clipped)
            std_rel_error = np.std(relative_errors_clipped)
            axes[2].text(
                0.05,
                0.95,
                f"μ = {mean_rel_error:.1f}%\nσ = {std_rel_error:.1f}%",
                transform=axes[2].transAxes,
                fontsize=FONT_SIZES["tiny"],
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Plot 4: Error vs True Values scatter
            abs_errors = np.abs(residuals.flatten())
            axes[3].scatter(
                target_sample.flatten(),
                abs_errors,
                alpha=0.5,
                s=MARKER_SIZES["tiny"],
                color=colors[3],
            )
            axes[3].set_xlabel("True Values", fontsize=FONT_SIZES["small"])
            axes[3].set_ylabel("Absolute Error", fontsize=FONT_SIZES["small"])
            axes[3].set_title("Error vs True Values", fontsize=FONT_SIZES["normal"])
            axes[3].grid(True, alpha=0.3)

            # Add MAE line
            mae = np.mean(abs_errors)
            axes[3].axhline(
                mae, color="orange", linestyle="--", 
                linewidth=LINE_WIDTHS["thick"], 
                label=f"MAE = {mae:.3e}"
            )
            axes[3].legend(fontsize=FONT_SIZES["tiny"])

            # Overall title
            fig.suptitle(f"{title_prefix}", fontsize=FONT_SIZES["large"], y=0.95)

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Prediction quality plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create prediction quality plot: {e}")

    def create_training_history_summary(
        self,
        histories_by_size: dict[int, dict[str, list[float]]],
        output_path: Path,
        title_prefix: str = "Training History Summary",
    ) -> None:
        """
        Create training history summary plot for different data sizes.

        Args:
            histories_by_size: Dictionary mapping data_size -> training_history
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
        """
        try:
            if not histories_by_size:
                self.logger.warning("No training histories available for summary")
                return

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", ratio=1.0))
            colors = get_color_cycle("high_contrast", len(histories_by_size))
            
            # Extract metrics
            metrics_to_plot = ['loss', 'val_loss', 'mae', 'val_mae']
            metric_titles = ['Training Loss', 'Validation Loss', 'Training MAE', 'Validation MAE']
            
            for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
                row, col = divmod(idx, 2)
                ax = axes[row, col]
                
                for color_idx, (data_size, history) in enumerate(sorted(histories_by_size.items())):
                    if metric in history and history[metric]:
                        epochs = range(1, len(history[metric]) + 1)
                        ax.plot(epochs, history[metric], 
                               color=colors[color_idx % len(colors)], 
                               linewidth=LINE_WIDTHS["normal"],
                               label=f"{data_size} events",
                               alpha=0.8)
                
                ax.set_xlabel('Epoch', fontsize=FONT_SIZES["small"])
                ax.set_ylabel(title, fontsize=FONT_SIZES["small"])
                ax.set_title(title, fontsize=FONT_SIZES["normal"])
                ax.grid(True, alpha=0.3)
                if idx == 0:  # Add legend only to first subplot
                    ax.legend(fontsize=FONT_SIZES["tiny"], loc='upper right')
                
            fig.suptitle(f"{title_prefix}", fontsize=FONT_SIZES["large"])
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Training history summary saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create training history summary: {e}")

    def create_metrics_vs_datasize_plot(
        self,
        data_size_metrics: dict[int, dict[str, float]],
        k_fold_stats: Optional[dict[int, dict[str, list[float]]]] = None,
        output_path: Path = None,
        title_prefix: str = "Metrics vs Data Size",
    ) -> None:
        """
        Create evaluation metrics vs data size plot with k-fold error bars.

        Args:
            data_size_metrics: Dictionary mapping data_size -> metrics
            k_fold_stats: Optional k-fold statistics for error bars
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
        """
        try:
            if not data_size_metrics:
                self.logger.warning("No data size metrics available")
                return

            data_sizes = sorted(data_size_metrics.keys())
            
            # Create 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", ratio=1.0))
            colors = get_color_cycle("high_contrast", 4)
            
            # Extract metrics with error bars
            metrics_info = [
                ('test_loss', 'Test Loss (MSE)', 'test_loss'),
                ('mae', 'Mean Absolute Error', 'mae'),
                ('mse', 'Mean Squared Error', 'mse'),
                ('r2', 'R² Score', 'r2')
            ]
            
            for idx, (metric_key, metric_title, kfold_key) in enumerate(metrics_info):
                row, col = divmod(idx, 2)
                ax = axes[row, col]
                
                # Extract values and error bars
                values = []
                errors = []
                valid_sizes = []
                
                for size in data_sizes:
                    if size in data_size_metrics:
                        metric_val = data_size_metrics[size].get(metric_key, 
                                   data_size_metrics[size].get('test_mse' if metric_key == 'test_loss' else metric_key, 0))
                        if isinstance(metric_val, (int, float)):
                            values.append(metric_val)
                            valid_sizes.append(size)
                            
                            # Add error bar if k-fold data available
                            if k_fold_stats and size in k_fold_stats:
                                kfold_values = k_fold_stats[size].get(kfold_key, [])
                                if kfold_values and len(kfold_values) > 1:
                                    errors.append(np.std(kfold_values))
                                else:
                                    errors.append(0)
                            else:
                                errors.append(0)
                
                if valid_sizes and values:
                    ax.errorbar(valid_sizes, values, yerr=errors,
                               color=colors[idx], marker='o', 
                               linewidth=LINE_WIDTHS["thick"],
                               markersize=MARKER_SIZES["normal"],
                               capsize=5, capthick=2)
                    
                    ax.set_xlabel('Training Data Size', fontsize=FONT_SIZES["small"])
                    ax.set_ylabel(metric_title, fontsize=FONT_SIZES["small"])
                    ax.set_title(metric_title, fontsize=FONT_SIZES["normal"])
                    ax.set_xscale('log')
                    if metric_key in ['test_loss', 'mae', 'mse'] and all(v > 0 for v in values):
                        ax.set_yscale('log')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'No valid {metric_title.lower()} data',
                           ha='center', va='center', transform=ax.transAxes)

            fig.suptitle(f"{title_prefix}", fontsize=FONT_SIZES["large"])
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Metrics vs data size plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create metrics vs data size plot: {e}")

    def create_pred_vs_true_subplots(
        self,
        predictions_by_size: dict[int, dict[str, np.ndarray]],
        output_path: Path,
        title_prefix: str = "Predictions vs True Values",
        sample_size: int = 1000,
    ) -> None:
        """
        Create pred vs true subplots for different data sizes.

        Args:
            predictions_by_size: Dictionary mapping data_size -> {predictions, targets}
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
            sample_size: Number of samples to plot per subplot
        """
        try:
            if not predictions_by_size:
                self.logger.warning("No predictions available for subplots")
                return

            # Filter valid predictions
            valid_predictions = {}
            for data_size, pred_dict in predictions_by_size.items():
                if ("predictions" in pred_dict and "targets" in pred_dict and
                    len(pred_dict["predictions"]) > 0 and len(pred_dict["targets"]) > 0):
                    valid_predictions[data_size] = pred_dict
            
            if not valid_predictions:
                self.logger.warning("No valid predictions for subplots")
                return
            
            # Create subplots grid
            n_sizes = len(valid_predictions)
            cols = min(3, n_sizes)
            rows = (n_sizes + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=get_figure_size("large", ratio=0.8))
            if n_sizes == 1:
                axes = [axes]
            elif rows == 1 and cols > 1:
                axes = axes.flatten()
            elif rows > 1 and cols == 1:
                axes = axes.flatten()
            elif rows > 1 and cols > 1:
                axes = axes.flatten()
            
            colors = get_color_cycle("high_contrast", n_sizes)
            
            for idx, (data_size, pred_dict) in enumerate(sorted(valid_predictions.items())):
                if idx >= len(axes):
                    break
                    
                ax = axes[idx]
                
                predictions = pred_dict["predictions"].flatten()
                targets = pred_dict["targets"].flatten()
                
                # Sample data if too large
                if len(predictions) > sample_size:
                    indices = np.random.choice(len(predictions), sample_size, replace=False)
                    predictions = predictions[indices]
                    targets = targets[indices]
                
                # Create scatter plot
                ax.scatter(targets, predictions, alpha=0.6, s=20, 
                          color=colors[idx % len(colors)])
                
                # Add perfect prediction line
                min_val = min(np.min(targets), np.min(predictions))
                max_val = max(np.max(targets), np.max(predictions))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                       alpha=0.8, linewidth=LINE_WIDTHS["normal"])
                
                # Calculate and display R²
                try:
                    ss_res = np.sum((targets - predictions) ** 2)
                    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           fontsize=FONT_SIZES["small"])
                except:
                    pass
                
                ax.set_title(f'{data_size} events', fontsize=FONT_SIZES["normal"])
                ax.set_xlabel('True Values', fontsize=FONT_SIZES["small"])
                ax.set_ylabel('Predictions', fontsize=FONT_SIZES["small"])
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for idx in range(n_sizes, len(axes)):
                axes[idx].set_visible(False)
            
            fig.suptitle(f"{title_prefix}", fontsize=FONT_SIZES["large"])
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            self.logger.info(f"Pred vs true subplots saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create pred vs true subplots: {e}")

    def create_error_analysis_plot(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_path: Path,
        title_prefix: str = "Error Analysis",
        n_bins: int = 50,
    ) -> None:
        """
        Create detailed error analysis plots.

        Args:
            predictions: Model predictions
            targets: True target values
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
            n_bins: Number of bins for histograms
        """
        try:
            # Calculate errors
            absolute_errors = np.abs(predictions - targets)
            relative_errors = np.abs(predictions - targets) / (np.abs(targets) + 1e-8)

            # Create 2x2 subplot
            fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", ratio=1.0))
            colors = get_color_cycle("high_contrast", 4)

            # Plot 1: Absolute errors histogram
            axes[0, 0].hist(
                absolute_errors.flatten(),
                bins=n_bins,
                alpha=0.7,
                color=colors[0],
                density=True,
            )
            axes[0, 0].set_xlabel("Absolute Error", fontsize=FONT_SIZES["small"])
            axes[0, 0].set_ylabel("Density", fontsize=FONT_SIZES["small"])
            axes[0, 0].set_title(
                "Absolute Error Distribution", fontsize=FONT_SIZES["normal"]
            )
            axes[0, 0].grid(True, alpha=0.3)

            # Add statistics
            mae = np.mean(absolute_errors)
            axes[0, 0].axvline(
                mae,
                color="r",
                linestyle="--",
                linewidth=LINE_WIDTHS["thick"],
                label=f"MAE = {mae:.3e}",
            )
            axes[0, 0].legend(fontsize=FONT_SIZES["tiny"])

            # Plot 2: Relative errors histogram
            # Clip extreme relative errors for better visualization
            rel_errors_clipped = np.clip(
                relative_errors, 0, np.percentile(relative_errors, 95)
            )
            axes[0, 1].hist(
                rel_errors_clipped.flatten(),
                bins=n_bins,
                alpha=0.7,
                color=colors[1],
                density=True,
            )
            axes[0, 1].set_xlabel("Relative Error", fontsize=FONT_SIZES["small"])
            axes[0, 1].set_ylabel("Density", fontsize=FONT_SIZES["small"])
            axes[0, 1].set_title(
                "Relative Error Distribution", fontsize=FONT_SIZES["normal"]
            )
            axes[0, 1].grid(True, alpha=0.3)

            # Add statistics
            mape = np.mean(relative_errors) * 100
            axes[0, 1].axvline(
                np.mean(rel_errors_clipped),
                color="r",
                linestyle="--",
                linewidth=LINE_WIDTHS["thick"],
                label=f"MAPE = {mape:.1f}%",
            )
            axes[0, 1].legend(fontsize=FONT_SIZES["tiny"])

            # Plot 3: Error vs True Values
            axes[1, 0].scatter(
                targets.flatten(),
                absolute_errors.flatten(),
                alpha=0.5,
                s=MARKER_SIZES["tiny"],
                color=colors[2],
            )
            axes[1, 0].set_xlabel("True Values", fontsize=FONT_SIZES["small"])
            axes[1, 0].set_ylabel("Absolute Error", fontsize=FONT_SIZES["small"])
            axes[1, 0].set_title("Error vs True Values", fontsize=FONT_SIZES["normal"])
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Cumulative error distribution
            sorted_errors = np.sort(absolute_errors.flatten())
            cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            axes[1, 1].plot(
                sorted_errors,
                cumulative,
                color=colors[3],
                linewidth=LINE_WIDTHS["thick"],
            )
            axes[1, 1].set_xlabel("Absolute Error", fontsize=FONT_SIZES["small"])
            axes[1, 1].set_ylabel(
                "Cumulative Probability", fontsize=FONT_SIZES["small"]
            )
            axes[1, 1].set_title(
                "Cumulative Error Distribution", fontsize=FONT_SIZES["normal"]
            )
            axes[1, 1].grid(True, alpha=0.3)

            # Add percentile lines
            for percentile in [50, 90, 95]:
                error_val = np.percentile(sorted_errors, percentile)
                axes[1, 1].axvline(
                    error_val,
                    linestyle="--",
                    alpha=0.7,
                    label=f"{percentile}th percentile",
                )
            axes[1, 1].legend(fontsize=FONT_SIZES["tiny"])

            # Overall title
            fig.suptitle(f"{title_prefix}", fontsize=FONT_SIZES["large"])

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Error analysis plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create error analysis plot: {e}")

    def create_multi_size_comparison_plot(
        self,
        multi_size_data: dict[int, dict[str, np.ndarray]],
        output_path: Path,
        title_prefix: str = "Multi-Size Prediction Comparison",
        max_samples_per_plot: int = 500,
    ) -> None:
        """
        Create comparison plot showing prediction quality across different data sizes.

        Args:
            multi_size_data: Dictionary with data_size -> {predictions, targets}
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
            max_samples_per_plot: Maximum samples per subplot
        """
        try:
            data_sizes = sorted(multi_size_data.keys())
            n_sizes = len(data_sizes)

            if n_sizes == 0:
                self.logger.warning("No multi-size data available")
                return

            # Create subplot grid
            n_cols = min(n_sizes, 4)
            n_rows = (n_sizes + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=get_figure_size("double", ratio=0.8)
            )
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            colors = get_color_cycle("high_contrast", n_sizes)

            for i, data_size in enumerate(data_sizes):
                if i >= len(axes):
                    break

                data = multi_size_data[data_size]
                predictions = data["predictions"]
                targets = data["targets"]

                # Sample data for performance
                if len(predictions) > max_samples_per_plot:
                    indices = np.random.choice(
                        len(predictions), max_samples_per_plot, replace=False
                    )
                    pred_sample = predictions[indices]
                    target_sample = targets[indices]
                else:
                    pred_sample = predictions
                    target_sample = targets

                # Scatter plot
                axes[i].scatter(
                    target_sample,
                    pred_sample,
                    alpha=0.6,
                    s=MARKER_SIZES["tiny"],
                    color=colors[i % len(colors)],
                )

                # Perfect prediction line
                min_val = min(np.min(target_sample), np.min(pred_sample))
                max_val = max(np.max(target_sample), np.max(pred_sample))
                axes[i].plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "r--",
                    linewidth=LINE_WIDTHS["thin"],
                    alpha=0.7,
                )

                # Calculate R²
                r2 = (
                    stats.pearsonr(target_sample.flatten(), pred_sample.flatten())[0]
                    ** 2
                )

                # Format data size label
                size_label = (
                    f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                )
                axes[i].set_title(
                    f"{size_label} events (R²={r2:.3f})", fontsize=FONT_SIZES["small"]
                )
                axes[i].grid(True, alpha=0.3)

                if i >= (n_rows - 1) * n_cols:  # Bottom row
                    axes[i].set_xlabel("True Values", fontsize=FONT_SIZES["small"])
                if i % n_cols == 0:  # Left column
                    axes[i].set_ylabel("Predictions", fontsize=FONT_SIZES["small"])

            # Hide unused subplots
            for i in range(n_sizes, len(axes)):
                axes[i].set_visible(False)

            # Overall title
            fig.suptitle(f"{title_prefix}", fontsize=FONT_SIZES["large"])

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Multi-size comparison plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create multi-size comparison plot: {e}")

    def create_prediction_vs_true_2d_histogram(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_path: Path,
        title_prefix: str = "Prediction vs True Values",
        n_bins: int = 50,
        sample_size: int = 10000,
    ) -> None:
        """
        Create 2D histogram of predictions vs true values.

        Args:
            predictions: Model predictions
            targets: True target values
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
            n_bins: Number of bins for 2D histogram
            sample_size: Number of samples to use (for performance)
        """
        try:
            # Sample data if too large
            if len(predictions) > sample_size:
                indices = np.random.choice(len(predictions), sample_size, replace=False)
                pred_sample = predictions[indices].flatten()
                target_sample = targets[indices].flatten()
            else:
                pred_sample = predictions.flatten()
                target_sample = targets.flatten()

            # Create figure
            fig, ax = plt.subplots(figsize=get_figure_size("single", ratio=1.0))

            # Create 2D histogram
            h, xedges, yedges = np.histogram2d(
                target_sample, pred_sample, bins=n_bins, density=True
            )

            # Create mesh for plotting
            X, Y = np.meshgrid(xedges, yedges)
            im = ax.pcolormesh(X, Y, h.T, cmap='viridis', shading='auto')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Density', fontsize=FONT_SIZES["normal"])

            # Perfect prediction line
            min_val = min(np.min(target_sample), np.min(pred_sample))
            max_val = max(np.max(target_sample), np.max(pred_sample))
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                'r--',
                linewidth=LINE_WIDTHS["thick"],
                label='Perfect Prediction'
            )

            # Formatting
            ax.set_xlabel('True Values', fontsize=FONT_SIZES["normal"])
            ax.set_ylabel('Predictions', fontsize=FONT_SIZES["normal"])
            ax.set_title(f'{title_prefix} - 2D Histogram', fontsize=FONT_SIZES["large"])
            ax.legend(fontsize=FONT_SIZES["small"])
            ax.grid(True, alpha=0.3)

            # Calculate and display R²
            r2 = stats.pearsonr(target_sample, pred_sample)[0] ** 2
            ax.text(
                0.05, 0.95,
                f'R² = {r2:.3f}\nN = {len(target_sample):,}',
                transform=ax.transAxes,
                fontsize=FONT_SIZES["small"],
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"2D histogram plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create 2D histogram plot: {e}")

    def create_relative_error_histogram(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_path: Path,
        title_prefix: str = "Relative Error Analysis",
        n_bins: int = 100,
        max_relative_error: float = 2.0,
    ) -> None:
        """
        Create relative error histogram (pred-true)/true * 100%.

        Args:
            predictions: Model predictions
            targets: True target values
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
            n_bins: Number of bins for histogram
            max_relative_error: Maximum relative error to display (to clip outliers)
        """
        try:
            # Calculate relative errors as (pred - true) / true * 100%
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            relative_errors = (predictions - targets) / (np.abs(targets) + epsilon) * 100.0

            # Flatten arrays
            relative_errors = relative_errors.flatten()

            # Remove extreme outliers
            valid_mask = np.isfinite(relative_errors)
            relative_errors = relative_errors[valid_mask]

            # Clip to reasonable range
            relative_errors_clipped = np.clip(
                relative_errors, 
                -max_relative_error * 100, 
                max_relative_error * 100
            )

            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=get_figure_size("single", ratio=1.2))
            colors = get_color_cycle("high_contrast", 3)

            # Plot 1: Full range histogram
            axes[0].hist(
                relative_errors_clipped,
                bins=n_bins,
                alpha=0.7,
                color=colors[0],
                density=True,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add vertical line at zero
            axes[0].axvline(
                0, color='red', linestyle='--', 
                linewidth=LINE_WIDTHS["thick"], 
                label='Perfect Prediction'
            )

            # Calculate and display statistics
            mean_rel_error = np.mean(relative_errors_clipped)
            std_rel_error = np.std(relative_errors_clipped)
            median_rel_error = np.median(relative_errors_clipped)

            axes[0].axvline(
                mean_rel_error, color='orange', linestyle='-', 
                linewidth=LINE_WIDTHS["thick"], 
                label=f'Mean = {mean_rel_error:.1f}%'
            )
            axes[0].axvline(
                median_rel_error, color='green', linestyle='-', 
                linewidth=LINE_WIDTHS["thick"], 
                label=f'Median = {median_rel_error:.1f}%'
            )

            axes[0].set_xlabel('Relative Error (%)', fontsize=FONT_SIZES["normal"])
            axes[0].set_ylabel('Density', fontsize=FONT_SIZES["normal"])
            axes[0].set_title(
                f'{title_prefix} - Full Range', 
                fontsize=FONT_SIZES["normal"]
            )
            axes[0].legend(fontsize=FONT_SIZES["small"])
            axes[0].grid(True, alpha=0.3)

            # Add statistics text box
            stats_text = (
                f'Statistics:\n'
                f'Mean: {mean_rel_error:.2f}%\n'
                f'Std: {std_rel_error:.2f}%\n'
                f'Median: {median_rel_error:.2f}%\n'
                f'N valid: {len(relative_errors_clipped):,}'
            )
            axes[0].text(
                0.05, 0.95,
                stats_text,
                transform=axes[0].transAxes,
                fontsize=FONT_SIZES["small"],
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
            )

            # Plot 2: Zoomed in histogram around zero
            zoom_range = min(50, np.percentile(np.abs(relative_errors_clipped), 90))
            mask_zoom = np.abs(relative_errors_clipped) <= zoom_range
            
            if np.sum(mask_zoom) > 0:
                axes[1].hist(
                    relative_errors_clipped[mask_zoom],
                    bins=n_bins//2,
                    alpha=0.7,
                    color=colors[1],
                    density=True,
                    edgecolor='black',
                    linewidth=0.5
                )

                axes[1].axvline(
                    0, color='red', linestyle='--', 
                    linewidth=LINE_WIDTHS["thick"], 
                    label='Perfect Prediction'
                )

                axes[1].set_xlabel('Relative Error (%)', fontsize=FONT_SIZES["normal"])
                axes[1].set_ylabel('Density', fontsize=FONT_SIZES["normal"])
                axes[1].set_title(
                    f'Zoomed View (±{zoom_range:.0f}%)', 
                    fontsize=FONT_SIZES["normal"]
                )
                axes[1].legend(fontsize=FONT_SIZES["small"])
                axes[1].grid(True, alpha=0.3)

                # Calculate percentiles for zoomed view
                p68 = np.percentile(np.abs(relative_errors_clipped[mask_zoom]), 68)
                p90 = np.percentile(np.abs(relative_errors_clipped[mask_zoom]), 90)
                
                zoom_stats_text = (
                    f'Central region:\n'
                    f'68% within ±{p68:.1f}%\n'
                    f'90% within ±{p90:.1f}%'
                )
                axes[1].text(
                    0.95, 0.95,
                    zoom_stats_text,
                    transform=axes[1].transAxes,
                    fontsize=FONT_SIZES["small"],
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
                )

            # Overall title
            fig.suptitle(f'{title_prefix}', fontsize=FONT_SIZES["large"])

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Relative error histogram saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create relative error histogram: {e}")

    def create_data_size_comparison_summary(
        self,
        data_size_results: dict[int, dict[str, float]],
        output_path: Path,
        title_prefix: str = "Data Size Effect Analysis",
        k_fold_results: Optional[dict[int, dict[str, list[float]]]] = None,
    ) -> None:
        """
        Create comprehensive data size comparison with k-fold error bars.

        Args:
            data_size_results: Dictionary with data_size -> metrics
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
            k_fold_results: Optional k-fold results for error bars
        """
        try:
            data_sizes = sorted(data_size_results.keys())
            if not data_sizes:
                self.logger.warning("No data size results available")
                return

            # Create 2x2 subplot
            fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", ratio=1.0))
            colors = get_color_cycle("high_contrast", 4)

            # Extract metrics
            test_losses = []
            maes = []
            mses = []
            r2_scores = []

            test_loss_errs = []
            mae_errs = []
            mse_errs = []
            r2_errs = []

            for size in data_sizes:
                result = data_size_results[size]
                test_losses.append(result.get('test_loss', result.get('test_mse', 0)))
                maes.append(result.get('mae', 0))
                mses.append(result.get('mse', result.get('test_mse', 0)))
                r2_scores.append(result.get('r2', 0))

                # Add error bars if k-fold data available
                if k_fold_results and size in k_fold_results:
                    kfold_data = k_fold_results[size]
                    test_loss_errs.append(np.std(kfold_data.get('test_loss', [0])))
                    mae_errs.append(np.std(kfold_data.get('mae', [0])))
                    mse_errs.append(np.std(kfold_data.get('mse', [0])))
                    r2_errs.append(np.std(kfold_data.get('r2', [0])))
                else:
                    test_loss_errs.append(0)
                    mae_errs.append(0)
                    mse_errs.append(0)
                    r2_errs.append(0)

            # Plot 1: Test Loss vs Data Size
            if any(loss > 0 for loss in test_losses):
                axes[0, 0].errorbar(
                    data_sizes, test_losses, yerr=test_loss_errs,
                    color=colors[0], linewidth=LINE_WIDTHS["thick"],
                    marker='o', markersize=MARKER_SIZES["normal"],
                    capsize=5, capthick=2
                )
                axes[0, 0].set_xlabel('Training Data Size', fontsize=FONT_SIZES["normal"])
                axes[0, 0].set_ylabel('Test Loss (MSE)', fontsize=FONT_SIZES["normal"])
                axes[0, 0].set_title('Test Loss vs Data Size', fontsize=FONT_SIZES["normal"])
                axes[0, 0].set_xscale('log')
                if all(loss > 0 for loss in test_losses):
                    axes[0, 0].set_yscale('log')
                axes[0, 0].grid(True, alpha=0.3)
            else:
                axes[0, 0].text(0.5, 0.5, 'No valid test loss data', 
                              ha='center', va='center', transform=axes[0, 0].transAxes)

            # Plot 2: MAE vs Data Size  
            if any(mae > 0 for mae in maes):
                axes[0, 1].errorbar(
                    data_sizes, maes, yerr=mae_errs,
                    color=colors[1], linewidth=LINE_WIDTHS["thick"],
                    marker='s', markersize=MARKER_SIZES["normal"],
                    capsize=5, capthick=2
                )
                axes[0, 1].set_xlabel('Training Data Size', fontsize=FONT_SIZES["normal"])
                axes[0, 1].set_ylabel('Mean Absolute Error', fontsize=FONT_SIZES["normal"])
                axes[0, 1].set_title('MAE vs Data Size', fontsize=FONT_SIZES["normal"])
                axes[0, 1].set_xscale('log')
                if all(mae > 0 for mae in maes):
                    axes[0, 1].set_yscale('log')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No valid MAE data', 
                              ha='center', va='center', transform=axes[0, 1].transAxes)

            # Plot 3: R² Score vs Data Size
            axes[1, 0].errorbar(
                data_sizes, r2_scores, yerr=r2_errs,
                color=colors[2], linewidth=LINE_WIDTHS["thick"],
                marker='^', markersize=MARKER_SIZES["normal"],
                capsize=5, capthick=2
            )
            axes[1, 0].set_xlabel('Training Data Size', fontsize=FONT_SIZES["normal"])
            axes[1, 0].set_ylabel('R² Score', fontsize=FONT_SIZES["normal"])
            axes[1, 0].set_title('R² Score vs Data Size', fontsize=FONT_SIZES["normal"])
            axes[1, 0].set_xscale('log')
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Summary table
            axes[1, 1].axis('off')
            
            # Create summary table
            table_data = []
            for i, size in enumerate(data_sizes):
                size_str = f"{size//1000}k" if size >= 1000 else str(size)
                row = [
                    size_str,
                    f"{test_losses[i]:.2e}",
                    f"{maes[i]:.2e}",
                    f"{r2_scores[i]:.3f}"
                ]
                table_data.append(row)

            table = axes[1, 1].table(
                cellText=table_data,
                colLabels=['Data Size', 'Test Loss', 'MAE', 'R²'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0.2, 1, 0.7]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(FONT_SIZES["small"])
            table.scale(1, 1.5)
            
            # Style the table
            for i in range(len(data_sizes) + 1):
                for j in range(4):
                    if i == 0:  # Header
                        table[(i, j)].set_facecolor('#E6E6FA')
                        table[(i, j)].set_text_props(weight='bold')
                    else:
                        table[(i, j)].set_facecolor('#F8F8FF')

            axes[1, 1].set_title('Summary Statistics', fontsize=FONT_SIZES["normal"])

            # Overall title
            fig.suptitle(f'{title_prefix}', fontsize=FONT_SIZES["large"])

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Data size comparison summary saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create data size comparison summary: {e}")
