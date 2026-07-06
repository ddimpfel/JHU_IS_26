from PIL import Image
import ast
import json
from typing import Any
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
import torch
import re

FIGURE_NUMBER = 1
PLOT_BLUE = "cornflowerblue"
def reset_figure_number(number=1):
    global FIGURE_NUMBER
    FIGURE_NUMBER = number


def _literal_tuple_key(value):
    if not isinstance(value, str):
        return value

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value

    if isinstance(parsed, tuple):
        return parsed
    return value


def _maybe_array(value):
    if isinstance(value, list):
        if not value:
            return np.array([])
        if all(isinstance(item, (int, float, bool, np.integer, np.floating)) for item in value):
            return np.asarray(value)
        if all(isinstance(item, list) for item in value):
            try:
                return np.asarray(value)
            except ValueError:
                return value
    return value


def _normalize_pred_cache_entry(value):
    if isinstance(value, dict):
        normalized = value.copy()
        if "logits" in normalized:
            normalized["logits"] = np.asarray(normalized["logits"])
        if "targets" in normalized:
            normalized["targets"] = np.asarray(normalized["targets"])
        if "paths" in normalized and normalized["paths"] is not None:
            normalized["paths"] = list(normalized["paths"])
        return normalized

    if isinstance(value, list) and len(value) in {2, 3}:
        normalized = {
            "logits": np.asarray(value[0]),
            "targets": np.asarray(value[1]),
        }
        if len(value) == 3:
            normalized["paths"] = list(value[2])
        return normalized

    return value


def _resolve_class_label(class_id, class_names=None):
    if class_names is None:
        return str(class_id)

    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_id))

    if isinstance(class_names, pd.DataFrame):
        class_id_column = "ClassId" if "ClassId" in class_names.columns else class_names.columns[0]
        label_column = "SignName" if "SignName" in class_names.columns else class_names.columns[-1]
        matches = class_names.loc[class_names[class_id_column] == class_id, label_column]
        if isinstance(matches, pd.Series) and not matches.empty:
            return str(matches.iloc[0])

    return str(class_id)


def _resolve_prediction_image_path(saved_path, image_root=None):
    path = Path(saved_path)
    if path.exists():
        return path
    if image_root is not None:
        rooted_path = Path(image_root) / path
        if rooted_path.exists():
            return rooted_path
    raise FileNotFoundError(f"Could not resolve image path: {saved_path}")


def _short_plot_label(value):
    text = Path(str(value)).stem
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", text) if part]
    if not parts:
        return text

    first = parts[0]
    initials = "".join(part[0].upper() for part in parts[1:5])
    return f"{first}{initials}"


def _resolve_short_label(row):
    if "Short Label" in row and pd.notna(row["Short Label"]):
        return row["Short Label"]

    label_source = row.get("Source File", row.get("Model", ""))
    return _short_plot_label(label_source)


def _with_short_labels(df: pd.DataFrame):
    plot_df = df.copy()
    if "Short Label" not in plot_df.columns:
        if "Source File" in plot_df.columns:
            plot_df["Short Label"] = plot_df["Source File"].map(_short_plot_label)
        elif "Model" in plot_df.columns:
            plot_df["Short Label"] = plot_df["Model"].map(_short_plot_label)
    return plot_df


def _build_single_color_palette(labels):
    return {label: PLOT_BLUE for label in pd.unique(pd.Series(labels))}


def _apply_short_legend(ax, labels, title="Model"):
    unique_labels = [label for label in pd.unique(pd.Series(labels)) if pd.notna(label)]
    handles = [
        Line2D([0], [0], color=PLOT_BLUE, marker="o", linewidth=2, linestyle="-", label=label)
        for label in unique_labels
    ]
    if handles:
        ax.legend(handles=handles, title=title)


def load_serialized_training_run(file_path: str | Path):
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    results = payload.get("results", {}).copy()
    eval_mat = np.asarray(payload.get("eval_mat", []), dtype=float)
    pred_cache = {}
    saved_cache = payload.get("pred_cache", {})
    if isinstance(saved_cache, dict):
        for key, value in saved_cache.items():
            pred_cache[_literal_tuple_key(key)] = _normalize_pred_cache_entry(value)
    elif isinstance(saved_cache, list):
        for it in saved_cache:
            if isinstance(it, str): print(it); print(file_path); continue
            for key, value in it.items():
                parsed_key = _literal_tuple_key(key)
                pred_cache[parsed_key] = _normalize_pred_cache_entry(value)

    for metric_name in [
        "Backward Transfer Per Task",
        "Backward Transfer Per Task Macro F1",
        "Backward Transfer Per Task Micro F1",
        "Backward Transfer Per Task Weighted F1",
        "Forward Transfer Per Task",
        "Forward Transfer Per Task Macro F1",
        "Forward Transfer Per Task Micro F1",
        "Forward Transfer Per Task Weighted F1",
        "Task Forgetting List",
        "Task Forgetting List Macro F1",
        "Task Forgetting List Micro F1",
        "Task Forgetting List Weighted F1",
        "Training Time per Stage",
        "Eval Matrix Macro F1",
        "Eval Matrix Micro F1",
        "Eval Matrix Weighted F1",
    ]:
        if metric_name in results:
            results[metric_name] = _maybe_array(results[metric_name])

    results["Eval Matrix"] = eval_mat

    model_name = results.get("Model") or file_path.stem
    return {
        "model_name": model_name,
        "file_path": file_path,
        "results": results,
        "pred_cache": pred_cache,
        "eval_mat": eval_mat,
    }


def load_serialized_training_runs(paths_or_dir, pattern="*.json"):
    if isinstance(paths_or_dir, (str, Path)):
        base_path = Path(paths_or_dir)
        if base_path.is_dir():
            paths = sorted(base_path.glob(pattern))
        else:
            paths = [base_path]
    else:
        paths = [Path(path) for path in paths_or_dir]

    runs = [load_serialized_training_run(path) for path in paths]
    if not runs:
        raise ValueError("No serialized training runs were found.")
    return runs


def build_serialized_results_table(runs):
    rows = []
    for run in runs:
        results = run["results"].copy()
        model_name = results.pop("Model", run["model_name"])
        rows.append({
            "Model": model_name,
            "Short Label": _short_plot_label(run["file_path"].name),
            **results,
        })

    comparison_df = pd.DataFrame(rows)
    numeric_columns = [
        "AvgAcc",
        "AvgAcc Macro F1",
        "AvgAcc Micro F1",
        "AvgAcc Weighted F1",
        "Baseline Accuracy",
        "Backward Transfer",
        "Backward Transfer Macro F1",
        "Backward Transfer Micro F1",
        "Backward Transfer Weighted F1",
        "Forward Transfer",
        "Forward Transfer Macro F1",
        "Forward Transfer Micro F1",
        "Forward Transfer Weighted F1",
        "Average Forgetting",
        "Average Forgetting Macro F1",
        "Average Forgetting Micro F1",
        "Average Forgetting Weighted F1",
        "Full Validation Loss",
        "Full Validation F1",
        "Full Validation Macro F1",
        "Full Validation Micro F1",
        "Full Validation Weighted F1",
        "Total Compute Time (s)",
        "Num Parameters",
    ]
    for column in numeric_columns:
        if column in comparison_df.columns:
            comparison_df[column] = pd.to_numeric(comparison_df[column], errors="coerce")

    return comparison_df


def build_transfer_table(comparison_df: pd.DataFrame, suffix: str = ""):
    metric_suffix = f" {suffix}" if suffix else ""
    transfer_rows = []
    for _, row in comparison_df.iterrows():
        bwt_values = np.atleast_1d(
            np.asarray(row.get(f"Backward Transfer Per Task{metric_suffix}", []), dtype=float)
        )
        fwt_values = np.atleast_1d(
            np.asarray(row.get(f"Forward Transfer Per Task{metric_suffix}", []), dtype=float)
        )
        max_len = min(len(bwt_values), len(fwt_values))

        for task_index in range(max_len):
            transfer_rows.append({
                "Model": row["Model"],
                "Short Label": _resolve_short_label(row),
                "Task Pair": f"Task {task_index + 1} -> Task {task_index + 2}",
                "Backward Transfer": bwt_values[task_index],
                "Forward Transfer": fwt_values[task_index],
            })

    return pd.DataFrame(transfer_rows)


def build_forgetting_table(comparison_df: pd.DataFrame, suffix: str = ""):
    metric_suffix = f" {suffix}" if suffix else ""
    forgetting_rows = []
    for _, row in comparison_df.iterrows():
        forgetting_values = np.atleast_1d(
            np.asarray(row.get(f"Task Forgetting List{metric_suffix}", []), dtype=float)
        )
        for task_index, forgetting_value in enumerate(forgetting_values, start=1):
            forgetting_rows.append({
                "Model": row["Model"],
                "Short Label": _resolve_short_label(row),
                "Task": f"Task {task_index}",
                "Forgetting": forgetting_value,
            })

    return pd.DataFrame(forgetting_rows)


def build_training_time_table(comparison_df: pd.DataFrame):
    rows = []
    for _, row in comparison_df.iterrows():
        stage_times = np.atleast_1d(np.asarray(row.get("Training Time per Stage", []), dtype=float))
        for stage_index, stage_time in enumerate(stage_times, start=1):
            rows.append({
                "Model": row["Model"],
                "Short Label": _resolve_short_label(row),
                "Stage": f"Stage {stage_index}",
                "Training Time (s)": stage_time,
            })

    return pd.DataFrame(rows)


def plot_incorrect_prediction_per_stage(
    run,
    eval_task_index=0,
    sample_index=0,
    image_root=None,
    class_names=None,
    figsize_per_stage=(4, 4),
):
    global FIGURE_NUMBER

    pred_cache = run.get("pred_cache", {})
    relevant_keys = sorted(
        [key for key in pred_cache if isinstance(key, tuple) and len(key) == 2 and key[1] == eval_task_index],
        key=lambda item: item[0],
    )
    if not relevant_keys:
        raise ValueError(f"No prediction cache entries found for evaluation task {eval_task_index + 1}.")

    num_stages = len(relevant_keys)
    fig, axes = plt.subplots(1, num_stages, figsize=(figsize_per_stage[0] * num_stages, figsize_per_stage[1]))
    if num_stages == 1:
        axes = [axes]

    for ax, cache_key in zip(axes, relevant_keys):
        stage_index, _ = cache_key
        entry = _normalize_pred_cache_entry(pred_cache[cache_key])
        if not isinstance(entry, dict):
            ax.text(0.5, 0.5, "Unsupported cache format", ha="center", va="center")
            ax.set_title(f"Stage {stage_index + 1}")
            ax.axis("off")
            continue

        logits = np.asarray(entry.get("logits", []))
        targets = np.asarray(entry.get("targets", []))
        paths = entry.get("paths")

        if logits.size == 0 or targets.size == 0:
            ax.text(0.5, 0.5, "No predictions saved", ha="center", va="center")
            ax.axis("off")
            continue

        preds = np.argmax(logits, axis=1)
        wrong_indices = np.flatnonzero(preds != targets)

        if wrong_indices.size == 0:
            ax.text(0.5, 0.5, "No incorrect predictions", ha="center", va="center")
            ax.set_title(f"Stage {stage_index + 1}")
            ax.axis("off")
            continue

        chosen_index = wrong_indices[min(sample_index, wrong_indices.size - 1)]
        true_label = int(targets[chosen_index])
        pred_label = int(preds[chosen_index])

        if not paths:
            ax.text(
                0.5,
                0.5,
                f"Missing image paths\nTrue: {_resolve_class_label(true_label, class_names)}\nPred: {_resolve_class_label(pred_label, class_names)}",
                ha="center",
                va="center",
            )
            ax.set_title(f"Stage {stage_index + 1}")
            ax.axis("off")
            continue

        image_path = _resolve_prediction_image_path(paths[chosen_index], image_root=image_root)
        image = Image.open(image_path).convert("RGB")

        ax.imshow(image)
        ax.set_title(
            "\n".join([
                f"Stage {stage_index + 1}",
                f"True: {_resolve_class_label(true_label, class_names)}",
                f"Pred: {_resolve_class_label(pred_label, class_names)}",
            ]),
            fontsize=10,
        )
        ax.axis("off")

    plt.suptitle(
        f"Figure {FIGURE_NUMBER}: Incorrect Prediction for Evaluation Task {eval_task_index + 1} by Training Stage"
    )
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1


def plot_metric_bar(comparison_df: pd.DataFrame, metric: str, ascending=False, figsize=(10, 6)):
    global FIGURE_NUMBER
    plot_df = _with_short_labels(comparison_df).sort_values(metric, ascending=ascending)

    plt.figure(figsize=figsize)
    sns.barplot(data=plot_df, x=metric, y="Short Label", color=PLOT_BLUE)
    plt.title(f"Figure {FIGURE_NUMBER}: {metric} by Model")
    plt.xlabel(metric)
    plt.ylabel("")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1


def plot_eval_matrix_heatmap(run, figsize=(8, 6), cmap=None):
    global FIGURE_NUMBER
    eval_mat = np.asarray(run["eval_mat"], dtype=float)
    model_name = _short_plot_label(run["file_path"].name)
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("cornflower_fade", ["#ffffff", PLOT_BLUE])

    plt.figure(figsize=figsize)
    sns.heatmap(
        eval_mat,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Macro F1"},
    )
    plt.title(f"Figure {FIGURE_NUMBER}: Eval Matrix for {model_name}")
    plt.xlabel("Training Stage")
    plt.ylabel("Evaluation Task")
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1


def plot_task_forgetting(comparison_df: pd.DataFrame, figsize=(10, 6)):
    global FIGURE_NUMBER
    plot_df = _with_short_labels(comparison_df)
    plt.figure(figsize=figsize)

    max_tasks = 0
    ax = plt.gca()
    for _, row in plot_df.iterrows():
        forgetting_list = np.atleast_1d(np.asarray(row["Task Forgetting List"], dtype=float))
        if forgetting_list.size == 0:
            continue
        tasks = np.arange(1, len(forgetting_list) + 1)
        max_tasks = max(max_tasks, len(forgetting_list))
        plt.plot(tasks, forgetting_list, marker="o", linewidth=2, color=PLOT_BLUE, label=row["Short Label"])

    plt.title(f"Figure {FIGURE_NUMBER}: Task Forgetting by Model")
    plt.xlabel("Task")
    plt.ylabel("Forgetting")
    if max_tasks:
        plt.xticks(np.arange(1, max_tasks + 1))
    plt.grid(True, linestyle="--", alpha=0.35)
    _apply_short_legend(ax, plot_df["Short Label"])
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1
    
    
def plot_feature_vbars(feature_data: pd.Series | list, feature_name, ylabel=None, title=None):
    global FIGURE_NUMBER
    if title:
        plt_title = f"Figure {FIGURE_NUMBER}: {title}"
    else:
        plt_title = f"Figure {FIGURE_NUMBER}: Bar Chart of {feature_name}"

    plt.figure(figsize=(10, 6))
    if isinstance(feature_data, pd.Series):
        data = (feature_data.value_counts(normalize=True).sort_index().reset_index())
        data.columns = [feature_name, "freq"]
        sns.barplot(data=data, x=feature_name, y="freq", color="C0", alpha=0.8)
        plt.ylabel(ylabel or "Frequency")
    else:
        sns.barplot(x=range(len(feature_data)), y=feature_data, color="C0", alpha=0.8)
        plt.ylabel(ylabel or "Counts")

    plt.title(plt_title)
    plt.xlabel(feature_name or "x")
    plt.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1


def plot_transfer_curves(transfer_df: pd.DataFrame, metric: str, figsize=(10, 6)):
    global FIGURE_NUMBER
    plot_df = _with_short_labels(transfer_df)
    plt.figure(figsize=figsize)
    ax = sns.lineplot(
        data=plot_df,
        x="Task Pair",
        y=metric,
        hue="Short Label",
        marker="o",
        palette=_build_single_color_palette(plot_df["Short Label"]),
    )
    plt.title(f"Figure {FIGURE_NUMBER}: {metric} by Task Pair")
    plt.xlabel("Task Pair")
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", alpha=0.35)
    _apply_short_legend(ax, plot_df["Short Label"])
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1


def plot_training_time_per_stage(training_time_df: pd.DataFrame, figsize=(10, 6)):
    global FIGURE_NUMBER
    plot_df = _with_short_labels(training_time_df)
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=plot_df,
        x="Stage",
        y="Training Time (s)",
        hue="Short Label",
        palette=_build_single_color_palette(plot_df["Short Label"]),
    )
    plt.title(f"Figure {FIGURE_NUMBER}: Training Time per Stage")
    plt.xlabel("Training Stage")
    plt.ylabel("Training Time (s)")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    _apply_short_legend(ax, plot_df["Short Label"])
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1


def plot_metric_scatter(
    comparison_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    size_metric: str | None = None,
    figsize=(10, 6),
):
    global FIGURE_NUMBER
    plot_df = _with_short_labels(comparison_df)

    plt.figure(figsize=figsize)
    scatter_kwargs = {
        "data": plot_df,
        "x": x_metric,
        "y": y_metric,
        "color": PLOT_BLUE,
        "legend": False,
    }
    if size_metric is not None and size_metric in comparison_df.columns:
        scatter_kwargs["size"] = size_metric
        scatter_kwargs["sizes"] = (100, 500)

    ax = sns.scatterplot(**scatter_kwargs)
    for _, row in plot_df.iterrows():
        ax.text(row[x_metric], row[y_metric], f" {row['Short Label']}", va="center")

    plt.title(f"Figure {FIGURE_NUMBER}: {y_metric} vs {x_metric}")
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.grid(True, linestyle="--", alpha=0.35)
    _apply_short_legend(ax, plot_df["Short Label"])
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1


def style_summary_table(comparison_df: pd.DataFrame):
    preferred_columns = [
        "Model",
        "Source File",
        "AvgAcc",
        "AvgAcc Macro F1",
        "AvgAcc Micro F1",
        "AvgAcc Weighted F1",
        "Backward Transfer",
        "Backward Transfer Macro F1",
        "Backward Transfer Micro F1",
        "Backward Transfer Weighted F1",
        "Forward Transfer",
        "Forward Transfer Macro F1",
        "Forward Transfer Micro F1",
        "Forward Transfer Weighted F1",
        "Average Forgetting",
        "Average Forgetting Macro F1",
        "Average Forgetting Micro F1",
        "Average Forgetting Weighted F1",
        "Full Validation F1",
        "Full Validation Macro F1",
        "Full Validation Micro F1",
        "Full Validation Weighted F1",
        "Full Validation Loss",
        "Baseline Accuracy",
        "Total Compute Time (s)",
        "Num Parameters",
    ]
    columns = [column for column in preferred_columns if column in comparison_df.columns]
    summary_df = pd.DataFrame(comparison_df[columns]).copy()
    if {"AvgAcc", "Full Validation F1"}.issubset(summary_df.columns):
        summary_df = pd.DataFrame(
            summary_df.sort_values(
                by=["AvgAcc", "Full Validation F1"],
                ascending=False,
            ).reset_index(drop=True)
        )

    formatters = {
        "AvgAcc": "{:.4f}",
        "AvgAcc Macro F1": "{:.4f}",
        "AvgAcc Micro F1": "{:.4f}",
        "AvgAcc Weighted F1": "{:.4f}",
        "Backward Transfer": "{:.4f}",
        "Backward Transfer Macro F1": "{:.4f}",
        "Backward Transfer Micro F1": "{:.4f}",
        "Backward Transfer Weighted F1": "{:.4f}",
        "Forward Transfer": "{:.4f}",
        "Forward Transfer Macro F1": "{:.4f}",
        "Forward Transfer Micro F1": "{:.4f}",
        "Forward Transfer Weighted F1": "{:.4f}",
        "Average Forgetting": "{:.4f}",
        "Average Forgetting Macro F1": "{:.4f}",
        "Average Forgetting Micro F1": "{:.4f}",
        "Average Forgetting Weighted F1": "{:.4f}",
        "Full Validation F1": "{:.4f}",
        "Full Validation Macro F1": "{:.4f}",
        "Full Validation Micro F1": "{:.4f}",
        "Full Validation Weighted F1": "{:.4f}",
        "Full Validation Loss": "{:.4f}",
        "Baseline Accuracy": "{:.4f}",
        "Total Compute Time (s)": "{:.1f}",
        "Num Parameters": "{:,.0f}",
    }
    active_formatters: dict[str, Any] = {
        key: value for key, value in formatters.items() if key in summary_df.columns
    }
    return summary_df.style.format(active_formatters)

def plot_frequencies(freq_list: list, xlabel, ylabel=None, title=None):
    global FIGURE_NUMBER
    plt.figure(figsize=(10,6))
    sns.barplot(x=range(len(freq_list)), y=freq_list, color="C0", alpha=0.8)
    plt.title(f"Figure {FIGURE_NUMBER}: {title}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel or "Frequencies")
    plt.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1

def visualize_images(
    image_df: pd.DataFrame, img_dir: str | Path,
    classes: pd.DataFrame | list, class_names: pd.DataFrame,
    dataset_name: str, nrows=3, ncols=5, figsize=(15, 9),
    fontsize_scaler=1
):
    global FIGURE_NUMBER
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, class_id in enumerate(classes):
        img_path = image_df[image_df.ClassId == class_id]['Path'].values[0]
        full_img_path = img_dir / img_path

        img = Image.open(full_img_path)

        # Get the descriptive name
        class_name_match = class_names[class_names.ClassId == class_id]['SignName'].values
        class_name = class_name_match[0] if len(class_name_match) > 0 else f"Class {class_id}"

        axes[i].imshow(img)
        axes[i].set_title(f"Class {class_id}:\n{class_name}", fontsize=14*fontsize_scaler)
        axes[i].axis('off')

    axes[-1].axis('off')
    axes[-2].axis('off')

    plt.suptitle(f"Figure {FIGURE_NUMBER}: Images from {dataset_name}", fontsize=20*fontsize_scaler)
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1

def visualize_image(
    image_df: pd.DataFrame, img_dir: str | Path,
    class_id: pd.DataFrame | list, class_name: str,
    dataset_name: str, figsize=(15, 9), fontsize_scaler=1
):
    global FIGURE_NUMBER
    plt.figure(figsize=figsize)
    img_path = image_df[image_df.ClassId == class_id]['Path'].values[0]
    full_img_path = img_dir / img_path
    img = Image.open(full_img_path)

    plt.imshow(img)
    plt.title(f"Class {class_id}:\n{class_name}", fontsize=14*fontsize_scaler)
    plt.axis('off')

    plt.suptitle(f"Figure {FIGURE_NUMBER}: Images from {dataset_name}", fontsize=20*fontsize_scaler)
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1

def plot_gs_results(model_df: pd.DataFrame):
    global FIGURE_NUMBER
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    fig.suptitle(f'Figure {FIGURE_NUMBER}: CIL Hyperparameter Grid Search: Accuracy Analysis', fontsize=18)

    sns.pointplot(data=model_df, x='Base Learning Rate', y='AvgAcc', hue='Model', ax=axes[0])
    axes[0].set_title('AvgAcc vs Base Learning Rate')
    axes[0].legend(loc='upper left', title='Model')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    sns.pointplot(data=model_df, x='Exemplar Ratio', y='AvgAcc', hue='Model', ax=axes[1])
    axes[1].set_title('AvgAcc vs Exemplar Ratio')
    axes[1].legend(loc='upper left', title='Model')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    sns.pointplot(data=model_df, x='Class Masking', y='AvgAcc', hue='Model', ax=axes[2])
    axes[2].set_title('AvgAcc vs Class Masking')
    axes[2].legend(loc='upper left', title='Model')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1

def plot_baseline_vs_avgacc(model_df: pd.DataFrame):
    global FIGURE_NUMBER
    best_idx = model_df.groupby('Model')['AvgAcc'].idxmax()
    best_models = pd.DataFrame(model_df.loc[best_idx, :]).copy()

    melted = pd.melt(best_models, id_vars=['Model'],
                     value_vars=['Baseline Accuracy', 'AvgAcc'],
                     var_name='Metric', value_name='Accuracy')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x='Model', y='Accuracy', hue='Metric')

    plt.title(f'Figure {FIGURE_NUMBER}: Baseline Accuracy (Task 1) vs. Average Accuracy (Task N)', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1

def plot_loss_history_per_task(loss_history_per_task):
    global FIGURE_NUMBER
    plt.figure(figsize=(12, 6))

    for task_idx, task_losses in enumerate(loss_history_per_task, start=1):
        loss_series = pd.Series(task_losses)
        plotted = loss_series.rolling(25, min_periods=1).mean()
        label = f"Task {task_idx}"
        steps = np.arange(1, len(plotted) + 1)
        plt.plot(steps, plotted, label=label, linewidth=2)

    plt.title(f"Figure {FIGURE_NUMBER}: Training Loss by Task")
    plt.xlabel("Training Batch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1

def display_table_metrics(model_metrics: dict):
    global FIGURE_NUMBER
    res = []
    model_names = []
    for model_name, metrics in model_metrics.items():
        model_names.append(model_name)
        expert_metrics_dict = metrics['Expert Metrics']
        ece = metrics['ECE']
        cost = metrics['Cost Proxy']
        row = {
            'Model Name': model_name,
            'E[C]': expert_metrics_dict['Expected Expert Calls'],
            'ECE': ece,
            'Cost Proxy': cost,
        }
        res.append(row)
    print(f"Figure {FIGURE_NUMBER}: Model Metrics Table: {' vs '.join(model_names)}")
    FIGURE_NUMBER += 1
    return pd.DataFrame(res)

# Google AI helped implement these fully
def plot_grid_heatmap(df, metric_name):
    global FIGURE_NUMBER
    df = df.copy()
    df['Condition'] = df.apply(lambda x: f"LoRA:{x['LoRA']}\nCal:{x['Calibration']}", axis=1)
    pivot_mean = df.pivot(index="Condition", columns="Top-k", values=metric_name)

    prefix = "AvgAcc" if "Acc" in metric_name else "Fbar"
    pivot_low = df.pivot(index="Condition", columns="Top-k", values=f"{prefix} 95% CI Low")
    pivot_high = df.pivot(index="Condition", columns="Top-k", values=f"{prefix} 95% CI High")

    error_margin = (pivot_high - pivot_low) / 2
    annot_text = np.array([[f"{m:.3f}\n±{e:.3f}" for m, e in zip(row_m, row_e)]
                           for row_m, row_e in zip(pivot_mean.values, error_margin.values)])

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_mean, annot=annot_text, fmt="", cmap="YlGnBu", cbar_kws={'label': metric_name})
    plt.title(f"Figure {FIGURE_NUMBER}: {metric_name} Across Configurations (95% CI)")
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1

def plot_tradeoff_scatter(df):
    global FIGURE_NUMBER
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, x='E[C]', y='AvgAcc',
        sizes=(50, 200)
    )
    plt.title(f"Figure {FIGURE_NUMBER}: Accuracy vs. Expected Expert Calls (E[C])")
    plt.xticks([1,2,3])
    plt.xlabel("Expected Expert Calls (Compute Cost)")
    plt.ylabel("Average Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1

def plot_reliability_comparison(logits_unscaled, logits_scaled, targets, n_bins=10):
    global FIGURE_NUMBER
    def get_calibration_stats(logits):
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        global_acc = (pred == targets).float().mean().item()
        bin_edges = torch.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        bin_weights = []
        
        for i in range(n_bins):
            if i == 0:
                mask = (conf >= bin_edges[i]) & (conf <= bin_edges[i+1])
            else:
                mask = (conf > bin_edges[i]) & (conf <= bin_edges[i+1])
            if mask.any():
                bin_accs.append( (pred[mask] == targets[mask]).float().mean().item() )
                bin_confs.append( conf[mask].mean().item() )
                bin_weights.append( mask.float().mean().item() ) # Proportion of samples in this bin
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.0)
                bin_weights.append(0.0)
                
        ece = sum([w * abs(a - c) for w, a, c in zip(bin_weights, bin_accs, bin_confs)])
        return bin_accs, global_acc, ece

    bin_centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)
    acc_unscaled, global_acc_unscaled, ece_unscaled = get_calibration_stats(logits_unscaled)
    acc_scaled, global_acc_scaled, ece_scaled = get_calibration_stats(logits_scaled)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly Calibrated")
    bar_width = 0.8 / n_bins
    plt.bar(bin_centers - bar_width/2, acc_unscaled, width=bar_width, 
            label="Unscaled", alpha=0.7, color='tab:red')
    plt.bar(bin_centers + bar_width/2, acc_scaled, width=bar_width, 
            label="Scaled", alpha=0.7, color='tab:blue')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Figure {FIGURE_NUMBER}: Reliability Diagram Comparison")
    plt.tight_layout()
    plt.show()
    plt.close()
    FIGURE_NUMBER += 1