import gc
import copy
import json
import time
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal, overload
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import KLDivLoss
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from statsmodels.stats.multitest import multipletests

# =========================================================================================    
# Class-Incremental Implementation
# =========================================================================================


@dataclass
class TrainingThroughput:
    """Optimizer-only work performed while training one or more CIL tasks."""

    total_samples: int = 0
    total_optimizer_steps: int = 0
    timed_samples: int = 0
    timed_optimizer_steps: int = 0
    optimization_seconds: float = 0.0

    def record(self, batch_size: int, elapsed_seconds: float | None) -> None:
        self.total_samples += batch_size
        self.total_optimizer_steps += 1
        if elapsed_seconds is not None:
            self.timed_samples += batch_size
            self.timed_optimizer_steps += 1
            self.optimization_seconds += elapsed_seconds

    def to_dict(self) -> dict[str, float | int]:
        return {
            "Training Samples": self.total_samples,
            "Training Optimizer Steps": self.total_optimizer_steps,
            "Timed Training Samples": self.timed_samples,
            "Timed Optimizer Steps": self.timed_optimizer_steps,
            "Measured Optimization Time (s)": self.optimization_seconds,
            "Training Throughput (samples/s)": (
                self.timed_samples / self.optimization_seconds
                if self.optimization_seconds > 0
                else float("nan")
            ),
        }


def _synchronize_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

class CILComputerVisionModel:
    """
    Class-incremental Learning computer vision model wrapper class.
    Provides any model the ability to include the exemplar set replay and knowledge distillation loss.

    - Exemplar Set Replay randomly stores a subset of old task data and replays
        it alongside new task data to enforce the network's old geometries/patterns.
    - Class Masking zeroes out the logits for classes not yet seen to stop standard
        cross-entropy from suppressing untouched outputs.
    - Knowledge Distillation Loss (Learning without Forgetting) freezes the model
        weights prior to learning a new task. While learning, the model minimizes the
        loss function for the current task and the Kullback-Leibler divergence
        between its current output and it's previous output across seen tasks.
    """
    def __init__(
        self, model: nn.Module, exemplar_ratio: float, rng=None, device='cpu',
        use_class_masking=False, kd_temperature=2.0, lambda_kd=0.5
    ):
        self.device = device
        self.model = model.to(device)
        self.prev_model = None
        self.rng = rng or np.random.default_rng()

        self.exemplar_ratio = exemplar_ratio
        self.exemplar_set = None

        self.use_class_masking = use_class_masking
        self.seen_classes = set()
        self.seen_classes_tensor = None
        self.prev_seen_classes_tensor = None

        self.kd_temperature = kd_temperature
        self.lambda_kd = lambda_kd

    def _mask_logits(self, logits):
        if not self.use_class_masking or self.seen_classes_tensor is None:
            return logits

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, self.seen_classes_tensor] = False
        return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

    def _get_model_aux_loss(self):
        # If this is a MoE model and has aux loss, return aux loss
        if hasattr(self.model, "get_auxiliary_loss"):
            aux_loss = self.model.get_auxiliary_loss()
            if aux_loss is not None:
                return aux_loss

        # Otherwise, return a zero tensor
        logits_dtype = next(self.model.parameters()).dtype
        return torch.zeros((), device=self.device, dtype=logits_dtype)

    def _update_seen_classes(self, train_dataloader):
        current_task_classes = set()
        for _, labels, _ in train_dataloader:
            current_task_classes.update(l["label"].item() for l in labels)

        if self.seen_classes:
            self.prev_seen_classes_tensor = torch.tensor(
                sorted(self.seen_classes),
                dtype=torch.long,
                device=self.device,
            )
        else:
            self.prev_seen_classes_tensor = None

        self.seen_classes.update(current_task_classes)
        self.seen_classes_tensor = torch.tensor(
            sorted(self.seen_classes),
            dtype=torch.long,
            device=self.device,
        )

    def _build_task_dataloader(self, train_dataloader):
        if self.exemplar_set is None:
            return train_dataloader

        combined_dataset = train_dataloader.dataset
        combined_dataset = torch.utils.data.ConcatDataset(
            [combined_dataset, self.exemplar_set]
        )
        return DataLoader(
            combined_dataset,
            batch_size=train_dataloader.batch_size,
            shuffle=True,
            collate_fn=train_dataloader.collate_fn,
            num_workers=getattr(train_dataloader, "num_workers", 0),
            pin_memory=getattr(train_dataloader, "pin_memory", False),
            persistent_workers=getattr(train_dataloader, "persistent_workers", False),
            generator=getattr(train_dataloader, "generator", None),
        )

    def get_exemplar_count(self):
        return 0 if self.exemplar_set is None else len(self.exemplar_set)

    def loss(self, loss_fn, images, target):
        new_pred = self.model(images)
        loss_pred = loss_fn(self._mask_logits(new_pred), target)
        aux_loss = self._get_model_aux_loss()

        if self.prev_model is None:
            return loss_pred + aux_loss

        with torch.no_grad():
            old_pred = self.prev_model(images)

        kd_idx = self.prev_seen_classes_tensor
        new_pred_kd = new_pred[:, kd_idx]
        old_pred_kd = old_pred[:, kd_idx]

        loss_kd = KLDivLoss(reduction="batchmean")(
            F.log_softmax(new_pred_kd / self.kd_temperature, dim=1),
            F.softmax(old_pred_kd / self.kd_temperature, dim=1),
        ) * (self.kd_temperature * self.kd_temperature)

        return (1 - self.lambda_kd) * loss_pred + self.lambda_kd * loss_kd + aux_loss

    def train(
        self,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        loss_fn,
        epochs: int,
        verbose: bool,
        *,
        throughput_warmup_batches: int = 0,
        return_measurement: bool = False,
    ):
        self.model.train()
        loss_history = []

        if verbose:
            print("\rUpdating seen classes...", end="", flush=True)
        self._update_seen_classes(train_dataloader)

        if verbose:
            print(f"\rSeen classes updated to {self.seen_classes}")
            print("\rBuilding new task dataloader...", end="", flush=True)
        task_dataloader = self._build_task_dataloader(train_dataloader)
        if verbose:
            print(f"\rTask dataloarder ready: {len(task_dataloader.dataset)} images in task.")

        measurement = TrainingThroughput()
        batch_number = 0
        for epoch in range(epochs):
            progress_bar = task_dataloader
            if verbose:
                progress_bar = tqdm(task_dataloader, desc=f"Epoch {epoch+1}")

            for images, labels, _ in progress_bar:
                images = torch.stack(images).to(self.device)
                y_true = torch.tensor(
                    [l["label"] for l in labels],
                    dtype=torch.long,
                    device=self.device,
                )

                measure_batch = batch_number >= throughput_warmup_batches
                start_time = 0.0
                if return_measurement and measure_batch:
                    _synchronize_if_cuda(self.device)
                    start_time = perf_counter()

                optimizer.zero_grad(set_to_none=True)
                loss_val = self.loss(loss_fn, images, y_true)
                loss_val.backward()
                optimizer.step()

                elapsed_seconds = None
                if return_measurement and measure_batch:
                    _synchronize_if_cuda(self.device)
                    elapsed_seconds = perf_counter() - start_time
                measurement.record(batch_size=len(y_true), elapsed_seconds=elapsed_seconds)
                batch_number += 1

                if hasattr(self.model, "reset_routing_state"):
                    # Reset GRE state
                    self.model.reset_routing_state()

                loss_history.append(loss_val.detach().item())

                if verbose:
                    progress_bar.set_postfix({"loss": f"{loss_val.detach().item():.4f}"})

        task_data_size = len(train_dataloader.dataset)
        task_exemplar_size = int(self.exemplar_ratio * task_data_size)

        if task_exemplar_size > 0:
            sample_indices = self.rng.choice(
                task_data_size,
                size=task_exemplar_size,
                replace=False,
            )
            sampled_exemplars = torch.utils.data.Subset(
                train_dataloader.dataset,
                sample_indices.tolist(),
            )

            if self.exemplar_set is None:
                self.exemplar_set = sampled_exemplars
            else:
                self.exemplar_set = torch.utils.data.ConcatDataset(
                    [self.exemplar_set, sampled_exemplars]
                )

        if hasattr(self.model, "reset_routing_state"):
            # Reset GRE state to prevent copy errors
            self.model.reset_routing_state()

        self.prev_model = copy.deepcopy(self.model).to(self.device)
        self.prev_model.eval()
        for param in self.prev_model.parameters():
            param.requires_grad = False

        if return_measurement:
            return loss_history, measurement
        return loss_history

    @overload
    def evaluate(
        self,
        test_dataloader: DataLoader,
        loss_fn,
        return_logits: Literal[False] = False,
        apply_class_masking: bool = True,
    ) -> tuple[float, dict[str, float]]:
        ...

    @overload
    def evaluate(
        self,
        test_dataloader: DataLoader,
        loss_fn,
        return_logits: Literal[True],
        apply_class_masking: bool = True,
    ) -> tuple[float, dict[str, float], np.ndarray, list[int], list[str]]:
        ...

    def evaluate(
        self,
        test_dataloader: DataLoader,
        loss_fn,
        return_logits=False,
        apply_class_masking=True,
    ):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_logits = [] if return_logits else None
        all_paths = [] if return_logits else None

        with torch.no_grad():
            for images, labels, paths in test_dataloader:
                images = torch.stack(images).to(self.device)
                y_true = torch.tensor(
                    [l["label"] for l in labels],
                    dtype=torch.long,
                    device=self.device,
                )

                y_pred_logits = self.model(images)
                eval_logits = (
                    self._mask_logits(y_pred_logits)
                    if apply_class_masking
                    else y_pred_logits
                )

                loss_val = loss_fn(eval_logits, y_true)
                total_loss += loss_val.item()

                y_preds = torch.argmax(eval_logits, dim=1)
                all_preds.extend(y_preds.cpu().numpy())
                all_targets.extend(y_true.cpu().numpy())
                if return_logits:
                    assert all_logits is not None and all_paths is not None
                    all_logits.append(y_pred_logits.cpu().numpy())
                    all_paths.extend(paths)

        metrics = {
            "Micro F1": f1_score(all_targets, all_preds, average="micro", zero_division=0),
            "Macro F1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
            "Weighted F1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        }
        if return_logits:
            assert all_logits is not None and all_paths is not None
            all_logits = np.concatenate(all_logits, axis=0)
            return total_loss / len(test_dataloader), metrics, all_logits, all_targets, all_paths
        return total_loss / len(test_dataloader), metrics

    def save(self, filename='cil_model_checkpoint.pth') -> None:
        checkpoint = {
            'model_state': self.model.state_dict(),
            'prev_model_state': self.prev_model.state_dict() if self.prev_model is not None else None,
            'exemplar_ratio': self.exemplar_ratio,
            'exemplar_set': self.exemplar_set,
            'use_class_masking': self.use_class_masking,
            'seen_classes': self.seen_classes,
            'seen_classes_tensor': self.seen_classes_tensor,
            'prev_seen_classes_tensor': self.prev_seen_classes_tensor,
            'kd_temperature': self.kd_temperature,
            'lambda_kd': self.lambda_kd,
            'device': self.device
        }
        torch.save(checkpoint, filename)

    @classmethod
    def load(
        cls,
        model_instance: nn.Module,
        filename='cil_model_checkpoint.pth',
        map_location=None,
        device=None,
    ):
        if map_location is None and device is not None:
            map_location = device

        checkpoint = torch.load(filename, weights_only=False, map_location=map_location)

        target_device = device if device is not None else checkpoint['device']

        cil_object = cls(
            model=model_instance,
            exemplar_ratio=checkpoint['exemplar_ratio'],
            use_class_masking=checkpoint['use_class_masking'],
            device=target_device
        )

        cil_object.model.load_state_dict(checkpoint['model_state'])

        if checkpoint['prev_model_state'] is not None:
            cil_object.prev_model = copy.deepcopy(model_instance).to(cil_object.device)
            cil_object.prev_model.load_state_dict(checkpoint['prev_model_state'])
            cil_object.prev_model.requires_grad_(False)

        cil_object.seen_classes         = checkpoint['seen_classes']
        cil_object.seen_classes_tensor  = checkpoint['seen_classes_tensor']
        cil_object.prev_seen_classes_tensor = checkpoint['prev_seen_classes_tensor']
        cil_object.kd_temperature       = checkpoint['kd_temperature']
        cil_object.lambda_kd            = checkpoint['lambda_kd']
        cil_object.exemplar_set         = checkpoint['exemplar_set']

        return cil_object
    
# =========================================================================================    
# Continual Learning Metrics
# =========================================================================================
    
def average_accuracy(eval_matrix: np.ndarray):
    # final average accuracy across tasks
    return np.mean(eval_matrix[:, -1])

def task_forgetting(eval_matrix: np.ndarray):
    # catastrophic forgetting per training stage
    max_accs = np.max(eval_matrix[:-1, :-1], axis=1)
    final_accs = eval_matrix[:-1, -1]
    return max_accs - final_accs

def average_forgetting(F_i: np.ndarray):
    return np.mean(F_i)


def backward_transfer(eval_matrix: np.ndarray):
    """
    Average backward transfer (BWT).

    - Positive BWT means later learning improved earlier tasks.
    - Negative BWT means later learning hurt earlier tasks.
    """
    num_tasks = eval_matrix.shape[0]
    if num_tasks < 2:
        return 0.0

    initial_scores = np.diag(eval_matrix)[:-1]
    final_scores = eval_matrix[:-1, -1]
    return np.mean(final_scores - initial_scores)

def backward_transfer_per_task(eval_matrix: np.ndarray):
    """
    Per-task backward transfer for all but the final task.
    """
    num_tasks = eval_matrix.shape[0]
    if num_tasks < 2:
        return np.array([])

    initial_scores = np.diag(eval_matrix)[:-1]
    final_scores = eval_matrix[:-1, -1]
    return final_scores - initial_scores


def forward_transfer(eval_matrix: np.ndarray, reference_scores: np.ndarray):
    """
    Average forward transfer (FWT).

    Uses the model's score on task t immediately before training on task t,
    compared against a reference baseline for that same task.

    Expected matrix convention:
        eval_matrix[t, t-1] = score on task t after training tasks 0..t-1
    reference_scores[t] = baseline score for task t
        e.g. untrained model score, random-init score, or another agreed reference.
    """
    num_tasks = eval_matrix.shape[0]
    if num_tasks < 2:
        return 0.0

    pre_task_scores = eval_matrix[np.arange(1, num_tasks), np.arange(0, num_tasks - 1)]
    baseline_scores = np.asarray(reference_scores)[1:]

    return np.mean(pre_task_scores - baseline_scores)

def forward_transfer_per_task(eval_matrix: np.ndarray, reference_scores: np.ndarray):
    num_tasks = eval_matrix.shape[0]
    if num_tasks < 2:
        return np.array([])

    pre_task_scores = eval_matrix[np.arange(1, num_tasks), np.arange(0, num_tasks - 1)]
    baseline_scores = np.asarray(reference_scores)[1:]

    return pre_task_scores - baseline_scores


def summarize_continual_metric_results(
    eval_matrix: np.ndarray,
    forward_eval_matrix: np.ndarray,
    baseline_forward_scores: np.ndarray,
    suffix: str = "",
):
    avg_acc = average_accuracy(eval_matrix)
    forgetting_list = task_forgetting(eval_matrix)
    avg_forgetting = average_forgetting(forgetting_list)
    avg_bwt = backward_transfer(eval_matrix)
    bwt_per_task = backward_transfer_per_task(eval_matrix)
    avg_fwt = forward_transfer(forward_eval_matrix, baseline_forward_scores)
    fwt_per_task = forward_transfer_per_task(forward_eval_matrix, baseline_forward_scores)

    metric_suffix = f" {suffix}" if suffix else ""
    return {
        f"AvgAcc{metric_suffix}": avg_acc,
        f"Backward Transfer{metric_suffix}": avg_bwt,
        f"Backward Transfer Per Task{metric_suffix}": bwt_per_task,
        f"Forward Transfer{metric_suffix}": avg_fwt,
        f"Forward Transfer Per Task{metric_suffix}": fwt_per_task,
        f"Task Forgetting List{metric_suffix}": forgetting_list,
        f"Average Forgetting{metric_suffix}": avg_forgetting,
        f"Eval Matrix{metric_suffix}": eval_matrix,
    }
    
# =========================================================================================    
# Continual Learning Statistical Testing
# =========================================================================================

def bootstrap_learning(preds, num_tasks, rng, resamples=1000):
    avg_acc_samples = []
    avg_f_samples = []

    for b in range(resamples):
        eval_mat = np.zeros((num_tasks, num_tasks))

        for t in range(num_tasks):
            y_true_ref = np.array(preds.get((t, t), [None, []])[1])
            n_samples = len(y_true_ref)
            indices = rng.choice(n_samples, n_samples, replace=True)

            for i in range(t, num_tasks):
                y_pred = np.argmax(np.array(preds[(i, t)][0]), axis=1)
                y_true = np.array(preds[(i, t)][1])
                f1 = f1_score(y_true[indices], y_pred[indices], average='macro', zero_division=0)
                eval_mat[t, i] = f1

        avg_acc_samples.append(average_accuracy(eval_mat))
        F_i = task_forgetting(eval_mat)
        avg_f_samples.append(average_forgetting(F_i))

    if not avg_acc_samples:
        return {}

    return {
        "AvgAcc Macro F1": [np.mean(avg_acc_samples)],
        "AAMF1 95% CI Low": [np.percentile(avg_acc_samples, 2.5)],
        "AAMF1 95% CI High": [np.percentile(avg_acc_samples, 97.5)],
        "Fbar": [np.mean(avg_f_samples)],
        "Fbar 95% CI Low": [np.percentile(avg_f_samples, 2.5)],
        "Fbar 95% CI High": [np.percentile(avg_f_samples, 97.5)]
    }
    
def bootstrap_learning_diff(model_logits: dict, num_tasks, rng, resamples=1000):
    avg_acc_delta_samples = []
    avg_f_delta_samples = []
    
    names, logits = zip(*model_logits.items())
    model_a_dict, model_b_dict = logits[0], logits[1]

    for b in range(resamples):
        # Bootstrap sample
        eval_mat_A = np.zeros((num_tasks, num_tasks))
        eval_mat_B = np.zeros((num_tasks, num_tasks))

        for t in range(num_tasks):
            # Sample for task
            y_true_ref = np.array(model_a_dict[(t, t)][1])
            n_samples = len(y_true_ref)
            indices = rng.choice(n_samples, n_samples, replace=True)

            for i in range(t, num_tasks):
                pred_A = np.argmax(np.array(model_a_dict[(i, t)][0]), axis=1)
                eval_mat_A[t, i] = f1_score(
                    y_true_ref[indices], pred_A[indices], 
                    average='macro', 
                    zero_division=0
                )
                pred_B = np.argmax(np.array(model_b_dict[(i, t)][0]), axis=1)
                eval_mat_B[t, i] = f1_score(
                    y_true_ref[indices], pred_B[indices], 
                    average='macro', 
                    zero_division=0
                )

        # Calculate sample deltas
        acc_A = average_accuracy(eval_mat_A)
        f_A = average_forgetting(task_forgetting(eval_mat_A))
        acc_B = average_accuracy(eval_mat_B)
        f_B = average_forgetting(task_forgetting(eval_mat_B))

        avg_acc_delta_samples.append(acc_A - acc_B)
        avg_f_delta_samples.append(f_A - f_B)

    avg_acc_delta_samples = np.array(avg_acc_delta_samples)
    avg_f_delta_samples   = np.array(avg_f_delta_samples)

    avg_acc_p_val = 2 * min((avg_acc_delta_samples <= 0).mean(), (avg_acc_delta_samples >= 0).mean())
    avg_f_p_val   = 2 * min((avg_f_delta_samples <= 0).mean(), (avg_f_delta_samples >= 0).mean())

    # Correct p-values for multiple comparisons
    #   This is wrong. The multiple tests needs to be a combination of the p-values:
    #   _, bh_acc_p_value, _, _ = multipletests([avg_acc_p_val, avg_f_p_val], method='fdr_bh')
    #       this will correct the possibility for higher false positives.
    _, bh_acc_p_value, _, _ = multipletests([avg_acc_p_val], method='fdr_bh')
    _, bh_f_p_value, _, _   = multipletests([avg_f_p_val], method='fdr_bh')

    acc_ci = np.percentile(avg_acc_delta_samples, [2.5, 97.5])
    f_ci = np.percentile(avg_f_delta_samples, [2.5, 97.5])

    return {
        "Model A": names[0],
        "Model B": names[1],
        "AvgAcc Delta (A-B)": (avg_acc_delta_samples).mean(),
        "AvgAcc 95% CI Low": acc_ci[0],
        "AvgAcc 95% CI High": acc_ci[1],
        "AvgAcc FDR BH P Value": bh_acc_p_value[0],
        "Fbar Delta (A-B)": (avg_f_delta_samples).mean(),
        "Fbar 95% CI Low": f_ci[0],
        "Fbar 95% CI High": f_ci[1],
        "Fbar FDR BH P Value": bh_f_p_value[0]
    }
    
def bootstrap_performance_diff(
    test_dataloader,
    models_dict,
    device,
    rng,
    resamples=10_000,
    average="macro",
    stratified=True,
):
    """
    Paired nonparametric bootstrap comparison of trained models on one
    held-out test set.

    The same resampled test examples are used for every model in each
    bootstrap replicate, preserving the paired prediction structure.
    """
    if resamples < 2:
        raise ValueError("resamples must be >= 2.")

    if len(models_dict) < 2:
        raise ValueError("At least two models are required.")

    if average not in {"macro", "micro", "weighted"}:
        raise ValueError(
            "average must be one of {'macro', 'micro', 'weighted'}."
        )

    # ---------------------------------------------------------------
    # 1. Run every model over exactly the same held-out examples.
    # ---------------------------------------------------------------

    all_targets = []
    model_preds = {
        name: []
        for name in models_dict
    }

    for model in models_dict.values():
        model.eval()

    with torch.no_grad():
        for images, labels, _ in test_dataloader:
            images = torch.stack(images).to(device)

            y_true = np.asarray(
                [label["label"] for label in labels],
                dtype=np.int64,
            )

            all_targets.extend(y_true.tolist())

            for name, model in models_dict.items():
                logits = model(images)
                predictions = torch.argmax(logits, dim=1)

                model_preds[name].extend(
                    predictions.cpu().numpy().tolist()
                )

                if hasattr(model, "reset_routing_state"):
                    model.reset_routing_state()

    all_targets = np.asarray(all_targets, dtype=np.int64)

    model_preds = {
        name: np.asarray(predictions, dtype=np.int64)
        for name, predictions in model_preds.items()
    }

    n_samples = len(all_targets)

    if n_samples == 0:
        raise ValueError("The test dataloader produced no samples.")

    for name, predictions in model_preds.items():
        if len(predictions) != n_samples:
            raise RuntimeError(
                f"{name!r} produced {len(predictions)} predictions "
                f"for {n_samples} targets."
            )

    # ---------------------------------------------------------------
    # 2. Observed test-set F1.
    # ---------------------------------------------------------------

    observed_f1 = {
        name: f1_score(
            all_targets,
            predictions,
            average=average,
            zero_division=0,
        )
        for name, predictions in model_preds.items()
    }

    # ---------------------------------------------------------------
    # 3. Prepare stratified bootstrap index groups.
    # ---------------------------------------------------------------

    if stratified:
        class_indices = [
            np.flatnonzero(all_targets == class_id)
            for class_id in np.unique(all_targets)
        ]

    bootstrap_f1 = {
        name: np.empty(resamples, dtype=np.float64)
        for name in models_dict
    }

    # ---------------------------------------------------------------
    # 4. Paired bootstrap.
    #
    # Critically: one `indices` vector is generated per replicate and
    # reused for every model.
    # ---------------------------------------------------------------

    for bootstrap_index in range(resamples):

        if stratified:
            indices = np.concatenate([
                rng.choice(
                    indices_for_class,
                    size=len(indices_for_class),
                    replace=True,
                )
                for indices_for_class in class_indices
            ])

            # Ordering is irrelevant to F1, but shuffling avoids leaving
            # every bootstrap replicate class-block ordered.
            rng.shuffle(indices)

        else:
            indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
            )

        y_true_b = all_targets[indices]

        for name, predictions in model_preds.items():
            y_pred_b = predictions[indices]

            bootstrap_f1[name][bootstrap_index] = f1_score(
                y_true_b,
                y_pred_b,
                average=average,
                zero_division=0,
            )

    # ---------------------------------------------------------------
    # 5. Per-model bootstrap summaries.
    # ---------------------------------------------------------------

    model_performance = []

    for name, samples in bootstrap_f1.items():
        ci_low, ci_high = np.percentile(
            samples,
            [2.5, 97.5],
        )

        model_performance.append({
            "Model": name,
            "Observed F1": observed_f1[name],
            "Bootstrap Mean F1": samples.mean(),
            "Bootstrap Std F1": samples.std(ddof=1),
            "95% CI Lower": ci_low,
            "95% CI Upper": ci_high,
            "F1 Average": average,
            "Bootstrap Resamples": resamples,
        })

    # ---------------------------------------------------------------
    # 6. Paired model differences.
    # ---------------------------------------------------------------

    pairwise_results = []

    for name_a, name_b in combinations(models_dict.keys(), 2):

        bootstrap_delta = (
            bootstrap_f1[name_a]
            - bootstrap_f1[name_b]
        )

        observed_delta = (
            observed_f1[name_a]
            - observed_f1[name_b]
        )

        ci_low, ci_high = np.percentile(
            bootstrap_delta,
            [2.5, 97.5],
        )

        # Supplemental bootstrap sign/tail probability.
        #
        # +1 correction prevents a reported value of exactly zero when
        # the number of bootstrap draws is finite.
        lower_tail = (
            np.count_nonzero(bootstrap_delta <= 0) + 1
        ) / (resamples + 1)

        upper_tail = (
            np.count_nonzero(bootstrap_delta >= 0) + 1
        ) / (resamples + 1)

        bootstrap_tail_probability = min(
            1.0,
            2.0 * min(lower_tail, upper_tail),
        )

        pairwise_results.append({
            "Model A": name_a,
            "Model B": name_b,
            "Observed Delta (A-B)": observed_delta,
            "Bootstrap Mean Delta": bootstrap_delta.mean(),
            "95% CI Lower": ci_low,
            "95% CI Upper": ci_high,
            "Bootstrap Tail Probability": bootstrap_tail_probability,
            "F1 Average": average,
            "Bootstrap Resamples": resamples,
        })

    # ---------------------------------------------------------------
    # 7. FDR correction over the COMPLETE pairwise family.
    # ---------------------------------------------------------------

    raw_tail_probabilities = np.asarray([
        row["Bootstrap Tail Probability"]
        for row in pairwise_results
    ])

    _, adjusted, _, _ = multipletests(
        raw_tail_probabilities,
        alpha=0.05,
        method="fdr_bh",
    )

    for row, adjusted_probability in zip(
        pairwise_results,
        adjusted,
    ):
        row["FDR-BH Adjusted Tail Probability"] = (
            float(adjusted_probability)
        )

        row["95% CI Excludes Zero"] = (
            row["95% CI Lower"] > 0
            or row["95% CI Upper"] < 0
        )

    return pairwise_results, model_performance
    
# =========================================================================================    
# Data Helpers
# =========================================================================================

def build_complete_dataloader(
    dataloaders: list[DataLoader],
    shuffle=False,
):
    if not dataloaders:
        raise ValueError("Expected at least one dataloader to build a complete dataset.")

    base_loader = dataloaders[0]
    complete_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in dataloaders])

    return DataLoader(
        complete_dataset,
        batch_size=base_loader.batch_size,
        collate_fn=base_loader.collate_fn,
        shuffle=shuffle,
        num_workers=getattr(base_loader, "num_workers", 0),
        pin_memory=getattr(base_loader, "pin_memory", False),
        persistent_workers=getattr(base_loader, "persistent_workers", False),
        generator=getattr(base_loader, "generator", None),
    )

def create_task_dataloaders(
    dataset: Dataset, task_class_list: list[list[int]],
    batch_size: int, collate_fn: Callable, generator=None,
    num_workers=0, pin_memory=False, persist_workers=False
):
    dataloaders = []

    for task_classes in task_class_list:
        task_indices = [i for i, target in enumerate(dataset.targets) if target in task_classes]
        task_subset = torch.utils.data.Subset(dataset, task_indices)

        dataloaders.append(
            DataLoader(
                task_subset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persist_workers,
                generator=generator
            )
        )
    return dataloaders

# =========================================================================================    
# Model Building
# =========================================================================================

def train_and_evaluate_cil_model(
    cil_model: CILComputerVisionModel, optimizer_cls, loss_fn, 
    train_dataloaders: list[DataLoader], val_dataloaders: list[DataLoader],
    epochs=5, num_tasks=5, base_lr=0.001, later_task_lr_factor=0.1,
    device='cpu', verbose=False, throughput_warmup_batches=0,
):
    model = cil_model.model.to(device)
    eval_matrices = {
        "Micro F1": np.zeros((num_tasks, num_tasks))
    }
    forward_eval_matrices = {
        "Micro F1": np.full((num_tasks, num_tasks), np.nan)
    }
    baseline_forward_scores = {
        "Micro F1": np.zeros(num_tasks)
    }
    task_throughput = []
    model_name = model.__class__.__name__ if hasattr(model, '__class__') else 'Unknown'
    loss_history_per_task = []

    full_val_dataloader = build_complete_dataloader(val_dataloaders, shuffle=False)

    # Baseline performance before any task training for forward transfer.
    for t in range(num_tasks):
        _, baseline_metrics = cil_model.evaluate(
            val_dataloaders[t],
            loss_fn,
            apply_class_masking=False,
        )
        baseline_forward_scores["Micro F1"][t] = baseline_metrics["Micro F1"]
    
    # Iterate pseudo-time training tasks
    for i in range(num_tasks):
        if verbose:
            print("=======================================================")
            print(f"Stage {i+1}/{num_tasks}: Model {model_name}")
            print("=======================================================")
        else:
            print(
                f"\rStage {i+1}/{num_tasks}: Model {model_name}",
                end="",
                flush=True
            )

        if i > 0:
            _, pre_task_metrics = cil_model.evaluate(
                val_dataloaders[i],
                loss_fn,
                apply_class_masking=False,
            )
            forward_eval_matrices["Micro F1"][i, i - 1] = pre_task_metrics["Micro F1"]

        current_lr = base_lr if i == 0 else base_lr * later_task_lr_factor
        optimizer = optimizer_cls(cil_model.model.parameters(), lr=current_lr)

        # Train. Throughput intentionally excludes data loading, validation, and I/O.
        task_lost_history, task_measurement = cil_model.train(
            train_dataloaders[i], optimizer, loss_fn,
            epochs,
            verbose=verbose,
            throughput_warmup_batches=throughput_warmup_batches,
            return_measurement=True,
        )

        task_throughput.append(task_measurement)
        loss_history_per_task.append(task_lost_history)

        # eval each task
        for t in range(i + 1):
            _, val_metrics = cil_model.evaluate(
                val_dataloaders[t],
                loss_fn,
            )
            eval_matrices["Micro F1"][t, i] = val_metrics["Micro F1"]
            if verbose: print(f"    Task {t+1}: Micro F1 = {val_metrics['Micro F1']:.4f}")
        if verbose: print(f"    Exemplar set size = {cil_model.get_exemplar_count()}")

    aggregate_throughput = TrainingThroughput()
    for measurement in task_throughput:
        aggregate_throughput.total_samples += measurement.total_samples
        aggregate_throughput.total_optimizer_steps += measurement.total_optimizer_steps
        aggregate_throughput.timed_samples += measurement.timed_samples
        aggregate_throughput.timed_optimizer_steps += measurement.timed_optimizer_steps
        aggregate_throughput.optimization_seconds += measurement.optimization_seconds

    micro_results = summarize_continual_metric_results(
        eval_matrix=eval_matrices["Micro F1"],
        forward_eval_matrix=forward_eval_matrices["Micro F1"],
        baseline_forward_scores=baseline_forward_scores["Micro F1"],
        suffix="Micro F1"
    )
    full_val_loss, full_val_metrics = cil_model.evaluate(full_val_dataloader, loss_fn)

    results = {
        **micro_results,
        'Full Validation Loss': full_val_loss,
        'Full Validation Macro F1': full_val_metrics['Macro F1'],
        'Full Validation Micro F1': full_val_metrics['Micro F1'],
        'Full Validation Micro F1': full_val_metrics['Weighted F1'],
        'Training Throughput per Stage (samples/s)': [
            measurement.to_dict()['Training Throughput (samples/s)']
            for measurement in task_throughput
        ],
        'Training Samples per Stage': [measurement.total_samples for measurement in task_throughput],
        'Timed Training Samples per Stage': [measurement.timed_samples for measurement in task_throughput],
        **aggregate_throughput.to_dict(),
    }
    
    # Cleanup memory to prevent memory eating
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return results, loss_history_per_task


def train_final_cil_model(
    cil_model: CILComputerVisionModel,
    optimizer_cls,
    loss_fn,
    train_dataloaders: list[DataLoader],
    epochs: int = 5,
    base_lr: float = 0.001,
    later_task_lr_factor: float = 0.1,
    verbose: bool = False,
    throughput_warmup_batches: int = 0,
) -> tuple[dict[str, Any], list[list[float]]]:
    """Train a final CIL replica on all source training data without validation."""
    task_throughput: list[TrainingThroughput] = []
    loss_history_per_task: list[list[float]] = []

    for task_index, train_dataloader in enumerate(train_dataloaders):
        current_lr = base_lr if task_index == 0 else base_lr * later_task_lr_factor
        optimizer = optimizer_cls(cil_model.model.parameters(), lr=current_lr)
        task_loss_history, task_measurement = cil_model.train(
            train_dataloader,
            optimizer,
            loss_fn,
            epochs,
            verbose=verbose,
            throughput_warmup_batches=throughput_warmup_batches,
            return_measurement=True,
        )
        task_throughput.append(task_measurement)
        loss_history_per_task.append(task_loss_history)

    aggregate_throughput = TrainingThroughput()
    for measurement in task_throughput:
        aggregate_throughput.total_samples += measurement.total_samples
        aggregate_throughput.total_optimizer_steps += measurement.total_optimizer_steps
        aggregate_throughput.timed_samples += measurement.timed_samples
        aggregate_throughput.timed_optimizer_steps += measurement.timed_optimizer_steps
        aggregate_throughput.optimization_seconds += measurement.optimization_seconds

    results: dict[str, Any] = {
        'Training Throughput per Stage (samples/s)': [
            measurement.to_dict()['Training Throughput (samples/s)']
            for measurement in task_throughput
        ],
        'Training Samples per Stage': [measurement.total_samples for measurement in task_throughput],
        'Timed Training Samples per Stage': [measurement.timed_samples for measurement in task_throughput],
        **aggregate_throughput.to_dict(),
        'Final Exemplar Count': cil_model.get_exemplar_count(),
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return results, loss_history_per_task

def train_and_evaluate_control_models(
    models_dict,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer_cls,
    lr=0.001,
    epochs=5,
    device='cpu',
    verbose=False,
):
    if isinstance(train_dataloader, (list, tuple)):
        train_dataloader = build_complete_dataloader(list(train_dataloader), shuffle=True)
    if isinstance(val_dataloader, (list, tuple)):
        val_dataloader = build_complete_dataloader(list(val_dataloader), shuffle=False)

    comparison_rows = []
    trained_models = {}
    loss_histories = {}
    prediction_caches = {}

    for name, get_model_fn in models_dict.items():
        print(f"\n=== Fully training control model: {name} ===")
        model = get_model_fn().to(device)
        optimizer = optimizer_cls(model.parameters(), lr=lr)
        loss_history = []

        model.train()
        start_time = time.time()
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: {name}")
            else:
                print(f"\rTraining {name} - Epoch {epoch + 1}/{epochs}...", end="", flush=True)

            for images, labels, _ in train_dataloader:
                images = torch.stack(images).to(device)
                y_true = torch.tensor(
                    [label['label'] for label in labels],
                    dtype=torch.long,
                    device=device,
                )

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = loss_fn(logits, y_true)
                loss.backward()
                optimizer.step()

                if hasattr(model, 'reset_routing_state'):
                    model.reset_routing_state()

                loss_history.append(loss.detach().item())

        train_time = time.time() - start_time

        model.eval()
        total_loss = 0.0
        all_logits = []
        all_preds = []
        all_targets = []
        all_paths = []

        with torch.no_grad():
            print(f"\rEvaluating {name}...", end="", flush=True)
            for images, labels, paths in val_dataloader:
                images = torch.stack(images).to(device)
                y_true = torch.tensor(
                    [label['label'] for label in labels],
                    dtype=torch.long,
                    device=device,
                )

                logits = model(images)
                loss_val = loss_fn(logits, y_true)
                preds = torch.argmax(logits, dim=1)

                total_loss += loss_val.item()
                all_logits.append(logits.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_true.cpu().numpy())
                all_paths.extend(paths)

        metrics = {
            'Macro F1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
            'Micro F1': f1_score(all_targets, all_preds, average='micro', zero_division=0),
            'Weighted F1': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
        }

        results = {
            'Model': name,
            'Validation Loss': total_loss / len(val_dataloader),
            'Validation Macro F1': metrics['Macro F1'],
            'Validation Micro F1': metrics['Micro F1'],
            'Validation Weighted F1': metrics['Weighted F1'],
            'Training Time (s)': train_time,
            'Train Dataset Size': len(train_dataloader.dataset),
            'Validation Dataset Size': len(val_dataloader.dataset),
        }

        print(
            f"\rControl {name} Macro F1={metrics['Macro F1']:.4f} "
            f"Micro F1={metrics['Micro F1']:.4f}"
        )

        comparison_rows.append(results)
        trained_models[name] = model
        loss_histories[name] = loss_history
        prediction_caches[name] = {
            'logits': np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0)),
            'targets': all_targets,
            'paths': all_paths,
            'metrics': metrics,
        }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(comparison_rows), trained_models, loss_histories, prediction_caches

# =========================================================================================    
# Results Serialization
# =========================================================================================

class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super(Encoder, self).default(o)

def _to_json_compatible(value):
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                key = str(key)
            elif not isinstance(key, (str, int, float, bool)) and key is not None:
                key = str(key)
            normalized[key] = _to_json_compatible(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value

def serialize_save_json(data: dict, file: str):
    with open(file, 'w') as f:
        json.dump(_to_json_compatible(data), f, indent=4, cls=Encoder)


def serialize_save_model_training(results: dict, file: str):
    """Persist aggregate training/validation results without per-example predictions."""
    eval_mat = results.get('Eval Matrix')
    out = {
        'results': {k: v for k, v in results.items() if k != 'Eval Matrix'},
        'eval_mat': eval_mat
    }
    serialize_save_json(out, file)