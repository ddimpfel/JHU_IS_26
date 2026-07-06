from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError

# =========================================================================================    
# Generalist Router Experts Implementation
# =========================================================================================

class GeneralistRouterExperts(nn.Module):
    def __init__(
        self, 
        generalist: nn.Module, 
        router: nn.Module, 
        expert_factory: Callable, 
        num_experts: int, 
        num_classes: int, 
        k=2, 
        lambda_aux=0.01
    ) -> None:
        """
        Hard-routed Mixture-of-Experts model for class-incremental learning. Implements additional
        auxillary loss to efficiently tune the router expert selection.

        Generalist:
            A strong backbone for producing rich features from raw input data.
            This should be implemented to return both logits for the router and
            features for the experts.
        Router:
            A shallow gating network to select which experts to use for final
            classification of generalist features.
        Experts:
            Shallow classification networks trained specifically to classify
            given inputs the router selects the experts for.
        """
        super().__init__()
        self.generalist = generalist
        self.router = router
        self.experts = nn.ModuleList([expert_factory() for _ in range(num_experts)])
        self.k = k
        self.lambda_aux = lambda_aux

        self.num_classes = num_classes
        self.num_experts = len(self.experts)

        self.temperature = nn.Parameter(torch.ones(1))

        self._last_router_probs = None
        self._last_topk_indices = None
        self._last_aux_loss = None

    def _compute_auxiliary_loss(self, router_probs, topk_indices):
        importance = router_probs.mean(dim=0)

        selection_mask = torch.zeros_like(router_probs)
        selection_mask.scatter_(1, topk_indices, 1.0)

        # Normalize by k so the expert usage vector sums to 1
        load = (selection_mask / self.k).mean(dim=0)

        return self.lambda_aux * self.num_experts * torch.sum(importance * load)

    def get_auxiliary_loss(self):
        if self._last_aux_loss is None:
            param = next(self.parameters())
            return torch.zeros((), device=param.device, dtype=param.dtype)
        return self._last_aux_loss

    def reset_routing_state(self):
        self._last_router_probs = None
        self._last_topk_indices = None
        self._last_aux_loss = None

    def routing_summary(self):
        if self._last_router_probs is None or self._last_topk_indices is None:
            return None

        selection_mask = torch.zeros_like(self._last_router_probs)
        selection_mask.scatter_(1, self._last_topk_indices, 1.0)

        return {
            "avg_router_prob": self._last_router_probs.mean(dim=0).detach().cpu(),
            "selection_rate": (selection_mask / self.k).mean(dim=0).detach().cpu(),
        }

    def forward(self, x):
        generalist_logits, features = self.generalist(x, return_features=True)
        if features.ndim != 2:
            features = torch.flatten(features, 1)

        router_logits = self.router(features)
        router_probs = F.softmax(router_logits, dim=1)

        topk_probs, topk_indices = torch.topk(router_probs, k=self.k, dim=1)
        topk_weights = topk_probs / topk_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

        experts_logits = torch.zeros_like(
            generalist_logits,
            device=features.device,
            dtype=features.dtype,
        )

        # Hard-routing
        for expert_idx, expert in enumerate(self.experts):
            selected_rows, selected_slots = torch.where(topk_indices == expert_idx)

            if selected_rows.numel() == 0:
                continue

            expert_features = features[selected_rows]
            expert_logits = expert(expert_features)
            expert_weights = topk_weights[selected_rows, selected_slots].unsqueeze(1)

            experts_logits[selected_rows] += expert_logits * expert_weights

        self._last_router_probs = router_probs.detach()
        self._last_topk_indices = topk_indices.detach()
        self._last_aux_loss = self._compute_auxiliary_loss(router_probs, topk_indices)

        return (generalist_logits + experts_logits) / self.temperature
    
    def get_model_parameters_by_component(self):
        return {
            'generalist': sum(p.numel() for p in self.generalist.parameters()),
            'router':     sum(p.numel() for p in self.router.parameters()),
            'expert':     sum(p.numel() for p in self.experts[0].parameters()),
        }

# =========================================================================================    
# GRE Metrics
# =========================================================================================

def expert_metrics(model: nn.Module, val_data: DataLoader, device='cpu'):
    result = {}

    model.eval()
    per_input_calls = []
    expert_usage = torch.zeros(len(model.experts), dtype=torch.long)

    with torch.no_grad():
        for images, _, _ in val_data:
            images = torch.stack(images).to(device)
            _ = model(images)

            topk_indices = model._last_topk_indices.detach().cpu()
            per_input_calls.append(
                torch.full((topk_indices.size(0),), topk_indices.size(1), dtype=torch.float32)
            )
            expert_usage += torch.bincount(
                topk_indices.reshape(-1),
                minlength=len(model.experts),
            )

    per_input_calls = torch.cat(per_input_calls)

    result["Model k"] = model.k
    result["Expected Expert Calls"] = per_input_calls.mean().item()
    result["Expert Selection Ratio"] = (
        expert_usage.float() / expert_usage.sum().clamp_min(1)
    ).numpy()

    return result

def cost_proxy(expert_calls: torch.Tensor, params: dict[str, int]):
    return expert_calls * params['expert'] + params['generalist'] + params['router']

def expected_calibration_error(
    model: nn.Module,
    num_classes: int,
    val_data: DataLoader, 
    n_bins: int=10, 
    device='cpu'
):
    model.eval()

    ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins).to(device)

    with torch.no_grad():
        for images, labels, _ in val_data:
            images = torch.stack(images).to(device)
            y_true = torch.tensor(
                [l["label"] for l in labels],
                dtype=torch.long,
                device=device,
            )
            y_prob: torch.Tensor = torch.softmax(model(images), dim=1)
            ece.update(y_prob, y_true)

    return ece.compute().item()

def trainable_parameters(model: nn.Module):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad == True:
            trainable += param.numel()

    return trainable, total, trainable / total

def router_entropy(model: nn.Module, val_data: DataLoader, device='cpu', eps=1e-12):
    """ 
    Measures how uncertain the gating model is across many inputs. 
    TODO FIXME
    """
    result = {}

    model.eval()
    all_entropies = []
    avg_router_probs = []

    with torch.no_grad():
        for images, _, _ in val_data:
            images = torch.stack(images).to(device)
            _ = model(images)

            router_probs = model._last_router_probs.clamp_min(eps)
            entropy = -(router_probs * router_probs.log()).sum(dim=1)

            all_entropies.append(entropy.cpu())
            avg_router_probs.append(router_probs.mean(dim=0).cpu())

    all_entropies = torch.cat(all_entropies)
    avg_router_probs = torch.stack(avg_router_probs).mean(dim=0)

    max_entropy = torch.log(torch.tensor(float(model.num_experts))).item()

    result["Mean Router Entropy"] = all_entropies.mean().item()
    result["Normalized Router Entropy"] = all_entropies.mean().item() / max_entropy
    result["Std Router Entropy"] = all_entropies.std(unbiased=False).item()
    result["Avg Router Prob"] = avg_router_probs.numpy()

    return result

# =========================================================================================    
# Reliability Extensions
# =========================================================================================

def calibrate_temperature(model, val_logits: torch.Tensor, val_labels: torch.Tensor, loss_fn, device='cpu'):
    model.eval()
    optimizer = LBFGS([model.temperature], lr=0.01, max_iter=50)

    val_logits = val_logits.detach().to(device, dtype=model.temperature.dtype)
    val_labels = val_labels.detach().to(device, dtype=torch.long)

    with torch.no_grad():
        model.temperature.fill_(1.0)

    def eval_iteration():
        optimizer.zero_grad()
        
        # Handle the masking pushing logits to -inf
        min_val = torch.finfo(val_logits.dtype).min
        mask = (val_logits == min_val)
        scaled_logits = (val_logits / model.temperature.clamp_min(1e-6)).masked_fill(mask, min_val)
        
        loss = loss_fn(scaled_logits, val_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_iteration)
    with torch.no_grad():
        model.temperature.clamp_(min=1e-6)
    print(f"Learned Temperature: {model.temperature.item():.4f}")