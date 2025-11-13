"""
Training utilities for VoidX models.

This module provides a Trainer class for training and validating
void detection models with checkpointing and early stopping.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import json


class Trainer:
	"""
	Generic trainer with early stopping and checkpointing.

	Mirrors the behavior of the notebook's train_MLP while being reusable.
	"""

	def __init__(
		self,
		model: nn.Module,
		dl_train: DataLoader,
		dl_val: DataLoader,
		ds_train,
		ds_val,
		optimizer: optim.Optimizer,
		criterion: nn.Module,
		scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
		device: str | torch.device = "cpu",
		model_name: str = "model",
		checkpoint_path_global: Optional[Path] = None,
		checkpoint_path_spec: Optional[Path] = None,
	) -> None:
		self.model = model
		self.dl_train = dl_train
		self.dl_val = dl_val
		self.ds_train = ds_train
		self.ds_val = ds_val
		self.optimizer = optimizer
		self.criterion = criterion
		self.scheduler = scheduler
		self.device = device
		self.model_name = model_name
		self.checkpoint_path_global = checkpoint_path_global
		self.checkpoint_path_spec = checkpoint_path_spec

	def _save_checkpoint_global(self, state: Dict[str, torch.Tensor]) -> None:
		torch.save(state, self.checkpoint_path_global)

	def _save_checkpoint_spec(self, state: Dict[str, torch.Tensor]) -> None:
		torch.save(state, self.checkpoint_path_spec)

	def _maybe_resume(self, resume: bool, verbose: bool = True) -> None:
		if resume and self.checkpoint_path_spec is not None and self.checkpoint_path_spec.exists():
			if verbose:
				print("Loading checkpoint:", self.checkpoint_path_spec)
			state = torch.load(self.checkpoint_path_spec, map_location=self.device)
			self.model.load_state_dict(state)

	def train(
		self,
		num_epochs: int = 10,
		patience: int = 20,
		resume: bool = True,
		verbose: bool = True,
		*,
		monitor: str = "val_loss",   # one of {"val_loss", "val_bce", "auc"}
		monitor_mode: str = "min",    # "min" for losses, "max" for metrics like AUC
	) -> Tuple[Dict[str, List[float]], Optional[Dict[str, torch.Tensor]], float]:
		"""
		Train the model with early stopping on validation loss.

		Returns: (history, best_state, best_val)
		"""
		if verbose and self.checkpoint_path_global is not None:
			print("Checkpoint path:", self.checkpoint_path_global)

		self._maybe_resume(resume=resume, verbose=verbose)

		history: Dict[str, List[float]] = {
			"train_loss": [],
			"val_loss": [],  # weighted BCE (uses training-set pos_weight)
			"val_bce": [],   # unweighted BCE for apples-to-apples comparison
			"val_pos_frac": [],  # class prevalence in validation this epoch
			"precision": [],
			"recall": [],
			"auc": [],
		}

		# Separate criterion to compute unweighted BCE on validation for comparability
		_val_bce = nn.BCEWithLogitsLoss(reduction="sum")
		# Track the best value based on monitor_mode
		best_val = float("inf") if monitor_mode == "min" else -float("inf")
		best_state: Optional[Dict[str, torch.Tensor]] = None
		no_improve = 0

		start_time = time.time()
		for epoch in range(1, num_epochs + 1):
			epoch_start = time.time()
			self.model.train()
			running = 0.0

			pbar = tqdm(self.dl_train, desc=f"Epoch {epoch:03d}/{num_epochs} [train]", leave=False)
			for xb, yb in pbar:
				xb = xb.to(self.device)
				yb = yb.to(self.device)
				self.optimizer.zero_grad()
				logits = self.model(xb)
				loss = self.criterion(logits, yb)
				loss.backward()
				self.optimizer.step()
				running += loss.item() * xb.size(0)
				pbar.set_postfix(loss=f"{loss.item():.4f}")
			train_loss = running / len(self.ds_train)

			# Validation
			self.model.eval()
			v_running = 0.0  # weighted (matches training criterion)
			v_running_unweighted = 0.0  # plain BCE
			n_val_samples = 0
			pos_count = 0.0
			all_logits, all_targets = [], []
			with torch.no_grad():
				pbar_val = tqdm(self.dl_val, desc=f"Epoch {epoch:03d}/{num_epochs} [val]  ", leave=False)
				for xb, yb in pbar_val:
					xb = xb.to(self.device)
					yb = yb.to(self.device)
					logits = self.model(xb)
					# Weighted loss (uses training-set pos_weight)
					loss = self.criterion(logits, yb)
					batch_size = xb.size(0)
					v_running += loss.item() * batch_size
					# Unweighted BCE for comparability across splits
					v_running_unweighted += _val_bce(logits, yb).item()
					n_val_samples += batch_size
					pos_count += yb.sum().item()
					all_logits.append(logits.cpu())
					all_targets.append(yb.cpu())
			val_loss = v_running / max(1, len(self.ds_val))
			val_bce = v_running_unweighted / max(1, n_val_samples)
			val_pos_frac = float(pos_count / max(1, n_val_samples))

			logits_np = torch.cat(all_logits).numpy().ravel()
			targets_np = torch.cat(all_targets).numpy().ravel()
			probs = 1 / (1 + np.exp(-logits_np))
			preds = (probs >= 0.5).astype(np.int32)
			precision = precision_score(targets_np, preds, zero_division=0)
			recall = recall_score(targets_np, preds, zero_division=0)
			auc = roc_auc_score(targets_np, probs)

			# Select monitored value
			monitored_value_map = {
				"val_loss": val_loss,
				"val_bce": val_bce,
				"auc": auc,
			}
			monitored_value = monitored_value_map.get(monitor, val_loss)

			if self.scheduler is not None:
				# ReduceLROnPlateau expects a metric value
				if hasattr(self.scheduler, "step"):
					try:
						self.scheduler.step(monitored_value)
					except TypeError:
						# In case a different scheduler signature is used
						self.scheduler.step()

			history["train_loss"].append(train_loss)
			history["val_loss"].append(val_loss)
			history["precision"].append(precision)
			history["recall"].append(recall)
			history["auc"].append(auc)
			history["val_bce"].append(val_bce)
			history["val_pos_frac"].append(val_pos_frac)

			current_lr = self.optimizer.param_groups[0]["lr"]
			is_better = (monitored_value < best_val) if monitor_mode == "min" else (monitored_value > best_val)
			if is_better:
				best_val = monitored_value
				best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
				self._save_checkpoint_global(best_state)
				self._save_checkpoint_spec(best_state)
				no_improve = 0
				improved = "*"
			else:
				no_improve += 1
				improved = ""

			epoch_time = time.time() - epoch_start
			total_time = time.time() - start_time

			if verbose:
				mon_str = f"monitor({monitor},{monitor_mode})={monitored_value:.4f}{improved}"
				print(
					f"Epoch {epoch:03d}/{num_epochs} | "
					f"train {train_loss:.4f} | val(w) {val_loss:.4f} | val(bce) {val_bce:.4f} | "
					f"val_pos {val_pos_frac:.3f} | prec {precision:.4f} | rec {recall:.4f} | auc {auc:.4f} | "
					f"{mon_str} | "
					f"lr {current_lr:.2e} | "
					f"no_improve {no_improve}/{patience} | "
					f"epoch {epoch_time:.1f}s | total {total_time/60:.1f}min"
				)

			if no_improve >= patience:
				if verbose:
					print("Early stopping")
				break

		return history, best_state, best_val


__all__ = ["Trainer"]
