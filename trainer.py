import itertools
import os
from typing import Optional, Dict, Any

import hydra.utils
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import EvalPrediction

import diffusion
import models


class DiffusionTrainer(Trainer):
    """
    HuggingFace Trainer wrapper for Diffusion and EBM models.

    This trainer adapts the Diffusion/EBM Lightning modules to work
    with HuggingFace's Trainer API, supporting only BERT backbone.
    """

    def __init__(
        self,
        model: diffusion.Diffusion,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        # Ensure model is using BERT backbone
        if not isinstance(model.backbone, models.bert.BERTForDiffusion):
            raise ValueError(
                "DiffusionTrainer only supports BERT backbone. "
                f"Got backbone type: {type(model.backbone)}"
            )

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Store original model reference
        self.diffusion_model = model

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss using the Diffusion/EBM model's _loss method.

        Args:
            model: The Diffusion or EBM model
            inputs: Batch dictionary with 'input_ids' and optionally 'attention_mask'
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor, and optionally outputs
        """
        # Extract input_ids and attention_mask from batch
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')

        if input_ids is None:
            raise ValueError("Batch must contain 'input_ids'")

        # Compute loss using the model's _loss method
        # The prefix is determined by whether we're in training or eval mode
        prefix = 'train' if model.training else 'val'
        loss_result = model._loss(input_ids, attention_mask, prefix=prefix)

        loss = loss_result.loss

        # Update metrics
        if prefix == 'train':
            model.train_metrics.update(loss_result.nlls, loss_result.token_mask)
        elif prefix == 'val':
            model.valid_metrics.update(loss_result.nlls, loss_result.token_mask)

        if return_outputs:
            # Return dummy outputs for compatibility
            outputs = type('Outputs', (), {
                'loss': loss,
                'logits': None,
            })()
            return (loss, outputs)

        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a training step, including EMA updates.
        """
        loss = super().training_step(model, inputs)

        # Update EMA if enabled
        if model.ema is not None:
            model.ema.update(itertools.chain(
                model.backbone.parameters(),
                model.noise.parameters()
            ))

        return loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log metrics, including custom Diffusion metrics.
        """
        # Get metrics from the model
        if self.diffusion_model.training:
            metrics = self.diffusion_model.train_metrics
        else:
            metrics = self.diffusion_model.valid_metrics

        # Compute and log metrics
        computed_metrics = metrics.compute()
        for key, value in computed_metrics.items():
            logs[key] = value.item() if torch.is_tensor(value) else value

        # Call parent log method
        super().log(logs)

    def evaluate(
        self,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys: Optional[list] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        """
        # Store EMA state and use EMA weights for evaluation
        if self.diffusion_model.ema is not None:
            self.diffusion_model.ema.store(itertools.chain(
                self.diffusion_model.backbone.parameters(),
                self.diffusion_model.noise.parameters()
            ))
            self.diffusion_model.ema.copy_to(itertools.chain(
                self.diffusion_model.backbone.parameters(),
                self.diffusion_model.noise.parameters()
            ))

        # Set model to eval mode
        self.diffusion_model.backbone.eval()
        self.diffusion_model.noise.eval()

        # Reset validation metrics
        self.diffusion_model.valid_metrics.reset()

        # Run evaluation
        eval_results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Restore original weights
        if self.diffusion_model.ema is not None:
            self.diffusion_model.ema.restore(itertools.chain(
                self.diffusion_model.backbone.parameters(),
                self.diffusion_model.noise.parameters()
            ))

        # Set model back to train mode
        self.diffusion_model.backbone.train()
        self.diffusion_model.noise.train()

        return eval_results

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model, optionally with EMA weights.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Save EMA state if enabled
        if self.diffusion_model.ema is not None:
            ema_state = self.diffusion_model.ema.state_dict()
            torch.save(ema_state, os.path.join(output_dir, 'ema.pt'))

        # Save model state
        torch.save(self.diffusion_model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

        # Save config and tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        Load model from checkpoint, including EMA state.
        """
        checkpoint_path = resume_from_checkpoint

        # Load model state
        model_state = torch.load(
            os.path.join(checkpoint_path, 'pytorch_model.bin'),
            map_location=self.args.device
        )
        self.diffusion_model.load_state_dict(model_state)

        # Load EMA state if it exists
        ema_path = os.path.join(checkpoint_path, 'ema.pt')
        if os.path.exists(ema_path) and self.diffusion_model.ema is not None:
            ema_state = torch.load(ema_path, map_location=self.args.device)
            self.diffusion_model.ema.load_state_dict(ema_state)

        return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def create_optimizer(self):
        """
        Create optimizer using model's configuration.
        This overrides the default HF Trainer optimizer creation.
        """
        optimizer = torch.optim.AdamW(
            itertools.chain(
                self.diffusion_model.backbone.parameters(),
                self.diffusion_model.noise.parameters()
            ),
            lr=self.diffusion_model.config.optim.lr,
            betas=(
                self.diffusion_model.config.optim.beta1,
                self.diffusion_model.config.optim.beta2
            ),
            eps=self.diffusion_model.config.optim.eps,
            weight_decay=self.diffusion_model.config.optim.weight_decay
        )
        return optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Create scheduler using model's configuration.
        This overrides the default HF Trainer scheduler creation.
        """
        if optimizer is None:
            optimizer = self.optimizer

        # Create scheduler based on model's config
        # Try to instantiate with num_training_steps if the scheduler supports it
        try:
            scheduler = hydra.utils.instantiate(
                self.diffusion_model.config.lr_scheduler,
                optimizer=optimizer,
                num_training_steps=num_training_steps
            )
        except (TypeError, ValueError):
            # If scheduler doesn't accept num_training_steps, create without it
            scheduler = hydra.utils.instantiate(
                self.diffusion_model.config.lr_scheduler,
                optimizer=optimizer
            )

        return scheduler


class EMBCallback(TrainerCallback):
    """
    Callback for EBM-specific functionality during training.
    """

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """
        Called at the end of each epoch.
        """
        # Add any EBM-specific epoch-end logic here
        pass


def create_trainer(
    model: diffusion.Diffusion,
    training_args: TrainingArguments,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    callbacks=None,
) -> DiffusionTrainer:
    """
    Factory function to create a DiffusionTrainer.

    Args:
        model: Diffusion or EBM model instance
        training_args: HuggingFace TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer instance
        data_collator: Data collator (optional)
        callbacks: List of callbacks (optional)

    Returns:
        DiffusionTrainer instance
    """
    # Add EBM callback if using EBM
    if isinstance(model, diffusion.EBM):
        if callbacks is None:
            callbacks = []
        callbacks.append(EMBCallback())

    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    return trainer

