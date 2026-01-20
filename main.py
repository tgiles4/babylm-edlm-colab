import os
import time

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

# Fix for PyTorch 2.6+ weights_only=True default
# Allowlist omegaconf and typing classes for checkpoint loading
# This is needed because Lightning checkpoints contain omegaconf and typing objects
try:
  import typing

  from omegaconf import dictconfig, listconfig
  if hasattr(torch.serialization, 'add_safe_globals'):
    # Add all omegaconf classes that might be in checkpoints
    safe_classes = [
      dictconfig.DictConfig,
      listconfig.ListConfig,
      typing.Any,
      typing.Union,
      typing.Optional,
      typing.Dict,
      typing.List,
      typing.Tuple,
    ]
    # Try to import and add ContainerMetadata (the one mentioned in the error)
    try:
      from omegaconf.base import ContainerMetadata
      safe_classes.append(ContainerMetadata)
    except (ImportError, AttributeError):
      pass
    # Try to add other omegaconf base classes if available
    try:
      from omegaconf.base import Container, Node
      safe_classes.extend([Container, Node])
    except (ImportError, AttributeError):
      pass
    torch.serialization.add_safe_globals(safe_classes)
except (AttributeError, TypeError, ImportError) as e:
  # Fallback for older PyTorch versions or if add_safe_globals doesn't exist
  pass

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  # Try to determine model type from checkpoint or config
  use_energy = config.get('use_energy', False)

  if config.ebm_backbone == 'ar':
    return diffusion.EBM(
      config, tokenizer=tokenizer).to('cuda')

  if use_energy:
    return diffusion.EBM.load_from_checkpoint(
      config.eval.checkpoint_path,
      tokenizer=tokenizer,
      config=config)
  else:
    return diffusion.Diffusion.load_from_checkpoint(
      config.eval.checkpoint_path,
      tokenizer=tokenizer,
      config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.

  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    # Save config to checkpoint directory
    config_dir = config.checkpointing.save_dir
    os.makedirs(config_dir, exist_ok=True)
    with fsspec.open(
      '{}/config_tree.txt'.format(config_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  model.entropy_metric.reset()
  model.time_metric.reset()

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      time_start = time.time()
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      time_end = time.time()
      # Time Metrics
      model.time_metric.update(time_end - time_start)
      # Entropy Metrics
      model.compute_entropy(samples)
      # Generative Perplexity Metrics
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
        model.gen_ppl_metric.compute().item(),
        'Entropy:',
        model.entropy_metric.compute().item(),
        'Time:',
        model.time_metric.compute().item())
  return text_samples


def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_config = dict(config.wandb)
    # Convert tags to strings (wandb requires all tags to be strings)
    if 'tags' in wandb_config and wandb_config['tags'] is not None:
      wandb_config['tags'] = [str(tag) for tag in wandb_config['tags']]
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** wandb_config)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_config = dict(config.wandb)
    # Convert tags to strings (wandb requires all tags to be strings)
    if 'tags' in wandb_config and wandb_config['tags'] is not None:
      wandb_config['tags'] = [str(tag) for tag in wandb_config['tags']]
    # Create save directory if specified
    if 'save_dir' in wandb_config:
      os.makedirs(wandb_config['save_dir'], exist_ok=True)
      logger.info(f'wandb logs will be saved to: {wandb_config["save_dir"]}')
    # Log wandb initialization
    logger.info('Initializing wandb logger...')
    logger.info(f'  Project: {wandb_config.get("project", "text-diffusion")}')
    logger.info(f'  Run name: {wandb_config.get("name", "auto-generated")}')
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** wandb_config)
    logger.info('✓ wandb logger initialized')

  # Resolve checkpoint path if resuming
  ckpt_path = None
  if config.checkpointing.resume_from_ckpt:
    resume_path = omegaconf.OmegaConf.to_container(
      config.checkpointing.resume_ckpt_path, resolve=True)
    if resume_path and utils.fsspec_exists(resume_path):
      ckpt_path = resume_path
      logger.info(f'Resuming from checkpoint: {ckpt_path}')
    else:
      logger.warning(
        f'Checkpoint path not found: {resume_path}. Starting fresh training.')

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  # Instantiate model based on use_energy flag
  if config.get('use_energy', False):
    logger.info('Using EBM model (DIT with Energy)')
    model = diffusion.EBM(
      config, tokenizer=valid_ds.tokenizer)
  else:
    logger.info('Using Diffusion model (DIT without Energy)')
    model = diffusion.Diffusion(
      config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)

  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()