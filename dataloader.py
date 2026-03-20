import functools
import itertools
import json
import math
import os
import re
import shutil
import time
import typing
import urllib
import zipfile

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers

import babylm_hf
import utils

LOGGER = utils.get_logger(__name__)


def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string


def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
      x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x


class Text8Tokenizer(transformers.PreTrainedTokenizer):
  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    **kwargs):
    self.characters = list('abcdefghijklmnopqrstuvwxyz ')
    self._vocab_str_to_int = {
      '[CLS]': 0,
      '[SEP]': 1,
      '[BOS]': 2,
      '[EOS]': 3,
      '[MASK]': 4,
      '[PAD]': 5,
      '[RESERVED]': 6,
      '[UNK]': 7,
      ** {ch: i + 8 for i, ch in enumerate(self.characters)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
      response = requests.get(url, stream=True)
      data_list = []

      # Process each line in the response content
      for line in response.iter_lines(decode_unicode=True):
        if line:
          data = json.loads(line)
          data_list.append(data)

      return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


def get_babylm_dataset(data_dir, cache_dir=None, dataset_size=None):
  """
  Load BabyLM dataset from .train files in a directory.

  There is no official validation split on the Hub for the 2026 strict training corpus;
  validation is a **local holdout**: either pre-materialized ``*.val`` files (100M Hub path)
  or an in-memory 90/10 split (``train_test_split(..., test_size=0.1, seed=42)``) when only
  ``*.train`` files exist.

  If cache_dir is set, the DatasetDict is cached; only one process builds the cache
  (single-writer lock).

  Args:
    data_dir: Path to directory containing .train files (e.g., /data/train_10M)
    cache_dir: Optional cache directory for the raw combined dataset
    dataset_size: ``10M`` or ``100M`` (from config); if None, inferred from ``data_dir`` basename

  Returns:
    datasets.DatasetDict with 'train' and 'validation' splits
  """
  import glob

  ds_size = dataset_size or babylm_hf.infer_dataset_size_from_data_dir(data_dir)
  if ds_size is None:
    raise ValueError(
      'Could not infer dataset_size (10M vs 100M) from data_dir; '
      'set dataset_size in config (e.g. dataset_size=100M) or use a path like .../train_100M')

  hub_cache = os.path.join(cache_dir, 'hf_hub') if cache_dir else None
  babylm_hf.ensure_babylm_train_files(
    data_dir, ds_size, hf_cache_dir=hub_cache)

  train_files = glob.glob(os.path.join(data_dir, '*.train'))
  val_files = glob.glob(os.path.join(data_dir, '*.val'))
  if not train_files:
    raise ValueError(
      f'No .train files found in directory: {data_dir}')

  cache_subdir = 'babylm_raw'
  data_dir_basename = os.path.basename(os.path.normpath(data_dir))
  cache_suffix = '_explicit_val' if val_files else ''
  cache_path = (
    os.path.join(cache_dir, cache_subdir, data_dir_basename + cache_suffix)
    if cache_dir else None)
  if cache_path and utils.fsspec_exists(cache_path):
    LOGGER.info(f'Loading BabyLM raw dataset from cache: {cache_path}')
    return datasets.load_from_disk(cache_path)

  builder = False
  lock_path = (cache_path + '.lock') if cache_path else None
  if cache_path:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
      open(lock_path, 'x').close()
      builder = True
    except FileExistsError:
      cache_ready_timeout = 3600
      deadline = time.monotonic() + cache_ready_timeout
      LOGGER.info(f'Waiting for BabyLM raw cache at {cache_path}...')
      while not utils.fsspec_exists(cache_path):
        if time.monotonic() > deadline:
          try:
            os.remove(lock_path)
          except OSError:
            pass
          raise TimeoutError(
            f'BabyLM raw cache at {cache_path} did not appear within {cache_ready_timeout}s.')
        time.sleep(2)
      LOGGER.info(f'Loading BabyLM raw dataset from cache: {cache_path}')
      return datasets.load_from_disk(cache_path)

  LOGGER.info(f'Found {len(train_files)} .train files in {data_dir}')
  LOGGER.info(f'Files: {[os.path.basename(f) for f in train_files]}')

  if val_files:
    LOGGER.info(
      f'Found {len(val_files)} .val files (train/val split on disk); '
      f'Files: {[os.path.basename(f) for f in val_files]}')
    all_train = []
    for train_file in sorted(train_files):
      LOGGER.info(f'Reading {os.path.basename(train_file)}...')
      with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
          line = line.strip()
          if line:
            all_train.append({'text': line})
    all_val = []
    for val_file in sorted(val_files):
      LOGGER.info(f'Reading {os.path.basename(val_file)}...')
      with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
          line = line.strip()
          if line:
            all_val.append({'text': line})
    dataset_dict = datasets.DatasetDict({
      'train': datasets.Dataset.from_list(all_train),
      'validation': datasets.Dataset.from_list(all_val),
    })
  else:
    all_texts = []
    for train_file in sorted(train_files):
      LOGGER.info(f'Reading {os.path.basename(train_file)}...')
      with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
          line = line.strip()
          if line:
            all_texts.append({'text': line})

    LOGGER.info(f'Loaded {len(all_texts)} total examples')

    full_dataset = datasets.Dataset.from_list(all_texts)
    split_dataset = full_dataset.train_test_split(
      test_size=0.1, seed=42, shuffle=True)
    dataset_dict = datasets.DatasetDict({
      'train': split_dataset['train'],
      'validation': split_dataset['test']
    })

  LOGGER.info(
    f'Split dataset: {len(dataset_dict["train"])} train, '
    f'{len(dataset_dict["validation"])} validation')

  if cache_path and builder:
    try:
      dataset_dict.save_to_disk(cache_path)
      LOGGER.info(f'Saved BabyLM raw dataset to cache: {cache_path}')
    finally:
      try:
        os.remove(lock_path)
      except OSError:
        pass

  return dataset_dict

def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
  """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
  """
  url = 'http://mattmahoney.net/dc/text8.zip'
  if not crop_train:
    cache_dir = f'{cache_dir}/text8'
  else:
    cache_dir = f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    utils.fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    # Check if raw data exists
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      utils.fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not utils.fsspec_exists(
        os.path.join(raw_cache_dir, 'text8.zip')):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL {}.'.format(url))
        with (urllib.request.urlopen(url) as in_stream,
              open(os.path.join(raw_cache_dir, 'text8.zip'),
                   'wb') as out_file):
          shutil.copyfileobj(in_stream, out_file)

      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read(
          'text8').decode('utf-8')

      # Splits taken from D3PM codebase
      splits = {
        'train': rawdata[:90000000],
        'validation': rawdata[90000000: 95000000],
        'test': rawdata[95000000:],
      }

      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()

    # Chunk and save as datasets.DatasetDict
    def chunks(lst, n):
      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
        yield lst[i:i + n]

    dataset_dict = {}
    for k, v in splits.items():
      if k == 'train' and crop_train == True:
        chunk_size = 2 * max_seq_length
      else:
        chunk_size = max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)

  return dataset


def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def get_dataset(
    dataset_name, tokenizer, wrap, mode, cache_dir,
    block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False,
    data_dir=None, dataset_size=None):
  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'
  _path = os.path.join(cache_dir, filename)

  if utils.fsspec_exists(_path):
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')

  # Multi-GPU: only one process builds the cache; others wait and then load.
  lock_path = _path + '.lock'
  cache_ready_timeout = 3600
  builder = False
  try:
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    open(lock_path, 'x').close()
    builder = True
  except FileExistsError:
    LOGGER.info(f'Waiting for cache at {_path} (another process is building)...')
    deadline = time.monotonic() + cache_ready_timeout
    while not utils.fsspec_exists(_path):
      if time.monotonic() > deadline:
        try:
          os.remove(lock_path)
        except OSError:
          pass
        raise TimeoutError(
          f'Cache at {_path} did not appear within {cache_ready_timeout}s. '
          'Remove the .lock file if a previous build failed.')
      time.sleep(2)
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')

  assert builder
  LOGGER.info(f'Generating new data at: {_path}')
  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for sub-sampling
    block_size *= 2

  if dataset_name == 'wikitext103':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-103-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'wikitext2':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-2-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'ptb':
    dataset = datasets.load_dataset(
      'ptb_text_only', cache_dir=cache_dir)
  elif dataset_name == 'lambada':
    dataset = get_lambada_test_dataset()
  elif dataset_name == 'text8':
    assert wrap
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size)
  elif dataset_name == 'text8-crop':
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_arxiv':
    dataset = datasets.load_dataset(
      'scientific_papers', 'arxiv',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_pubmed':
    dataset = datasets.load_dataset(
      'scientific_papers', 'pubmed',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'ag_news':
    dataset = datasets.load_dataset(
      'ag_news',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'babylm':
    # BabyLM dataset - load from .train files in data directory
    if data_dir is None:
      raise ValueError(
        'babylm dataset requires data_dir to be set. '
        'Set config.data.data_dir to the directory containing .train files')
    dataset = get_babylm_dataset(
      data_dir, cache_dir=cache_dir, dataset_size=dataset_size)
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming)

  if dataset_name in ['lambada', 'openwebtext-train',
                      'openwebtext-valid']:
    data = dataset
  elif dataset_name == 'babylm':
    # BabyLM already returns DatasetDict with train/validation splits
    data = dataset[mode]
  else:
    data = dataset[mode]

  if dataset_name.startswith('wikitext'):
    detokenizer = wt_detokenizer
  elif dataset_name == 'ptb':
    detokenizer = ptb_detokenizer
  elif dataset_name == 'lm1b':
    detokenizer = lm1b_detokenizer
  elif dataset_name == 'lambada':
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith('scientific_papers'):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  def _apply_detokenizer(detokenizer):
    def detok(text):
      for i, t in enumerate(text, 0):
        text[i] = detokenizer(t)
      return text
    return detok

  EOS = tokenizer.eos_token_id
  BOS = tokenizer.bos_token_id

  def preprocess_and_tokenize(example):
    if dataset_name == 'ptb':
      text = example['sentence']
    elif 'scientific_papers' in dataset_name:
      text = example['article']
    else:
      text = example['text']

    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True)
    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      desc='Tokenizing')
  else:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Tokenizing')
  if dataset_name == 'ptb':
    tokenized_dataset = tokenized_dataset.remove_columns(
      'sentence')
  elif 'scientific_papers' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns([
      'article', 'abstract', 'section_names'])
  elif dataset_name == 'ag_news':
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['text', 'label'])
  else:
    tokenized_dataset = tokenized_dataset.remove_columns(
      'text')

  if not wrap:
    tokenized_dataset.save_to_disk(_path)
    try:
      os.remove(lock_path)
    except OSError:
      pass
    return tokenized_dataset.with_format('torch')

  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      desc='Grouping')
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Grouping')
    chunked_dataset.save_to_disk(_path)
  try:
    os.remove(lock_path)
  except OSError:
    pass
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def get_tokenizer(config):
  if config.data.tokenizer_name_or_path == 'text8':
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased')
  else:
    tokenizer_path = config.data.tokenizer_name_or_path

    # Check if it's a single .json file (BPE tokenizer from tokenizers library)
    # Use fsspec for path checking to handle Google Drive and other remote paths
    if tokenizer_path.endswith('.json'):
      # It's a JSON file - try to load it
      if not utils.fsspec_exists(tokenizer_path):
        raise FileNotFoundError(
          f'Tokenizer file not found: {tokenizer_path}\n'
          f'\n'
          f'Since you are training from scratch, you need to create a tokenizer first.\n'
          f'Run the tokenizer training script (match dataset_size, e.g. 10M or 100M):\n'
          f'  python train_tokenizer.py --project_dir /path/to/project --size 10M\n'
          f'Or on a cluster: sbatch scripts/job_train_tokenizer_10m.slurm\n'
          f'  (for 100M: scripts/job_train_tokenizer_100m.slurm)\n'
          f'\n'
          f'This writes tokenizer to project_dir/tokenizer/tokenizer_<size>.json using\n'
          f'data from project_dir/data/train_<size>/*.train')
      LOGGER.info(f'Loading tokenizer from JSON file: {tokenizer_path}')
      try:
        # Load using tokenizers library
        tokenizer_obj = tokenizers.Tokenizer.from_file(tokenizer_path)
        # Wrap in PreTrainedTokenizerFast for compatibility
        # Set special tokens explicitly to ensure they're recognized
        tokenizer = transformers.PreTrainedTokenizerFast(
          tokenizer_object=tokenizer_obj,
          pad_token="[PAD]",
          unk_token="[UNK]",
          cls_token="[CLS]",
          sep_token="[SEP]",
          mask_token="[MASK]",
          bos_token="[CLS]",  # Use [CLS] as BOS
          eos_token="[SEP]",  # Use [SEP] as EOS
        )
        # Ensure special token attributes are set (sometimes PreTrainedTokenizerFast
        # doesn't set them properly from the tokenizer_object)
        if tokenizer.bos_token is None:
          # Try to get from cls_token
          if tokenizer.cls_token is not None:
            tokenizer.bos_token = tokenizer.cls_token
            tokenizer.bos_token_id = tokenizer.cls_token_id
          else:
            # Fallback: try to get token ID directly
            try:
              bos_id = tokenizer_obj.token_to_id("[CLS]")
              if bos_id is not None:
                tokenizer.bos_token = "[CLS]"
                tokenizer.bos_token_id = bos_id
            except:
              pass
        if tokenizer.eos_token is None:
          # Try to get from sep_token
          if tokenizer.sep_token is not None:
            tokenizer.eos_token = tokenizer.sep_token
            tokenizer.eos_token_id = tokenizer.sep_token_id
          else:
            # Fallback: try to get token ID directly
            try:
              eos_id = tokenizer_obj.token_to_id("[SEP]")
              if eos_id is not None:
                tokenizer.eos_token = "[SEP]"
                tokenizer.eos_token_id = eos_id
            except:
              pass
        LOGGER.info(f'✓ Tokenizer loaded successfully from {tokenizer_path}')
        LOGGER.info(f'  Vocab size: {tokenizer.vocab_size}')
        LOGGER.info(f'  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})')
        LOGGER.info(f'  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})')
      except Exception as e:
        raise ValueError(
          f'Failed to load tokenizer from {tokenizer_path}.\n'
          f'Error: {e}\n'
          f'Make sure the file is a valid tokenizer.json file from the tokenizers library.')
    elif utils.fsspec_exists(tokenizer_path):
      # It's a directory, use AutoTokenizer
      LOGGER.info(f'Loading tokenizer from directory: {tokenizer_path}')
      tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    else:
      # Assume it's a HuggingFace model ID or file doesn't exist
      # Try HuggingFace first, then provide helpful error if that fails
      try:
        LOGGER.info(f'Attempting to load tokenizer from HuggingFace: {tokenizer_path}')
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
      except Exception as e:
        raise FileNotFoundError(
          f'Tokenizer not found: {tokenizer_path}\n'
          f'HuggingFace error: {e}\n'
          f'\n'
          f'Since you are training from scratch, you need to create a tokenizer first.\n'
          f'Run the tokenizer training script (match dataset_size, e.g. 10M or 100M):\n'
          f'  python train_tokenizer.py --project_dir /path/to/project --size 10M\n'
          f'Or on a cluster: sbatch scripts/job_train_tokenizer_10m.slurm\n'
          f'  (for 100M: scripts/job_train_tokenizer_100m.slurm)\n'
          f'\n'
          f'This writes tokenizer to project_dir/tokenizer/tokenizer_<size>.json using\n'
          f'data from project_dir/data/train_<size>/*.train')

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  # Calculate what batch_size should be to match global_batch_size
  # Effective batch size = batch_size * num_nodes * num_gpus * accumulate_grad_batches
  # So: batch_size = global_batch_size / (num_nodes * num_gpus * accumulate_grad_batches)
  expected_batch_size = (config.loader.global_batch_size
                         // (config.trainer.num_nodes
                             * num_gpus
                             * config.trainer.accumulate_grad_batches))

  # Check if the calculated batch_size matches what's expected
  if config.loader.batch_size != expected_batch_size:
    LOGGER.warning(
      f'Batch size mismatch: config has batch_size={config.loader.batch_size}, '
      f'but expected {expected_batch_size} for '
      f'global_batch_size={config.loader.global_batch_size}, '
      f'num_nodes={config.trainer.num_nodes}, '
      f'num_gpus={num_gpus}, '
      f'accumulate_grad_batches={config.trainer.accumulate_grad_batches}. '
      f'Using expected batch_size={expected_batch_size}.')
    # Override with correct batch_size
    config.loader.batch_size = expected_batch_size

  # Verify effective batch size matches global
  effective_batch_size = (config.loader.batch_size
                          * config.trainer.num_nodes
                          * num_gpus
                          * config.trainer.accumulate_grad_batches)
  if config.loader.global_batch_size != effective_batch_size:
    raise ValueError(
      f'Cannot achieve global_batch_size={config.loader.global_batch_size} with '
      f'batch_size={config.loader.batch_size}, num_nodes={config.trainer.num_nodes}, '
      f'num_gpus={num_gpus}, accumulate_grad_batches={config.trainer.accumulate_grad_batches}. '
      f'Effective batch size would be {effective_batch_size}.')
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  if skip_train:
    train_set = None
  else:
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      data_dir=getattr(config.data, 'data_dir', None),
      dataset_size=getattr(config, 'dataset_size', None))

  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  else:
    validation_split = 'validation'
  if skip_valid:
    valid_set = None
  else:
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=False,
      data_dir=getattr(config.data, 'data_dir', None),
      dataset_size=getattr(config, 'dataset_size', None))

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=config.loader.num_workers > 0)
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0
