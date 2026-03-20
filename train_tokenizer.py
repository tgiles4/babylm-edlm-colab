"""
Train a BPE tokenizer from BabyLM dataset files.

10M and 100M use separate tokenizers (tokenizer_10M.json vs tokenizer_100M.json);
they are not interchangeable.

Default paths are project_dir/data/train_{size} and project_dir/tokenizer/tokenizer_{size}.json.
  - Hopper: set --project_dir in your sbatch script (e.g. $SCRATCH/babylm-edlm).
  - Colab/custom: pass --project_dir and optionally --data_dir/--output_path (see colab/COLAB.md).

100M: empty train_100M/ triggers Hugging Face download of BabyLM-2026-Strict (see babylm_hf.py).
10M: populate train_10M/*.train first; no default Hub download.
"""

import argparse
import glob
import os

from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from transformers import PreTrainedTokenizerFast

from babylm_hf import ensure_babylm_train_files

BABYLM_SIZES = ('10M', '100M')
DEFAULT_SIZE = '10M'


def train_bpe_tokenizer(
    data_dir,
    output_path,
    vocab_size=32000,
    min_frequency=2,
    show_progress=True
):
    """
    Train a BPE tokenizer on BabyLM .train files.

    Args:
        data_dir: Directory containing .train files
        output_path: Path to save tokenizer.json
        vocab_size: Vocabulary size for BPE tokenizer
        min_frequency: Minimum frequency for tokens
        show_progress: Whether to show progress

    Returns:
        PreTrainedTokenizerFast: The trained tokenizer
    """
    # Find all .train files
    train_files = glob.glob(os.path.join(data_dir, '*.train'))
    if not train_files:
        raise ValueError(
            f'No .train files found in directory: {data_dir}')

    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Set up normalizer (lowercase, strip accents)
    # Use Sequence for combining normalizers (compatible with all tokenizers versions)
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    # Set up pre-tokenizer (split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # Initialize trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[BOS]",
            "[EOS]"
        ],
        show_progress=show_progress
    )

    # Train tokenizer on all .train files
    files = sorted(train_files)
    tokenizer.train(files, trainer=trainer)

    # Set up post-processor (add [BOS] and [EOS] tokens)
    tokenizer.post_processor = processors.BertProcessing(
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ("[CLS]", tokenizer.token_to_id("[CLS]"))
    )

    # Set UNK token
    tokenizer.unk_token = "[UNK]"

    # Handle output path - if it's a directory, append tokenizer.json
    if os.path.isdir(output_path) or output_path.endswith('/'):
        output_path = os.path.join(output_path.rstrip('/'), 'tokenizer.json')
    elif not output_path.endswith('.json'):
        output_path = output_path + '.json'

    # Save tokenizer
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    tokenizer.save(output_path)

    # Wrap in PreTrainedTokenizerFast for compatibility
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token="[CLS]",  # Use [CLS] as BOS
        eos_token="[SEP]",  # Use [SEP] as EOS
    )

    return wrapped_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description='Train a BPE tokenizer from BabyLM dataset files. '
        '10M and 100M use separate tokenizers (tokenizer_10M.json vs tokenizer_100M.json).')
    parser.add_argument(
        '--project_dir',
        type=str,
        default=None,
        help='Project base dir; data_dir and output_path default to project_dir/data/train_{size} and project_dir/tokenizer/tokenizer_{size}.json. Default: $SCRATCH/babylm-edlm or ./scratch/babylm-edlm')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Directory containing .train files (default: project_dir/data/train_{size})')
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save tokenizer JSON (default: project_dir/tokenizer/tokenizer_{size}.json)')
    parser.add_argument(
        '--size',
        type=str,
        choices=BABYLM_SIZES,
        default=DEFAULT_SIZE,
        help=f'Dataset size for default paths (default: {DEFAULT_SIZE})')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=32000,
        help='Vocabulary size for BPE tokenizer (default: 32000)')
    parser.add_argument(
        '--min_frequency',
        type=int,
        default=2,
        help='Minimum frequency for tokens (default: 2)')
    parser.add_argument(
        '--no_download',
        action='store_true',
        help='Do not download BabyLM-2026-Strict from Hugging Face (100M only); fail if *.train missing')
    parser.add_argument(
        '--hf_cache_dir',
        type=str,
        default=None,
        help='Cache dir for Hub snapshot (default: HF_HOME or ~/.cache/huggingface)')

    args = parser.parse_args()

    expand = lambda p: os.path.expandvars(os.path.expanduser(p)) if p else p

    project_dir = args.project_dir
    if project_dir is None:
        base = os.environ.get('SCRATCH') or os.path.abspath('./scratch')
        project_dir = os.path.join(base, 'babylm-edlm')
    project_dir = expand(project_dir)

    if args.data_dir is None:
        args.data_dir = os.path.join(project_dir, 'data', f'train_{args.size}')
    if args.output_path is None:
        args.output_path = os.path.join(project_dir, 'tokenizer', f'tokenizer_{args.size}.json')

    args.data_dir = expand(args.data_dir)
    args.output_path = expand(args.output_path)
    args.hf_cache_dir = expand(args.hf_cache_dir) if args.hf_cache_dir else None

    ensure_babylm_train_files(
        args.data_dir,
        args.size,
        hf_cache_dir=args.hf_cache_dir,
        no_download=args.no_download,
    )

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(
            f'Data directory not found: {args.data_dir}\n'
            'See agent-tasks/BABYLM_2026_HF_DOWNLOAD_PLAN.md')

    train_bpe_tokenizer(
        data_dir=args.data_dir,
        output_path=args.output_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )


if __name__ == '__main__':
    main()

