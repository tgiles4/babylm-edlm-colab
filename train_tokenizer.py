"""
Train a BPE tokenizer from BabyLM dataset files.

Usage:
    python train_tokenizer.py \
        --data_dir /data/train_10M \
        --output_path /content/drive/MyDrive/babylm-edlm/tokenizer.json \
        --vocab_size 32000

Or in Colab:
    !python train_tokenizer.py --data_dir /data/train_10M --output_path /content/drive/MyDrive/babylm-edlm/tokenizer.json
"""

import argparse
import glob
import os
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from transformers import PreTrainedTokenizerFast


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
    print(f"Training BPE tokenizer on data from: {data_dir}")
    print(f"Output path: {output_path}")
    print(f"Vocabulary size: {vocab_size}")

    # Find all .train files
    train_files = glob.glob(os.path.join(data_dir, '*.train'))
    if not train_files:
        raise ValueError(
            f'No .train files found in directory: {data_dir}')

    print(f"\nFound {len(train_files)} .train files:")
    for f in sorted(train_files):
        print(f"  - {os.path.basename(f)}")

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
    print(f"\nTraining tokenizer on {len(train_files)} files...")
    print("This may take a few minutes...")

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
        print(f"Output path is a directory, saving to: {output_path}")
    elif not output_path.endswith('.json'):
        # If no extension, assume it should be .json
        output_path = output_path + '.json'
        print(f"Adding .json extension, saving to: {output_path}")

    # Save tokenizer
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    tokenizer.save(output_path)
    print(f"\n✓ Tokenizer saved to: {output_path}")

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

    # Print tokenizer info
    print(f"\nTokenizer Info:")
    print(f"  Vocab size: {wrapped_tokenizer.vocab_size}")
    print(f"  Special tokens:")
    print(f"    PAD: {wrapped_tokenizer.pad_token} (ID: {wrapped_tokenizer.pad_token_id})")
    print(f"    UNK: {wrapped_tokenizer.unk_token} (ID: {wrapped_tokenizer.unk_token_id})")
    print(f"    CLS/BOS: {wrapped_tokenizer.cls_token} (ID: {wrapped_tokenizer.cls_token_id})")
    print(f"    SEP/EOS: {wrapped_tokenizer.sep_token} (ID: {wrapped_tokenizer.sep_token_id})")
    print(f"    MASK: {wrapped_tokenizer.mask_token} (ID: {wrapped_tokenizer.mask_token_id})")

    # Test tokenizer
    print(f"\nTesting tokenizer:")
    test_text = "This is a test sentence for the tokenizer."
    tokens = wrapped_tokenizer.encode(test_text)
    decoded = wrapped_tokenizer.decode(tokens)
    print(f"  Input: {test_text}")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: {decoded}")

    return wrapped_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description='Train a BPE tokenizer from BabyLM dataset files')
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing .train files (e.g., /data/train_10M)')
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save tokenizer.json (e.g., /content/drive/MyDrive/babylm-edlm/tokenizer.json)')
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

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(
            f'Data directory not found: {args.data_dir}')

    # Train tokenizer
    tokenizer = train_bpe_tokenizer(
        data_dir=args.data_dir,
        output_path=args.output_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )

    print(f"\n{'='*60}")
    print("Tokenizer training complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Verify tokenizer exists at: {args.output_path}")
    print(f"2. Update configs/data/babylm.yaml if needed")
    print(f"3. Start training: python main.py data=babylm noise=cosine")


if __name__ == '__main__':
    main()

