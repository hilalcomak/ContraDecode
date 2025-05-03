#!/usr/bin/env python

import argparse
import sys
from metrics import translator
sys.path.append('probability-of-a-word')
from wordsprobability import get_surprisal_per_word
import wordsprobability.models as wp_models


def main(args):
  assert args.lang in wp_models.LANGUAGES[args.model], f"Language {args.lang} not supported by model {args.model}"
  content = args.src.read()
  content = content.translate(translator)
  df = get_surprisal_per_word(content, args.model)
  print(df)
  exit(0)

def valid_languages():
  ret = set()
  for v in wp_models.LANGUAGES.values():
    ret = ret | v
  return ret

example = """example:
  python %(prog)s src.txt ref.txt k-tau
"""
if __name__ == "__main__":
    assert wp_models.MODELS.keys() == wp_models.LANGUAGES.keys(), "All models must have a language"
    parser = argparse.ArgumentParser(
        epilog=example,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("src", type=argparse.FileType('r', encoding='UTF-8'), help="Source file")
    parser.add_argument("--model", type=str, required=True, choices=wp_models.MODELS.keys())
    parser.add_argument("--lang", type=str, required=True, choices=valid_languages())
    #parser.add_argument("measure", type=str, choices=["k-tau", "cross-entropy", "perplexity", "fuzzy-reordering"])
    args = parser.parse_args()
    main(args)