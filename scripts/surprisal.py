#!/usr/bin/env python

import argparse
import numpy as np
from tqdm import tqdm
import sys
from metrics import translator

sys.path.append('probability-of-a-word')
from wordsprobability import get_surprisal_per_word
import wordsprobability.models as wp_models


def main(args):
  assert args.lang in wp_models.LANGUAGES[args.model], f"Language {args.lang} not supported by model {args.model}"
  measures = {k: list() for k in set(args.measure)}

  sentences = args.src.readlines()
  sentences = [s.translate(translator).strip() for s  in sentences]
  sentences = [s for s in sentences if s != ""]
  for sent in tqdm(sentences):
    s = get_surprisal_per_word(sent, args.model)['surprisal'].to_numpy()
    #https://arxiv.org/pdf/2306.03734
    if 'uid_variance' in measures:
      # Eq 1
      μ = np.mean(s)
      measures['uid_variance'].append(np.mean(np.square(s - μ)))
    # https://arxiv.org/pdf/2306.03734
    if 'uid_w2w_change' in measures:
      if s.size > 1:
        # Eq 2
        measures['uid_w2w_change'].append(np.sum(np.square(s[1:]-s[:-1]))/(s.size-1))
  for k,v in measures.items():
    print(f"{k}: {np.mean(v)}")
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
    parser.add_argument("--model", type=str, required=True, choices=wp_models.MODELS.keys())
    parser.add_argument("--lang", type=str, required=True, choices=valid_languages())
    parser.add_argument("src", type=argparse.FileType('r', encoding='UTF-8'), help="Source file")
    parser.add_argument("measure",
                        type=str,
                        nargs='+',
                        choices=["uid_variance", "uid_w2w_change"])
    args = parser.parse_args()
    main(args)