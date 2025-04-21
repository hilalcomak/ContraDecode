#!/usr/bin/env python

import argparse
import math
from scipy.stats import kendalltau
from numpy import mean
import pathlib
import re
import tempfile
import string
import subprocess

# Remove punctuation
translator = str.maketrans('', '', string.punctuation)


def decode_pair(s):
    s = s.split('-')
    return int(s[0]), int(s[1])

def decode_alignments(alignments):
    alignments = alignments.decode('utf-8')
    alignments = [l.strip() for l in alignments.splitlines()]
    alignments = [l.split(' ') for l in alignments if l != ""]
    alignments = [[decode_pair(p) for p in l] for l in alignments]
    return alignments

#https://aclanthology.org/W11-2102.pdf
# https://github.com/google/topdown-btg-preordering/blob/master/evaluate_preordering.py
def fuzzy_reordering(alignment, src_text, trg_text):
    """
    Compute Fuzzy reordering.
    """

    # Remove punctuation and count words
    if isinstance(src_text, str):
        src_wc = len([_ for _ in src_text.strip().translate(translator).split(" ") if _ != ""])
    else:
        assert isinstance(src_text, int)
        src_wc = src_text
    if isinstance(trg_text, str):
        trg_wc = len([_ for _ in trg_text.strip().translate(translator).split(" ") if _ != ""])
    else:
        assert isinstance(trg_text, int)
        trg_wc = trg_text

    assert len(set(a[1] for a in alignment)) == len(alignment), "Each target word can only map to a single src word."
    assert max(a[1] for a in alignment) <= trg_wc, "More tokens in alignment than in sentence!"
    if trg_wc < 2: # There is no way to reorder less than 2 words.
        return 1.
    assert trg_wc > 1, f"Translation of '{src_text}' is '{trg_text}', which has {trg_wc} words."

    alignment = {dst:src for src, dst in alignment}
    # Discontinuities
    jumps = 0
    for cur in range(1, trg_wc, 1):
        prv = cur - 1
        src_prv = alignment.get(prv, -2)
        src_cur = alignment.get(cur, -2)
        if src_prv != src_cur and src_prv + 1 != src_cur:
            jumps += 1
    return 1.- jumps / (trg_wc-1.)

def main(args):
    fp = tempfile.NamedTemporaryFile(mode='w', encoding="UTF-8", delete=False)
    src_lines, tra_lines = args.src.readlines(), args.tra.readlines()
    assert len(src_lines) == len(tra_lines)
    for s, t in zip(src_lines, tra_lines):
        s, t = s.strip().translate(translator), t.strip().translate(translator)
        p = f"{s} ||| {t}\n"
        fp.write(p)
    fp.close()
    bin = str(pathlib.Path(__file__).parent.parent.resolve()) + "/fast_align/build/fast_align"
    cmd = f"{bin} -i {fp.name} -v -d -o"
    print(cmd)
    output = subprocess.run(cmd, shell=True, check=True, capture_output=True)
    alignments = decode_alignments(output.stdout)
    assert len(src_lines) == len(alignments)
    if args.measure == "k-tau":
        k_taus = []
        for a in alignments:
            Y = [_[0] for _ in a]
            X = sorted(Y)
            if len(X) > 2:
                k = kendalltau(X, Y, variant='c')
                #dropping the p-value. we don#t need it. otherwise it returns NAN
                k = k.statistic
            else:
                k = 1.
            k_taus.append(k)
        print(f"Mean K-tau: {mean(k_taus)}")
        return
    if args.measure == "cross-entropy":
        m = re.findall(r"cross entropy: (\d+\.\d+)?\b", output.stderr.decode('utf-8'))[-1]
        print(f"Cross entropy: {m}")
        return
    if args.measure == "perplexity":
        m = re.findall(r"perplexity: (\d+\.\d+)?\b", output.stderr.decode('utf-8'))[-1]
        print(f"perplexity: {m}")
        return
    if args.measure == "fuzzy-reordering":
        fr = mean(list(
            fuzzy_reordering(alignment, src, trg) for src, trg, alignment in zip(src_lines, tra_lines, alignments)
        ))
        print(f"Mean Fuzzy-reordering: {fr}")
        return
    else:
        raise NotImplementedError(f"{args.measure} not implemented.")


example = """example:
  python %(prog)s src.txt ref.txt k-tau
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog=example,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("src", type=argparse.FileType('r', encoding='UTF-8'), help="Sources")
    parser.add_argument("tra", type=argparse.FileType('r', encoding='UTF-8'), help="Translations")
    parser.add_argument("measure", type=str, choices=["k-tau", "cross-entropy", "perplexity", "fuzzy-reordering"])
    args = parser.parse_args()
    main(args)