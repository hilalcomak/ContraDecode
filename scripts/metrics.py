#!/usr/bin/env python

import argparse
from scipy.stats import kendalltau
from scipy import mean
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
            k = kendalltau(X, Y, variant='c')
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
    parser.add_argument("measure", type=str, choices=["k-tau", "cross-entropy", "perplexity"])
    args = parser.parse_args()
    main(args)