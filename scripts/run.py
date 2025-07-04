import logging

import translation_models
from mt_task import MTTask
from translation_models import load_translation_model, valid_translation_models
import ast
import argparse

logging.basicConfig(level=logging.INFO)

def main(args):

    model = load_translation_model(args.model_path, device=0)
    language_pairs = args.language_pairs.split(',')
    language_pairs = [x.split("-") for x in language_pairs]

    if args.oneshot:
        assert isinstance(model, translation_models.llama.LLaMaTranslationModel)
        model.one_shot = True

    tasks = []
    
    for lang_pair in language_pairs:
        tasks.append(MTTask(lang_pair[0],lang_pair[1],args.dataset))
        print(f"Task added {lang_pair[0]} - {lang_pair[1]}")
    if args.out_prefix:
      prefix = "-".join([args.out_prefix, args.model_path])
    else:
      prefix = args.model_path
    for task in tasks:
        if args.source_contrastive or args.language_contrastive or args.prompt_contrastive is not None:
            print(f"Evaluating {task} multi_source")
            out_path = task.evaluate(
                model.translate_multi_source,
                'contrastive',
                args.source_contrastive,
                args.source_weight,
                args.language_contrastive,
                args.language_weight,
                args.prompt_contrastive,
                prefix=prefix,
                small_dev=args.small_dev)
            print(f"Translations saved in {out_path}")
        else:
            print(f"Evaluating {task} direct")
            out_path = task.evaluate(model.translate, 'direct', prefix=prefix, small_dev=args.small_dev)
            print(f"Translations saved in {out_path}")


def prompt_contrastive(filename):
  with open(filename, 'r', encoding='UTF-8') as file:
    templates = [line.rstrip() for line in file]
    templates = [ast.literal_eval(t) for t in templates if t != ""]
    for t in templates:
      assert isinstance(t, tuple)
      assert len(t) == 2
      assert isinstance(t[0], float)
      assert isinstance(t[1], str)
      if not "{src_sent}" in t[1]:
        raise argparse.ArgumentTypeError("No {src_sent} found in " + t[1])
      if not "{tgt_lang}" in t[1]:
        raise argparse.ArgumentTypeError("No {tgt_lang} found in " + t[1])
    return templates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", choices=valid_translation_models(),
                        help="The HF model path")
    parser.add_argument("--dataset", default="flores", choices=["flores", "wmt20", "wmt19", "wmt18"])
    parser.add_argument("--language_pairs", type=str, default="",
                        help="Pairs of languages for which to generate translations.")
    parser.add_argument("--source_contrastive", nargs='?', const=1, type=int,
                        help="enable contrastive input (randomly shuffled segments from input file). Optional INT defines number of contrastive inputs (default: 1).")
    parser.add_argument("--source_weight", type=float, default=-0.1,
                        help="weight of source-contrastive variant. Default -0.1. If multiple contrastive inputs are used, this defines total weight assigned to them.")
    parser.add_argument("--language_contrastive", type=str, nargs='+', default=None,
                        help="language codes of languages for which to construct contrastive variants. Can be multiple"
                             "(space-separated); 'src' will be mapped to language code of source language. "
                             "Example: '--language_contrastive en src' will create two contrastive inputs, one with"
                             " English, one with the source language as desired output language.")
    parser.add_argument("--language_weight", type=float, default=-0.1,
                        help="weight of contrastive variants with wrong language indicator. Default -0.1."
                             " If multiple contrastive inputs are used, this specifies weight assigned to each of them individually.")
    parser.add_argument("--prompt_contrastive", type=prompt_contrastive, default=None,
                        help="Text file with the contrastive prompts to be used.")

    parser.add_argument("--oneshot", action='store_true', default=False,
                        help="For LLaMa: provide one-shot translation example")
    parser.add_argument("--out_prefix", type=str, default=None,
                        help="Add out-prefix to the output filename.")
    parser.add_argument("--small-dev", action=argparse.BooleanOptionalAction,
                        help="Use a small (5) subset of data, for development purposes.")
    args = parser.parse_args()
    main(args)
