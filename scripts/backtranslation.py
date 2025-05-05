#!/usr/bin/env python

import argparse, os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def main(args):
  sentences = args.src.readlines()

  de_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
  de_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
  en_de_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
  en_de_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
  back_translated = []
  for sentence in tqdm(sentences):
    input_ids = de_en_tokenizer.encode(sentence, return_tensors="pt")
    outputs = de_en_model.generate(input_ids)
    translated = de_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

    input_ids = en_de_tokenizer.encode(translated, return_tensors="pt")
    outputs = en_de_model.generate(input_ids)
    back_translated.append(en_de_tokenizer.decode(outputs[0], skip_special_tokens=True))

  file_name = os.path.abspath(args.src.name)+".backtranslate"
  with open(file_name, 'w', encoding="utf-8") as f:
    f.write("\n".join(back_translated))
  print(f"Output writen at {file_name}")

example = """example:
  python %(prog)s src.txt en-de
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog=example,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("src", type=argparse.FileType('r', encoding='UTF-8'), help="Sources")
    #parser.add_argument("translation direction", type=str, choices=["en-de", "de-tr", "en-tr", "tr-en"])
    args = parser.parse_args()
    main(args)