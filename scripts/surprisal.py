#!/usr/bin/env python

import argparse
import numpy as np
from tqdm import tqdm
from metrics import translator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODELS = ["LeoLM/leo-hessianai-7b"]

def get_model_and_tokenizer(args):
  model = AutoModelForCausalLM.from_pretrained(args.model,
                                            device_map='auto',
                                            load_in_4bit=True,
                                            torch_dtype=torch.bfloat16)
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  tokenizer.pad_token = tokenizer.eos_token
  model.config.pad_token_id = model.config.eos_token_id
  return model, tokenizer

#Based on https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
def get_surprisal_per_token(model, tokenizer, input_texts):
  with torch.no_grad():
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    outputs = model(input_ids)
    log_probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    log_probs = log_probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_log_probs = torch.gather(log_probs, 2, input_ids[:, :, None]).squeeze(-1)
    return -gen_log_probs

def log_prob_variance(s):
  # https://arxiv.org/pdf/2306.03734 Eq1
  if s.size < 2:
    return None
  μ = np.mean(s)
  return np.mean(np.square(s - μ))

def log_prob_t2t_change(s):
  # https://arxiv.org/pdf/2306.03734 Eq2
  if s.size < 2:
    return None
  return np.sum(np.square(s[1:] - s[:-1])) / (s.size - 1)

def log_prob_mean(s):
  return np.mean(s)

MEASURES = {
  'log_prob_variance' : log_prob_variance,
  'log_prob_t2t_change' : log_prob_t2t_change,
  'log_prob_mean': log_prob_mean,
}

def main(args):
  measures = {k: list() for k in set(args.measure)}
  sentences = args.src.readlines()
  sentences = [s.translate(translator).strip() for s  in sentences]
  sentences = [s for s in sentences if s != ""]
  model, tokenizer = get_model_and_tokenizer(args)
  for sent in tqdm(sentences):
    s = get_surprisal_per_token(model, tokenizer, [sent])[0].float().cpu().numpy()
    #print(f"{sent}")
    for k in measures:
      r = MEASURES[k](s)
      if r is not None:
        #print(f"\t{k}: {r}")
        measures[k].append(r)

  for k,v in measures.items():
    print(f"{k}: {np.mean(v)}")
  exit(0)

example = """example:
  python %(prog)s src.txt --model LeoLM/leo-hessianai-7b out/flores/en-de/ref.txt log_prob_mean
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog=example,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, choices=MODELS)
    parser.add_argument("src", type=argparse.FileType('r', encoding='UTF-8'), help="Source file")
    parser.add_argument("measure",
                        type=str,
                        nargs='+',
                        choices=MEASURES.keys())
    main(parser.parse_args())
