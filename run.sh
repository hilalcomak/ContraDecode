#!/usr/bin/env bash
set -e


# Function to process translation and compute metrics for a given prompt name
#
# Arguments:
#   $1: The prompt_name to use for file paths.
#   $2: The dataset to use. "flores", "flores-dev", "wmt20", "wmt19", "wmt18"
#   $3: Target language
function process_translation_and_metrics() {
    local prompt_name="$1" # Assign the first argument to a local variable
    local dataset="$2"
    local tgt_lang="$3"
    local results_file="out/$dataset/en-$tgt_lang/${prompt_name}-llama-3.2-3b-instruct-contrastive-prompt.txt.run.txt"
    local translation_file="out/$dataset/en-$tgt_lang/${prompt_name}-llama-3.2-3b-instruct-contrastive-prompt.txt"

    if [ ! -f "$translation_file" ]; then
        echo "$(date +'%H:%M:%S') Building translations to $translation_file"
        python -m scripts.run --dataset "$dataset" --model_path llama-3.2-3b-instruct --language_pairs "en-$tgt_lang" --prompt_contrastive prompts/$prompt_name.txt --out_prefix $prompt_name
    else
        echo "$(date +'%H:%M:%S') Skipping translations to $translation_file"
    fi

    echo "$(date +'%H:%M:%S') Computing comet score from $translation_file"
    echo "COMET" >> "$results_file"
    comet-score -s "out/$dataset/en-$tgt_lang/src.txt" -r "out/$dataset/en-$tgt_lang/ref.txt" -t "$translation_file" >> "$results_file"

    echo "$(date +'%H:%M:%S') Computing comet referenceless score from $translation_file"
    echo "COMET REFERENCELESS" >> "$results_file"
    comet-score -s "out/$dataset/en-$tgt_lang/src.txt" -t "$translation_file" --model Unbabel/wmt22-cometkiwi-da >> "$results_file"

    echo "$(date +'%H:%M:%S') Metrics $translation_file"

    echo "KENDALL TAU" >> "$results_file"
    scripts/metrics.py "out/$dataset/en-$tgt_lang/src.txt" "$translation_file" k-tau >> "$results_file"

    echo "FUZZY REORDERING" >> "$results_file"
    scripts/metrics.py "out/$dataset/en-$tgt_lang/src.txt" "$translation_file" fuzzy-reordering >> "$results_file"

    echo "SURPRISAL" >> "$results_file"
    scripts/surprisal.py --model LeoLM/leo-hessianai-7b "$translation_file" log_prob_mean log_prob_variance log_prob_t2t_change >> "$results_file"

    echo "$(date +'%H:%M:%S') Processing complete for prompt: $prompt_name"
}


# Function to process translation and compute metrics for a given prompt name
#
# Arguments:
#   $1: The prompt_name to use for file paths.
#   $2: translation prompt name
#   $3: Translations to be paraphrased
#   $4: Dataset
#   $5: Target language
function process_paraphrase_and_metrics() {
    local prompt_name="$1" # Assign the first argument to a local variable
    local src_name="$2"
    local translations="$3"
    local dataset="$4"
    local tgt_lang="$5"
    local results_file="out/$dataset/$tgt_lang-$tgt_lang/${prompt_name}-${src_name}-llama-3.2-3b-instruct-paraphrase-prompt.txt.run.txt"
    local paraphrased_file="out/$dataset/$tgt_lang-$tgt_lang/${prompt_name}-${src_name}-llama-3.2-3b-instruct-paraphrase-prompt.txt"

    if [ ! -f "$paraphrased_file" ]; then
        echo "$(date +'%H:%M:%S') Building paraphrases to $paraphrased_file"
        echo "Source of translation $translations" >> "$results_file"
        python -m scripts.run --dataset "$dataset" --model_path llama-3.2-3b-instruct --language_pairs "$tgt_lang-$tgt_lang" --prompt_paraphrase prompts/$prompt_name.txt --translations $translations --out_prefix "${prompt_name}-${src_name}"
    else
        echo "$(date +'%H:%M:%S') Skipping paraphrases to $paraphrased_file"
    fi
    echo "$(date +'%H:%M:%S') Computing comet score from $paraphrased_file"
    echo "COMET" >> "$results_file"
    comet-score -s "out/$dataset/en-$tgt_lang/src.txt" -r "out/$dataset/en-$tgt_lang/ref.txt" -t "$paraphrased_file" >> "$results_file"

    echo "$(date +'%H:%M:%S') Computing comet referenceless score from $paraphrased_file"
    echo "COMET REFERENCELESS" >> "$results_file"
    comet-score -s "out/$dataset/en-$tgt_lang/src.txt" -t "$paraphrased_file" --model Unbabel/wmt22-cometkiwi-da >> "$results_file"

    echo "$(date +'%H:%M:%S') Metrics $paraphrased_file"

    echo "KENDALL TAU" >> "$results_file"
    scripts/metrics.py "out/$dataset/en-$tgt_lang/src.txt" "$paraphrased_file" k-tau >> "$results_file"

    echo "FUZZY REORDERING" >> "$results_file"
    scripts/metrics.py "out/$dataset/en-$tgt_lang/src.txt" "$paraphrased_file" fuzzy-reordering >> "$results_file"

    echo "SURPRISAL" >> "$results_file"
    case "$tgt_lang" in
      "tr")
        scripts/surprisal.py --model ytu-ce-cosmos/Turkish-Llama-8b-v0.1 "$paraphrased_file" log_prob_mean log_prob_variance log_prob_t2t_change >> "$results_file"
        ;;
      "fi")
        scripts/surprisal.py --model Finnish-NLP/llama-7b-finnish "$paraphrased_file" log_prob_mean log_prob_variance log_prob_t2t_change >> "$results_file"
        ;;
      "ca")
        scripts/surprisal.py --model catallama/CataLlama-v0.1-Base "$paraphrased_file" log_prob_mean log_prob_variance log_prob_t2t_change >> "$results_file"
        ;;
      "lt")
        scripts/surprisal.py --model neurotechnology/Lt-Llama-2-7b-hf "$paraphrased_file" log_prob_mean log_prob_variance log_prob_t2t_change >> "$results_file"
        ;;
      "de")
        scripts/surprisal.py --model LeoLM/leo-hessianai-7b "$paraphrased_file" log_prob_mean log_prob_variance log_prob_t2t_change >> "$results_file"
        ;;
      *)
        echo "Language is neither Turkish nor German, or not set (from case statement)."
        exit 1
        ;;
    esac
    echo "$(date +'%H:%M:%S') Processing complete for prompt: $prompt_name"
}

#process_translation_and_metrics "persona7" "flores-dev" "tr"
#exit 0

#process_translation_and_metrics "base" "flores"
#process_translation_and_metrics "connecting_word" "flores" "de"
#process_translation_and_metrics "grammar" "flores" "de"
#process_translation_and_metrics "less_common" "flores" "de"
#process_translation_and_metrics "literal" "flores" "de"
#process_translation_and_metrics "not_natural" "flores" "de"
#process_translation_and_metrics "robotic" "flores" "de"
#process_translation_and_metrics "simple" "flores" "de"
#process_translation_and_metrics "structure" "flores" "de"
#process_translation_and_metrics "less_common_minus02" "flores" "de"
#process_translation_and_metrics "less_common_minus04" "flores" "de"
#process_translation_and_metrics "less_common_minus06" "flores" "de"
#process_translation_and_metrics "less_common_minus08" "flores" "de"
#process_translation_and_metrics "structure_minus02" "flores" "de"
#process_translation_and_metrics "structure_minus04" "flores" "de"
#process_translation_and_metrics "structure_minus06" "flores" "de"
#process_translation_and_metrics "robotic_minus04" "flores" "de"
#process_translation_and_metrics "robotic_minus02" "flores" "de"
#process_translation_and_metrics "robotic_minus06" "flores" "de"
#process_translation_and_metrics "robotic_minus08" "flores" "de"
#process_translation_and_metrics "not_natural_minus02" "flores" "de"
#process_translation_and_metrics "not_natural_minus04" "flores" "de"
#process_translation_and_metrics "not_natural_minus06" "flores" "de"
#process_translation_and_metrics "not_natural_minus08" "flores" "de"
#process_translation_and_metrics "literal_minus02" "flores" "de"
#process_translation_and_metrics "literal_minus04" "flores" "de"
#process_translation_and_metrics "literal_minus06" "flores" "de"
#process_translation_and_metrics "literal_minus08" "flores" "de"
#process_translation_and_metrics "robotic_minus01" "flores" "de"
#process_translation_and_metrics "robotic_minus005" "flores" "de"
#process_translation_and_metrics "literal_minus01" "flores" "de"
#process_translation_and_metrics "literal_minus005" "flores" "de"
#process_translation_and_metrics "not_natural_minus01" "flores" "de"
#process_translation_and_metrics "not_natural_minus005" "flores" "de"
#process_translation_and_metrics "direct1" "flores" "de"
#process_translation_and_metrics "direct2" "flores" "de"
#process_translation_and_metrics "direct3" "flores" "de"
#process_translation_and_metrics "direct4" "flores" "de"
#process_translation_and_metrics "direct5" "flores" "de"
#process_translation_and_metrics "direct6" "flores" "de"
#process_translation_and_metrics "direct7" "flores" "de"
#process_translation_and_metrics "direct8" "flores" "de"
######################################################
#process_translation_and_metrics "direct9" "flores" "de"
#process_translation_and_metrics "direct10" "flores" "de"
#process_translation_and_metrics "direct11" "flores" "de"
#process_paraphrase_and_metrics garbage_paraphrase direct1 out/flores/en-de/direct1-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "de"
#process_translation_and_metrics "misc1" "flores" "de"
#process_translation_and_metrics "misc2" "flores" "de"
#process_translation_and_metrics "misc3" "flores" "de"
#process_translation_and_metrics "misc4" "flores" "de"
#process_translation_and_metrics "misc5" "flores" "de"
#process_translation_and_metrics "misc6" "flores" "de"
#process_translation_and_metrics "misc7" "flores" "de"
#process_translation_and_metrics "misc8" "flores" "de"
#process_translation_and_metrics "persona1" "flores" "de"
#process_translation_and_metrics "persona2" "flores" "de"
#process_translation_and_metrics "persona3" "flores" "de"
#process_translation_and_metrics "persona4" "flores" "de"
#process_translation_and_metrics "persona5" "flores" "de"
#process_translation_and_metrics "persona6" "flores" "de"
#process_translation_and_metrics "persona7" "flores" "de"
#process_translation_and_metrics "tone1" "flores" "de"
#process_translation_and_metrics "audience1" "flores" "de"
#process_translation_and_metrics "fewshot1" "flores" "de"
#process_paraphrase_and_metrics paraphrase1 base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "de"
#process_paraphrase_and_metrics paraphrase2 base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "de"
#process_paraphrase_and_metrics paraphrase3 base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "de"
#process_paraphrase_and_metrics paraphrase4 base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "de"
#process_translation_and_metrics "phrase_contrast_minus01" "flores" "de"
#process_translation_and_metrics "phrase_contrast_minus02" "flores" "de"
#process_translation_and_metrics "phrase_contrast_minus04" "flores" "de"
#process_translation_and_metrics "phrase_contrast_minus06" "flores" "de"
#process_translation_and_metrics "phrase_contrast_minus08" "flores" "de"
#process_translation_and_metrics "connecting_minus01" "flores" "de"
#process_translation_and_metrics "connecting_minus02" "flores" "de"
#process_translation_and_metrics "connecting_minus04" "flores" "de"
#process_translation_and_metrics "connecting_minus06" "flores" "de"
#process_translation_and_metrics "connecting_minus08" "flores" "de"
#process_paraphrase_and_metrics "paraphrase_sys_1" base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "de"
#process_paraphrase_and_metrics "paraphrase_sys_2" base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "de"
#process_paraphrase_and_metrics "paraphrase_sys_3" base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "de"
#process_paraphrase_and_metrics "paraphrase_sys_4" base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt"flores" "de"
#process_translation_and_metrics "direct11" "flores" "tr"
#process_translation_and_metrics "direct2" "flores" "tr"
#process_translation_and_metrics "literal_minus08" "flores" "tr"
#process_translation_and_metrics "misc2" "flores" "tr"
#process_translation_and_metrics "persona2" "flores" "tr"
#process_translation_and_metrics "fewshot1" "flores" "tr"

#process_translation_and_metrics "base" "flores" "tr"
#process_paraphrase_and_metrics "paraphrase1" base out/flores/en-tr/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "tr"

#process_translation_and_metrics "base" "flores" "ca"
#process_paraphrase_and_metrics "paraphrase1" base out/flores/en-ca/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "ca"

#process_translation_and_metrics "base" "flores" "fi"
#process_paraphrase_and_metrics "paraphrase1" base out/flores/en-fi/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "fi"

#process_translation_and_metrics "base" "flores" "lt"
#process_paraphrase_and_metrics "paraphrase1" base out/flores/en-lt/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores" "lt"


#process_translation_and_metrics "direct11" "flores" "fi"
#process_translation_and_metrics "direct2" "flores" "fi"
#process_translation_and_metrics "literal_minus08" "flores" "fi"
#process_translation_and_metrics "misc2" "flores" "fi"
#process_translation_and_metrics "persona2" "flores" "fi"
#process_translation_and_metrics "fewshot1" "flores" "fi"

#process_translation_and_metrics "direct11" "flores" "ca"
#process_translation_and_metrics "direct2" "flores" "ca"
#process_translation_and_metrics "literal_minus08" "flores" "ca"
#process_translation_and_metrics "misc2" "flores" "ca"
#process_translation_and_metrics "persona2" "flores" "ca"
#process_translation_and_metrics "fewshot1" "flores" "ca" 

#process_translation_and_metrics "direct11" "flores" "lt"
#process_translation_and_metrics "direct2" "flores" "lt"
#process_translation_and_metrics "literal_minus08" "flores" "lt"
#process_translation_and_metrics "misc2" "flores" "lt"
#process_translation_and_metrics "persona2" "flores" "lt"
#process_translation_and_metrics "fewshot1" "flores" "lt"

#process_translation_and_metrics "audience1" "flores" "tr"
#process_translation_and_metrics "audience1" "flores" "fi"
#process_translation_and_metrics "audience1" "flores" "ca"
#process_translation_and_metrics "audience1" "flores" "lt"

#process_translation_and_metrics "base" "flores-dev" "de"
#process_translation_and_metrics "literal_minus08" "flores-dev" "de"
#process_translation_and_metrics "direct11" "flores-dev" "de"
#process_translation_and_metrics "misc2" "flores-dev" "de"
#process_translation_and_metrics "persona2" "flores-dev" "de"
#process_translation_and_metrics "fewshot1" "flores-dev" "de"
#process_translation_and_metrics "audience1" "flores-dev" "de"
#process_paraphrase_and_metrics "paraphrase1" base out/flores-dev/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt "flores-dev" "de"

process_translation_and_metrics "direct3" "flores" "fi"
process_translation_and_metrics "direct3" "flores" "ca"
process_translation_and_metrics "direct3" "flores" "lt"
process_translation_and_metrics "direct3" "flores" "tr"
process_translation_and_metrics "direct3" "flores-dev" "de"