#!/usr/bin/env bash
set -e


# Function to process translation and compute metrics for a given prompt name
#
# Arguments:
#   $1: The prompt_name to use for file paths.
#
function process_translation_and_metrics_de() {
    local prompt_name="$1" # Assign the first argument to a local variable
    local results_file="out/flores/en-de/${prompt_name}-llama-3.2-3b-instruct-contrastive-prompt.txt.run.txt"
    local translation_file="out/flores/en-de/${prompt_name}-llama-3.2-3b-instruct-contrastive-prompt.txt"

    if [ ! -f "$translation_file" ]; then
        echo "$(date +'%H:%M:%S') Building translations to $translation_file"
        python -m scripts.run --model_path llama-3.2-3b-instruct --language_pairs en-de --prompt_contrastive prompts/$prompt_name.txt --out_prefix $prompt_name
    else
        echo "$(date +'%H:%M:%S') Skipping translations to $translation_file"
    fi

    echo "$(date +'%H:%M:%S') Computing comet score from $translation_file"
    echo "COMET" >> "$results_file"
    comet-score -s out/flores/en-de/src.txt -r out/flores/en-de/ref.txt -t "$translation_file" >> "$results_file"

    echo "$(date +'%H:%M:%S') Computing comet referenceless score from $translation_file"
    echo "COMET REFERENCELESS" >> "$results_file"
    comet-score -s out/flores/en-de/src.txt -t "$translation_file" --model Unbabel/wmt22-cometkiwi-da >> "$results_file"

    echo "$(date +'%H:%M:%S') Metrics $translation_file"

    echo "KENDALL TAU" >> "$results_file"
    scripts/metrics.py out/flores/en-de/src.txt "$translation_file" k-tau >> "$results_file"

    echo "FUZZY REORDERING" >> "$results_file"
    scripts/metrics.py out/flores/en-de/src.txt "$translation_file" fuzzy-reordering >> "$results_file"

    echo "SURPRISAL" >> "$results_file"
    scripts/surprisal.py --model LeoLM/leo-hessianai-7b "$translation_file" log_prob_mean log_prob_variance log_prob_t2t_change >> "$results_file"

    echo "$(date +'%H:%M:%S') Processing complete for prompt: $prompt_name"
}


# Function to process translation and compute metrics for a given prompt name
#
# Arguments:
#   $1: The prompt_name to use for file paths.
#   $2: Translations to be paraphrased
function process_paraphrase_and_metrics_de() {
    local prompt_name="$1" # Assign the first argument to a local variable
    local src_name="$2"
    local translations="$3"
    local results_file="out/flores/de-de/${prompt_name}-${src_name}-llama-3.2-3b-instruct-paraphrase-prompt.txt.run.txt"
    local translation_file="out/flores/de-de/${prompt_name}-${src_name}-llama-3.2-3b-instruct-paraphrase-prompt.txt"

    if [ ! -f "$translation_file" ]; then
        echo "$(date +'%H:%M:%S') Building paraphrases to $translation_file"
        echo "Source of translation $translations" >> "$results_file"
        python -m scripts.run --model_path llama-3.2-3b-instruct --language_pairs de-de --prompt_paraphrase prompts/$prompt_name.txt --translations $translations --out_prefix "${prompt_name}-${src_name}"
    else
        echo "$(date +'%H:%M:%S') Skipping paraphrases to $translation_file"
    fi
    echo "$(date +'%H:%M:%S') Computing comet score from $translation_file"
    echo "COMET" >> "$results_file"
    comet-score -s out/flores/en-de/src.txt -r out/flores/en-de/ref.txt -t "$translation_file" >> "$results_file"

    echo "$(date +'%H:%M:%S') Computing comet referenceless score from $translation_file"
    echo "COMET REFERENCELESS" >> "$results_file"
    comet-score -s out/flores/en-de/src.txt -t "$translation_file" --model Unbabel/wmt22-cometkiwi-da >> "$results_file"

    echo "$(date +'%H:%M:%S') Metrics $translation_file"

    echo "KENDALL TAU" >> "$results_file"
    scripts/metrics.py out/flores/en-de/src.txt "$translation_file" k-tau >> "$results_file"

    echo "FUZZY REORDERING" >> "$results_file"
    scripts/metrics.py out/flores/en-de/src.txt "$translation_file" fuzzy-reordering >> "$results_file"

    echo "SURPRISAL" >> "$results_file"
    scripts/surprisal.py --model LeoLM/leo-hessianai-7b "$translation_file" log_prob_mean log_prob_variance log_prob_t2t_change >> "$results_file"

    echo "$(date +'%H:%M:%S') Processing complete for prompt: $prompt_name"
}

#process_translation_and_metrics_de "base"
process_translation_and_metrics_de "connecting_word"
process_translation_and_metrics_de "grammar"
process_translation_and_metrics_de "less_common"
process_translation_and_metrics_de "literal"
process_translation_and_metrics_de "not_natural"
process_translation_and_metrics_de "robotic"
process_translation_and_metrics_de "simple"
process_translation_and_metrics_de "structure"
process_translation_and_metrics_de "less_common_minus02"
process_translation_and_metrics_de "less_common_minus04"
process_translation_and_metrics_de "less_common_minus06"
process_translation_and_metrics_de "less_common_minus08"
process_translation_and_metrics_de "structure_minus02"
process_translation_and_metrics_de "structure_minus04"
process_translation_and_metrics_de "structure_minus06"
process_translation_and_metrics_de "robotic_minus04"
process_translation_and_metrics_de "robotic_minus02"
process_translation_and_metrics_de "robotic_minus06"
process_translation_and_metrics_de "robotic_minus08"
process_translation_and_metrics_de "not_natural_minus02"
process_translation_and_metrics_de "not_natural_minus04"
process_translation_and_metrics_de "not_natural_minus06"
process_translation_and_metrics_de "not_natural_minus08"
process_translation_and_metrics_de "literal_minus02"
process_translation_and_metrics_de "literal_minus04"
process_translation_and_metrics_de "literal_minus06"
process_translation_and_metrics_de "literal_minus08"
process_translation_and_metrics_de "robotic_minus01"
process_translation_and_metrics_de "robotic_minus005"
#process_translation_and_metrics_de "literal_minus01"
#process_translation_and_metrics_de "literal_minus005"
#process_translation_and_metrics_de "not_natural_minus01"
#process_translation_and_metrics_de "not_natural_minus005"
process_translation_and_metrics_de "direct1"
process_translation_and_metrics_de "direct2"
process_translation_and_metrics_de "direct3"
process_translation_and_metrics_de "direct4"
process_translation_and_metrics_de "direct5"
process_translation_and_metrics_de "direct6"
process_translation_and_metrics_de "direct7"
process_translation_and_metrics_de "direct8"
process_translation_and_metrics_de "direct9"
process_translation_and_metrics_de "direct10"
process_translation_and_metrics_de "direct11"
#process_paraphrase_and_metrics_de garbage_paraphrase direct1 out/flores/en-de/direct1-llama-3.2-3b-instruct-contrastive-prompt.txt
process_translation_and_metrics_de "misc1"
process_translation_and_metrics_de "misc2"
process_translation_and_metrics_de "misc3"
process_translation_and_metrics_de "misc4"
process_translation_and_metrics_de "misc5"
process_translation_and_metrics_de "misc6"
process_translation_and_metrics_de "misc7"
process_translation_and_metrics_de "misc8"
process_translation_and_metrics_de "persona1"
process_translation_and_metrics_de "persona2"
process_translation_and_metrics_de "persona3"
process_translation_and_metrics_de "persona4"
process_translation_and_metrics_de "persona5"
process_translation_and_metrics_de "persona6"
process_translation_and_metrics_de "persona7"
process_translation_and_metrics_de "tone1"
process_translation_and_metrics_de "audiance1"
process_translation_and_metrics_de "fewshot1"
#process_paraphrase_and_metrics_de paraphrase1 base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt
#process_paraphrase_and_metrics_de paraphrase2 base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt
#process_paraphrase_and_metrics_de paraphrase3 base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt
#process_paraphrase_and_metrics_de paraphrase4 base out/flores/en-de/base-llama-3.2-3b-instruct-contrastive-prompt.txt





