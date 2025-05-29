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


process_translation_and_metrics_de "garbage"
