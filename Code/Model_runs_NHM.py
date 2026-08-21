import sys
sys.path.append('..')
from functions import *


### Loading in prompts
# Structured Prompt
with open(Path("prompts/zero_shot_prompt_FP_mixed-2.txt")) as f:
    structured_zeroshot_prompt=f.read()

# Unstructured Prompt
with open(Path("prompts/unstructured_zero_shot_prompt-2.txt")) as f: 
    unstructured_zeroshot_prompt=f.read()

# One/few shot Prompt for Examples (technically, only for structured output)
with open(Path("prompts/one:few_shot_prompt_FP_mixed-2.txt")) as f:
    structured_onefew_shot_prompt=f.read()


### Loading in data
with open("data/mammals_dataset_compressed_3.json", "r", encoding ="utf-8") as f:
    mammals_dataset_compressed = json.load(f)


###### Claude Sonnet Runs


### Structured Zero Shot
run_openrouter_pipeline(
    dataset=mammals_dataset_compressed,
    model = "~anthropic/claude-sonnet-latest",
    examples = [],
    user_prompt = structured_zeroshot_prompt,
    temperature = 0.7,
    output_path = "results/claude_zeroshot_nhm_structured_run-3.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_zeroshot_nhm_structured_run-3.jsonl", 
    output_path="corpus_metrics/claude_zeroshot_nhm_structured_run-3.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_zeroshot_nhm_structured_run-3.jsonl",
    page_types_to_include=["table", "mixed"]
    )


#### Structured one shot run
run_openrouter_pipeline(
    dataset=mammals_dataset_compressed,
    model = "~anthropic/claude-sonnet-latest",
    examples = nhm_specific_oneshot_example,
    user_prompt = structured_onefew_shot_prompt,
    temperature = 0.7,
    output_path = "results/claude_oneshot_nhm_specific_structured_run-1.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_oneshot_nhm_specific_structured_run-1.jsonl", 
    output_path="corpus_metrics/claude_oneshot_nhm_specific_structured_run-1.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_oneshot_nhm_specific_structured_run-1.jsonl",
    page_types_to_include=["table", "mixed"]
    )

##### Structured few shot run
run_openrouter_pipeline(
    dataset=mammals_dataset_compressed,
    model = "~anthropic/claude-sonnet-latest",
    examples = nhm_specific_table_schema_few_shot_examples,
    user_prompt = structured_onefew_shot_prompt,
    temperature = 0.7,
    output_path = "results/claude_fewshot_nhm_specific_structured_run-1.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_fewshot_nhm_specific_structured_run-1.jsonl", 
    output_path="corpus_metrics/claude_fewshot_nhm_specific_structured_run-2.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_fewshot_nhm_specific_structured_run-2.jsonl",
    page_types_to_include=["table", "mixed"]
    )


##### Structured one shot non-specific ex run
run_openrouter_pipeline(
    dataset=mammals_dataset_compressed,
    model = "~anthropic/claude-sonnet-latest",
    examples = bl_specific_one_shot_example,
    user_prompt = structured_onefew_shot_prompt,
    temperature=0.7,
    output_path = "results/claude_unspecific_oneshot_nhm_structured_run-1.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_unspecific_oneshot_nhm_structured_run-1.jsonl", 
    output_path="corpus_metrics/claude_unspecific_oneshot_nhm_structured_run-2.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_unspecific_oneshot_nhm_structured_run-2.jsonl",
    page_types_to_include=["table", "mixed"]
    )

##### Structured few shot non-specific ex run
run_openrouter_pipeline(
    dataset=mammals_dataset_compressed,
    model = "~anthropic/claude-sonnet-latest",
    examples = bl_specific_few_shot_examples,
    user_prompt = structured_onefew_shot_prompt,
    temperature=0.7,
    output_path = "results/claude_mixed_fewshot_nhm_structured_run-1.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_mixed_fewshot_nhm_structured_run-1.jsonl", 
    output_path="corpus_metrics/claude_mixed_fewshot_nhm_structured_run-2.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_mixed_fewshot_nhm_structured_run-2.jsonl",
    page_types_to_include=["table", "mixed"]
    )



