import sys
sys.path.append('..')
from functions import *


# load in data
with open("data/mammals_dataset_compressed_3.json", "r", encoding ="utf-8") as f:
    mammals_dataset_compressed = json.load(f)

with open("data/fullpage_test_data.json", "r", encoding="utf-8") as f:
    fullpage_test_data = json.load(f)

# load in prompt
with open(Path("prompts/one:few_shot_prompt_FP_mixed-2.txt")) as f:
    structured_onefew_shot_prompt=f.read()


###########
###########
# Claude OpenRouter Iterations on NHM Dataset

# Best model set up
run_openrouter_pipeline(
    dataset=mammals_dataset_compressed,
    model = "~anthropic/claude-sonnet-latest",
    examples = nhm_specific_few_shot_examples,
    user_prompt = structured_onefew_shot_prompt,
    temperature = 0.7,
    output_path = "results/claude_fewshot_nhm_specific_structured_iteration-1.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_fewshot_nhm_specific_structured_iteration-1.jsonl", 
    output_path="corpus_metrics/claude_fewshot_nhm_specific_structured_iteration-1.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_fewshot_nhm_specific_structured_iteration-1.jsonl",
    page_types_to_include=["table", "mixed"]
    )



###### Iteration 2
run_openrouter_pipeline(
    dataset=mammals_dataset_compressed,
    model = "~anthropic/claude-sonnet-latest",
    examples = nhm_specific_few_shot_examples,
    user_prompt = structured_onefew_shot_prompt,
    temperature = 0.7,
    output_path = "results/claude_fewshot_nhm_specific_structured_iteration-2.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_fewshot_nhm_specific_structured_iteration-2.jsonl", 
    output_path="corpus_metrics/claude_fewshot_nhm_specific_structured_iteration-2.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_fewshot_nhm_specific_structured_iteration-2.jsonl",
    page_types_to_include=["table", "mixed"]
    )

########### Iteration 3
run_openrouter_pipeline(
    dataset=mammals_dataset_compressed,
    model = "~anthropic/claude-sonnet-latest",
    examples = nhm_specific_few_shot_examples,
    user_prompt = structured_onefew_shot_prompt,
    temperature = 0.7,
    output_path = "results/claude_fewshot_nhm_specific_structured_iteration-3.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_fewshot_nhm_specific_structured_iteration-3.jsonl", 
    output_path="corpus_metrics/claude_fewshot_nhm_specific_structured_iteration-3.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_fewshot_nhm_specific_structured_iteration-3.jsonl",
    page_types_to_include=["table", "mixed"]
    )

############# Iteration 4
run_openrouter_pipeline(
    dataset=mammals_dataset_compressed,
    model = "~anthropic/claude-sonnet-latest",
    examples = nhm_specific_few_shot_examples,
    user_prompt = structured_onefew_shot_prompt,
    temperature = 0.7,
    output_path = "results/claude_fewshot_nhm_specific_structured_iteration-4.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_fewshot_nhm_specific_structured_iteration-4.jsonl", 
    output_path="corpus_metrics/claude_fewshot_nhm_specific_structured_iteration-4.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_fewshot_nhm_specific_structured_iteration-4.jsonl",
    page_types_to_include=["table", "mixed"]
    )


###########
###########
# Claude OpenRouter Iterations on BL Dataset

run_openrouter_pipeline(
    dataset=fullpage_test_data,
    model = "~anthropic/claude-sonnet-latest",
    examples = bl_specific_few_shot_examples,
    user_prompt = structured_onefew_shot_prompt,
    temperature=0.7,
    output_path = "results/claude_fewshot_bl_specific_structured_iteration-1.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_fewshot_bl_specific_structured_iteration-1.jsonl", 
    output_path="corpus_metrics/claude_fewshot_bl_specific_structured_iteration-1.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_fewshot_bl_specific_structured_iteration-1.jsonl",
    page_types_to_include=["text"]
    )

########### Iteration 2
run_openrouter_pipeline(
    dataset=fullpage_test_data,
    model = "~anthropic/claude-sonnet-latest",
    examples = bl_specific_few_shot_examples,
    user_prompt = structured_onefew_shot_prompt,
    temperature=0.7,
    output_path = "results/claude_fewshot_bl_specific_structured_iteration-2.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_fewshot_bl_specific_structured_iteration-2.jsonl", 
    output_path="corpus_metrics/claude_fewshot_bl_specific_structured_iteration-2.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_fewshot_bl_specific_structured_iteration-2.jsonl",
    page_types_to_include=["text"]
    )

############ Iteration 3
run_openrouter_pipeline(
    dataset=fullpage_test_data,
    model = "~anthropic/claude-sonnet-latest",
    examples = bl_specific_few_shot_examples,
    user_prompt = structured_onefew_shot_prompt,
    temperature=0.7,
    output_path = "results/claude_fewshot_bl_specific_structured_iteration-3.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_fewshot_bl_specific_structured_iteration-3.jsonl", 
    output_path="corpus_metrics/claude_fewshot_bl_specific_structured_iteration-3.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_fewshot_bl_specific_structured_iteration-3.jsonl",
    page_types_to_include=["text"]
    )

############ Iteration 4
run_openrouter_pipeline(
    dataset=fullpage_test_data,
    model = "~anthropic/claude-sonnet-latest",
    examples = bl_specific_few_shot_examples,
    user_prompt = structured_onefew_shot_prompt,
    temperature=0.7,
    output_path = "results/claude_fewshot_bl_specific_structured_iteration-4.jsonl"
)

build_per_record_metrics(
    input_path="results/claude_fewshot_bl_specific_structured_iteration-4.jsonl", 
    output_path="corpus_metrics/claude_fewshot_bl_specific_structured_iteration-4.jsonl")

corpus_metrics(
    result_path = "corpus_metrics/claude_fewshot_bl_specific_structured_iteration-4.jsonl",
    page_types_to_include=["text"]
    )