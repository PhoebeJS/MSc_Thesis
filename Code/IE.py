## Information Extraction Case Study

# Using the transcriptions to perform a small species distribution model on species present within the NHM database (& potentially a subset of the BL? Still debating. Definetly NHM tho because it can then also act as a secondary test for the unstructured transcription). 


# Schema for IE

prompt_format_species = {
    "type": "array",
    "items":{
        "type":"object",
    "properties":{
        "speciesName": {
            "type": "string",
            "description": "Taxonomic species name (latin binomial) or 'spp.'",
            "default": ""
        },
        "verbatimLocality":{
            "type": "string",
            "description":"The original textual description of a location.",
            "default": ""
        },
        "date":{
            "type": "string",
            "description": "The time the species was found, spotted, collected, or identified. May take a numeric form that follows regional conventional. Example: day-month-year, month-day-year, or year-month-day. Numeric dates may use different separators, such as slashes (13/07/1895), hyphens (13-07-1895), and dots (13.07.1895). Numeric dates may also use shortened forms of the date. Example: using '95' instead of '1895' or '7' instead of '07'. Dates may also be written out in long format, where the full month is written out. Example: '13 July 1895' or 'July 13 1895'. In some cases, the day and year may also be written out. Example: 'day 13 of July 1895' or '13 July in year 1895'. Prepositions may also be used. Example: 'the 13th of July, 1895'. Only extract the year.",
            "default": ""
            },
        "verbatimDate":{
            "type": "string",
            "description": "The original textual description of date."
            }
        },
        "required": ["speciesName"],
        "additionalProperties": False
    }
}



# Load in the excel
import pandas as pd

gt_df = pd.read_excel("data/IE_species_v2.xlsx")
gt_df.head()


# Grouping by document id (volume) & page id so i can pull GT rows per page for alignment later
gt_df.columns = gt_df.columns.str.strip()

gt_grouped = gt_df.groupby(["document_id", "page_id"])



### Functions used - altered from functions/py
def build_messages_ie(user_prompt, transcription, sys_prompt=system_prompt, schema=prompt_format_species):
    messages = [
        {
            'role': 'system',
            'content': sys_prompt
        },
        {
            'role': 'user',
            'content': f"""
                {user_prompt}

                Transcription:
                {transcription}

                Provide the information following this JSON schema exactly and output JSON only: {schema}
                """
        }
    ]
    return messages

def call_openrouter_ie(user_prompt, transcription, model, temperature, sys_prompt=system_prompt, schema=prompt_format_species):
    messages = build_messages_ie(user_prompt, transcription, sys_prompt=sys_prompt, schema=schema)
    response = openrouter_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    usage = {
        "prompt_tokens": response.usage.prompt_tokens, # Extracts number of input tokens in api request
        "completion_tokens": response.usage.completion_tokens, # Extracts the number of output tokens from the LLM
        "total_tokens": response.usage.total_tokens, # Total input & output tokens
        "temperature": temperature
    }
    raw_output = response.choices[0].message.content
    return raw_output, usage



def openrouter_ie_pipeline(transcription, user_prompt, output_path, model, temperature, sys_prompt=system_prompt, schema=prompt_format_species, batch_size=10):
    # Check if output file already exists to prevent duplicates
    if os.path.exists(output_path):
        print(f"Warning: {output_path} already exists. Delete it first to avoid duplicates.")
        return
    results_buffer = []   # in-memory storage until we flush
    batch_counter = 0     # For counting how many batches we've gone through
    records = []
    with open(transcription, "r", encoding="utf-8") as f: 
        for line in f:
            record = json.loads(line.strip())
            records.append(record)
    for record in records:
        model_transcription = record["table"]
        document_id = record["document_id"]
        page_id = record["page_id"]  

    # Call openrouter api with current image and prompt
        raw_output, usage = call_openrouter_ie(
            transcription=model_transcription,
            user_prompt=user_prompt,
            sys_prompt=sys_prompt,
            schema=schema,
            model=model,
            temperature=temperature)
        # Then pass string into function
        result = string_to_json(raw_output)
        # Handle case where result is a list instead of a dictionary
        if isinstance(result, dict):
            result = [result]

        for entry in result:
            model_speciesName = entry.get("speciesName", "")
            model_verbatimLocality = entry.get("verbatimLocality", "")
            model_date = entry.get("date", "")
            model_verbatimDate = entry.get("verbatimDate", "")

            result_record = {
                "model_speciesName": model_speciesName,
                "model_verbatimLocality": model_verbatimLocality,
                "model_date": model_date,
                "model_verbatimDate": model_verbatimDate,
                "document_id": document_id,
                "page_id": page_id,
            }
            results_buffer.append(result_record)
            batch_counter += 1

        # Flush logic - checking if counter has reached batch size & should be flushed
        # NB: changing to JSONL logic since when we append json files like this with flushing, it will create an invalid JSON with two separate lists stuck together with no comma/wrapper
        if batch_counter >= batch_size:
            with open(output_path, "a", encoding="utf-8") as f:     
                # json.dump writes our records list into the file as formatted JSON
                # ensure_ascii=False: preserves special characters like & or accented letters as-is
                # indent=2: adds 2 space indentation so the JSON is human readable
                for r in results_buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"✔ Saved batch")

            # reset buffer
            results_buffer = [] 
            batch_counter = 0
    # Final flushing block after the loop ends - catches any remaining records that didn't fill a complete batch
    if len(results_buffer) > 0:
        # Flush remaining records
        with open(output_path, "a", encoding="utf-8") as f:     
            # json.dump writes our records list into the file as formatted JSON
            # ensure_ascii=False: preserves special characters like & or accented letters as-is
            # indent=2: adds 2 space indentation so the JSON is human readable
            for record in results_buffer:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✔ Final batch saved")
        


#####
# Now using with openrouter
import sys
sys.path.append('..')
from functions import *

# Load in prompt
with open(Path("prompts/prompt_species_v2.txt")) as f:
    species_extraction_prompt_2 = f.read()


# Running for claude few-show, dataset specific examples, structured prompt, 0.7 temperature
openrouter_ie_pipeline(
    transcription="extracted_transcriptions_IE.jsonl",
    model="~anthropic/claude-sonnet-latest",
    output_path="model_species_extractions.jsonl",
    user_prompt=species_extraction_prompt_2,
    temperature=0.7
)
