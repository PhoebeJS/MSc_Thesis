
#### Necessary Packages

import os
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import random # Random seed
import cv2
from json_repair import repair_json
#### For main loop & model performance
import time
from ollama import Client
from jiwer import cer, wer 
import Levenshtein
from rapidfuzz.distance import Levenshtein as RapidLevenshtein
import statistics
from dotenv import load_dotenv
from openai import OpenAI
import base64
from PIL import Image, ImageDraw
from PIL import ImageFont
from pathlib import Path
from thefuzz import fuzz
from collections import Counter # Counts occurrences of each unique value in a list
import re
import unicodedata
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
from scipy.stats import rankdata
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIConnectionError, APITimeoutError
from ollama import Message, Image

BASE_DIR = Path(__file__).parent            # File path of the current script, with it's parent directory being extracted; allows the cript to find its own local assets
DATA_DIR = BASE_DIR/"data" # Path to data


# client updated for HPC
host = "http://ollama.runai-shared.svc.cluster.local"
client = Client(host=host)


load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
    max_retries=2,
    http_client=httpx.Client(
        limits=httpx.Limits(
            max_keepalive_connections=5,
            keepalive_expiry=15.0
        )
    )
)



################
################

## SHARED/CONSTANT FUNCTIONS, SCHEMA, AND PROMPTS


# System prompt held constant
# Purpose: Loading in the prompt txt as a string
with open(BASE_DIR/"Prompts"/"system_prompt-1.txt") as f:
    system_prompt = f.read()



# Schema updated to deal with tables
# Purpose: this is the schema meant to keep the LLM output structured & to help the LLM navigate through the information (hence the description). Note the schema is a dictionary and the values wihtin it keys
# The page_type field is designed to let the model self-classify each page & determine which of text_transcription or table should be meaningfully populated
table_prompt_format_transcriptions = {
    #"title": "TranscriptionTable",
    "type": "object", # Model knows output must be JSON object
    "properties": {      
        # Each key here defines one field the model may populate. Description is the lever for steering behaviour since the model will read it as a natural-language instruction on how to fill that field
        "page_type": {
            # Tells the model there are 3 allowed values (text, table & mixed)
            "type": "string",
            "enum": ["text", "table", "mixed"],
            "description": "Classify the page as 'text' if it contains only prose/unstructured text, 'table' if it contains only a table, or 'mixed' if it contains both."
        },
        "text_transcription": {
            # Tells the model when to populate the key & how. Default is a fallback if the model doesn't fill
            "type": "string",
            "description": "A full length transcription of the image input that follows the natural reading order of the page. Only include text within the image. Do not add or adjust language or make grammatical corrections. Required if page_type is 'text' or 'mixed', otherwise null.",
            "default": "Could not transcribe."
        },
        "table": {
            # Table is a nested object so the model first has to decide is this record needs a table field & then within it, populate headers & rows, which each have their own descriptions guiding format
            "type": "object",
            "description": "Required if page_type is 'table' or 'mixed', otherwise null.",
            "properties": {
                "headers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column headers from the table, transcribed faithfully."
                },
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": "Each row as a list of cell values, in column order matching the headers."
                }
            },
            "required": ["headers", "rows"]
        }
    },
    "required": ["page_type"],
    "additionalProperties": False
}


# String to JSON function
# Purpose: The model outputs a string but we want a structured JSON output 
def string_to_json(json_string):
    if isinstance(json_string, str): # If the object is a string...
        # Remove any leading/training whitespace from string
        cleaned = json_string.strip()

        # Strip thinking tokens if present
        if "<think>" in cleaned:
            cleaned = cleaned[cleaned.rfind("</think>") + len("</think>"):].strip()
        
        # Remove markdown fences (case insensitive)
        if cleaned.lower().startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        # remove closing fence if present
        if cleaned.endswith("```"):
            cleaned = cleaned[:-len("```")].strip()

        # Now try parsing cleaned sring into python dictionary
        # json.loads converts JSON string to dictionary (the opposite of json.dumps)
        try:
            json_string = json.loads(cleaned)
        # If parsing fails - i.e., output isn't valid json - print an error message & return none rather than crashing
        except json.JSONDecodeError:
            # Try to repair malformed JSON before giving up
            try:
                repaired = repair_json(cleaned)
                return json.loads(repaired)
            except:
                print("Error in string_to_json. Returning None.")
                print(f"Raw output that failed to parse: {cleaned[:200]}")  # Print the first 200 characters of cleaned to see what the model outputted
                return None

    return(json_string) # Return the output 

#######################
#######################

# Examples

# One-shot example string 
# BL Dataset Specific
bl_specific_one_shot_example = [
    {
        "image_path": str(DATA_DIR/"ior_htr_groundtruth"/"ior!p!242!38_Feb_1802_pp_406-24"/"0001_ior!p!242!38_Feb_1802_pp_406-24_f001r.jpg"),
        "transcription": "406\n\nFort St George - February 1802\n\nQuality of the Investment on a Salary of Pagodas 200 for Month.– \n\nFort St George 10th February 1802 \n\nI am &ca GG Keble Secry to Government \n\nReceived the following Letter To the Right Honble Lord Clive Governor in Council &ca &ca Fort St George. \n\nMy Lord, I have the honor of enclosing a report of the farther specimens of fine grained woods I have been able to procure for - - - the Honble the Court of Directors and have to request your Lordship's Orders for the few specified, in Number nine, being Shipped on Board one of the Indiamen now under dispatch – I have noticed in the report that I have about 100 speci mens of small sized woods of which an Account will be given as soon as they are sufficiently \n\ndried \n\nLetter from Dr Berry-Sub mitting a Report of further Specimens of fine grained word procured for trans mission to England request an order for their being received on Board of the India man–will hereafter submit an account of some smallsized word already procured - The amount  pence submitted with observations Requests instructions in Regard to the prosecution or otherwise of the researches for Speci mens and repeat the accommodation Peon"
    }
]


# Few-shot examples string 
# BL Dataset Specific
bl_specific_few_shot_examples = [
    {
        "image_path": str(DATA_DIR/"ior_htr_groundtruth"/"ior!p!241!31_pp597-99_744-46"/"0004_P_241_31_004.jpg"),
        "transcription": "744 \n\nFort St George 13th March 1792 \n\nResolved that the Resident at Ingeram be informed that the Board's orders to him of the 5th Ultimo, authorized the interference of the Chief and Council as well with respect to the Price as to the distribution of all grain consigned to him or imported at Coringa on the Company's Account -- \n\norder thereon \n\nRead the following Letter from Doctor Anderson with the Paper accompanying it \n\n(Entered M.B. No 363) \n\nTo the Honble Sir Charles Oakeley Bart \n\nActing Governor and Council \n\nFort St George March 12th 1792 \n\nHonble Sirs \n\nBy Captain Simpson who brought the Tallow and Lacquer Trees in safety from China I have sent to the Mallabar Coast five Cart Load of Nopals chiefly of the sort that came from Kew garden, and hav-ing a perfect reliance on the integrity and attention of this Gentleman, I have the honor to enclose the Copy of his answer to me, which you will be pleas'd to transmit to the Government at Bombay with a requisition \n\non \n\nLetter from Dr Anderson Nopals delivered to Capt Simpson for the Malabar Coasts - requests that a Letter be written to Bombay relative to the care of them, as of great use on the receiving of the Cochineal Insect from America - no trees yet received from Sumatra - tho' promised. requests another application by the Asia, & for two kinds of Bread Fruit Trees -"
    },
    {
        "image_path": str(DATA_DIR/"ior_htr_groundtruth"/"ior!p!274!40_17_Jun_1786_pp_1251-66"/"0003_ior!p!274!40_17_Jun_1786_pp_1251-66_f002r.jpg"),
        "transcription": "1253 \n\nFort St George 17th June 1786 \n\nSaheb has prohibited the Exportation of Pepper, particularly, as well as other Articles of Trade from the Mallabar Coast. \n\nWhen the Board of Masulipatam are furnished with Extracts from Mr Roxburgh's Letters they may receive Instructions to give him such immediate Assistance as he may require. I take it for granted that this Government will an approve the Undertaking. And Address from the President to the Zemindars Suggapetty Rawze of Peddipore, and Mahapetty Row of Pettapore will produce the necessary Sanction for ground to cultivate, which may be charged to the Company at a medium Rate of Cultivation \n\nIt may be necessary to add that I am informed from good authority that were the Government of Bombay to pay the Rajah of Travencore his Arrears, amounting to between three and four Lacks of Rupees, he would be \n\nable"
    },
    {
        "image_path": str(DATA_DIR/"ior_htr_groundtruth"/"ior!p!249!17_15_Feb_1853_pp_747-903"/"0103_ior!p!249!17_15_Feb_1853_pp_747-903_f052r.jpg"),
        "transcription": "849\n\nFort Saint George 15th February 1853\n\nTable 1st\n\nOotacamund\n\nHortl. Gardens\n\n17th December 1852\n\nSigned W. G. McIvor\nSupt. Hortl. Garden\n\n(True Copies)\n\nSigned Robert Wight.\n\nTable", "table": {"headers": ["Months", "Estimated amount realized by sales, collections and Donations.", "Value of seeds given gratis to subscribers including a reduction of 25 per cent on the selling prices of all purchases made by them.", "Amount realized from the sale of Produce", "Value of Produce sold but money not yet realized", "Monthly expenditure of the Garden Establishment"], "rows": [["From 25th Decr. 1851 to 25th Jany. 1852", "43 „ „", "39 „ „", "24 „ „", "37 6 „", "119 5 „"], ["„ 25th January to 25th February", "14 „ „", "67 12 „", "84 2 „", "„ „ „", "118 3 „"], ["„ 25th February to 25th March", "21 „ „", "49 „ „", "25 1 „", "„ „ „", "121 6 „"], ["„ 25th March to 25th April", "19 „ „", "1 8 „", "24 8 „", "„ „ „", "137 10 1"], ["„ 25th April to 25th May", "23 „ „", "22 12 „", "39 2 „", "„ „ „", "141 14 „"], ["„ 25th May to 25th June", "19 „ „", "78 1 „", "83 2 „", "„ „ „", "174 3 8"], ["„ 25th June to 25th July", "20 „ „", "44 „ „", "87 2 „", "26 6 „", "195 „ „"], ["„ 25th July to 25th August", "21 „ „", "111 9 „", "192 7 „", "18 2 „", "190 „ „"], ["„ 25th August to 25th Sept.", "17 „ „", "24 „ „", "209 1 „", "„ „ „", "117 „ „"], ["„ 25th September to 25th October", "30 „ „", "110 „ „", "430 9 „", "„ „ „", "133 10 8"], ["„ 25th October to 25th Novr.", "25 „ „", "11 „ „", "162 6 „", "304 „ „", "155 14 0"], ["„ 25th Novr. to 25th Decr.", "25 „ „", "27 2 „", "168 „ „", "89 „ „", "145 6 „"], ["277 „ „", "585 12 „", "1527 8 „", "474 4 „", "1755 8 11"]]}
    }
]

# One-Shot Example String
# NHM Dataset Specific
nhm_specific_oneshot_example = [
    {
        "image_path": str(DATA_DIR/"Mammal_Osteology_held_out_compressed_3"/"Vol1"/"NHM-UK_A_DF_ZOO_218_12_1_014_M_1.jpg"),
        "transcription": json.dumps({
            "headers": ["Year + Collection Number", "Species", "Sex", "Life Stage", "Location", "Collector/Donor", "Collection", "Notes"], 
            "rows": [
                ["1914.V.14.1", "Mirounga leoninus", "♂", "ad.", "S.Georgia.", "Pres. Rupert Vallentin Esq. Carwinion Vean Mawnan, Falmouth.", "14.V.14.", "The os-penis bone."], 
                ["1914.V.18.1", "Rhinoceros bicornis", "♀", "juv", "Mombasa.", "H. Woosnam Esq.", "18.V.14.", "a foetus ♀ received in pickle."], 
                ["1914.VI.27.1", "Hippopotamus liberiensis. (this is 21 on the skeleton!)", "♀", "ad.", "Moa River, near Daru, Sierra Leone.", "Purch.", "", "an ad ♀ shot by R.M.S. Baynes Esq. Skull exhibited. Reg. no in Mammal Registr. = 14.6.21.1."], 
                ["1914.VII.4.1", "Otaria jubata", "♂", "", "Falkland Islands", "Mr. W. Harding Esq. the Falkland Isd. Co.", "20.XI.13.", ""], 
                ["1914.VII.5.1", "Meles meles.", "", "♀", "Midhurst, Sussex.", "H.A. Price Esq.", "28.3.13", ""], 
                ["1914.VII.10.1", "Tapirus americanus", "", "", "Brazil.", "Zoological Society.", "22.V.13.", "Newly born, in captivity."], 
                ["1914.VIII.15.1", "Dolichotis patachonica", "", "", "\"", "Horsham, Sussex.", "18.XII.13.", "Sir Edmund Loder, Bart. Leonardslee, Horsham. Bred in captivity."], 
                ["1914.VIII.17.1", "Ovis lervia", "", "♂", "Africa", "Leonardslee, Horsham, Sussex", "21.4.13", "Sir Edmund Loder Bart. Captive specimen."], 
                ["1914.VIII.20.1", "Equus burchelli chapmani", "", "♂", "", "Zoological Gardens.", "19.4.13.", "Died at birth."], 
                ["1914.VIII.22.1", "Antilope Cervi-Capra.", "", "♀", "", "Leonardslee, Horsham Sussex.", "7.1.13.", "Sir Edmund Loder, bred. A foetus taken from this specimen was preserved in spirit."], 
                ["1914.VIII.28.1", "Giraffa camelopardalis antiquorum", "", "♂", "", "Zoological Gardens London", "11.VI.13.", "Newly born, skin mounted for gallery."]
            ]
        })
    }
]


# Few-shot Example String
# NHM Specific
nhm_specific_few_shot_examples = [
    {
        "image_path": str(DATA_DIR/"Mammal_Osteology_held_out_compressed_3"/"Vol1"/"NHM-UK_A_DF_ZOO_218_12_1_014_M_1.jpg"),
        "transcription": json.dumps({
            "headers": ["Year + Collection Number", "Species", "Sex", "Life Stage", "Location", "Collector/Donor", "Collection", "Notes"], 
            "rows": [
                ["1914.V.14.1", "Mirounga leoninus", "♂", "ad.", "S.Georgia.", "Pres. Rupert Vallentin Esq. Carwinion Vean Mawnan, Falmouth.", "14.V.14.", "The os-penis bone."], 
                ["1914.V.18.1", "Rhinoceros bicornis", "♀", "juv", "Mombasa.", "H. Woosnam Esq.", "18.V.14.", "a foetus ♀ received in pickle."], 
                ["1914.VI.27.1", "Hippopotamus liberiensis. (this is 21 on the skeleton!)", "♀", "ad.", "Moa River, near Daru, Sierra Leone.", "Purch.", "", "an ad ♀ shot by R.M.S. Baynes Esq. Skull exhibited. Reg. no in Mammal Registr. = 14.6.21.1."], 
                ["1914.VII.4.1", "Otaria jubata", "♂", "", "Falkland Islands", "Mr. W. Harding Esq. the Falkland Isd. Co.", "20.XI.13.", ""], 
                ["1914.VII.5.1", "Meles meles.", "", "♀", "Midhurst, Sussex.", "H.A. Price Esq.", "28.3.13", ""], 
                ["1914.VII.10.1", "Tapirus americanus", "", "", "Brazil.", "Zoological Society.", "22.V.13.", "Newly born, in captivity."], 
                ["1914.VIII.15.1", "Dolichotis patachonica", "", "", "\"", "Horsham, Sussex.", "18.XII.13.", "Sir Edmund Loder, Bart. Leonardslee, Horsham. Bred in captivity."], 
                ["1914.VIII.17.1", "Ovis lervia", "", "♂", "Africa", "Leonardslee, Horsham, Sussex", "21.4.13", "Sir Edmund Loder Bart. Captive specimen."], 
                ["1914.VIII.20.1", "Equus burchelli chapmani", "", "♂", "", "Zoological Gardens.", "19.4.13.", "Died at birth."], 
                ["1914.VIII.22.1", "Antilope Cervi-Capra.", "", "♀", "", "Leonardslee, Horsham Sussex.", "7.1.13.", "Sir Edmund Loder, bred. A foetus taken from this specimen was preserved in spirit."], 
                ["1914.VIII.28.1", "Giraffa camelopardalis antiquorum", "", "♂", "", "Zoological Gardens London", "11.VI.13.", "Newly born, skin mounted for gallery."]
            ]
        })
    },
    {
        "image_path": str(DATA_DIR/"Mammal_Osteology_held_out_compressed_3"/"Vol2"/"NHM-UK_A_643_DFZOO_218_12_2_086_M_1.jpg"),
        "transcription": json.dumps({
            "headers": ["Year + Collection NUmber", "Species", "Sex", "Life Stage", "Location", "Body Part", "Collector/Donator"], 
            "rows": [
                ["1949.2.3.16", "Mirounga leonina", "", "juv.", "", "Skull.", "Rothschild Coll. Tring."], 
                ["1949.2.3.17", "Leptonychotes weddelli", "", "juv", "", "Skull. no lower jaw.", "\"\""], 
                ["1949.2.3.18", "Otaria stelleri", "", "", "","Skull", "\"\""], 
                ["1949.3.17.1", "Arctocephalus australis", "♂", "", "Falkland Is.", "Skull. no lower jaw", "J.E. Hamilton Coll."], 
                ["1949.3.17.2", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.3", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.4", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.5", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.6", "\"\"", "♂", "", "\"", "\"", "\""], 
                ["1949.3.17.7", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.8", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.9", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.10", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.11", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.12", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.13", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.14", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.15", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.16", "\"\"", "", "", "\"", "Skull + jaw", "\""], 
                ["1949.3.17.17", "\"\"", "", "", "\"", "Skull, no l. jaw.", "\""], 
                ["1949.3.17.18", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.19", "\"\"", "", "", "\"", "\"", "\""], 
                ["1949.3.17.20", "\"\"",  "", "", "\"", "\"", "\""], 
                ["1949.3.17.21", "\"\"", "♂", "", "\"", "\"", "\""], 
                ["1949.3.17.22", "\"\"", "", "", "\"", "\"", "\""]
            ]
        })
    },
    {
        "image_path": str(DATA_DIR/"Mammal_Osteology_held_out_compressed_3"/"Vol3"/"NHM-UK_A_DF_ZOO_218_12_3_051_M_1.jpg"),
        "transcription": json.dumps({
            "headers": ["Year + Collection Number", "Species", "Sex", "Life Stage", "Location", "Body Part", "Collector/Donator", "Notes"], 
            "rows": [
                ["1968.9.26.7.", "Arctocephalus forsteri.", "♂", "Pup", "Seal Rock, Recherche Archipelago, W. Australia.", "Skull, skeleton.", "Coll. Miss J.E. King.", "Found dead. Umb. cord off - scar visible. JEK 5. No obvious injury. Lengths:- Tip of snout to tip of tail. 73cm. \" \" \" \" \" \" hind flips. 82cm. \" \" \" \" \" \" fore flip. 58cm. \" \" \" \" ant. corner of eye. 6cm, \" \" \" \" tip of ear. 16cm. Nostril 1.5cm. Eye 2cm Ear. 4cm. Tail. 5cm."], 
                ["1968.9.26.8.", "Arctocephalus forsteri.", "♂", "Pup", "Seal Rock, Recherche Archipelago, W. Australia.", "Skull, skeleton.", "Coll. Miss J.E. King.", "Found dead. Umb cord off - scar visible. JEK 6. No obvious injury. Very rotten. Lengths:- Tip of snout to tip of tail. 70cm. \" \" \" \" \" \" hind flips. 78cm. \" \" \" \" \" \" fore flip. 54cm."], 
                ["1968.9.26.9.", "Arctocephalus forsteri.", "♂", "", "Seal Rock, Recherche Archipelago, W. Australia.", "Skull. (No l. jaw)", "Coll. Miss J.E. King.", "JEK 7."], 
                ["1968.9.26.10.", "Arctocephalus doriferus.", "♂", "", "Seal Rocks, Phillip Id., Victoria.", "Skull + skeleton.", "Coll. Miss J.E. King.", "Shot ca 9am Oct. 10th 1967. JEK 17. Lengths:- Tip of snout to tip of tail. 221cm. \" \" \" \" \" hind flips. 250cm. \" \" \" \" ♂ opening. 190cm. \" \" \" \" tip of fore flip. 163cm. Eye. 3.5cm Cornea 3cm x 2½cm. Ear. 5cm. Tail. 8cm. Spread hind flips at nail level. 28cm. Length hind nails (mid ones) 3cm. Spread fore flips at nail level 31cm. Girth behind fore flips. 150cm. Oesophagus ca. 1M. Small Intestine. 3017cm. Caecum. c. 2cm. Large Intestine. 176cm. Weight Heart. 1008gm. Stomach empty except for a few nematodes. Intestine searched and one? tapeworm found near caecum. Cysticerci found in bladder."], 
                ["1968.9.26.11.", "Arctocephalus doriferus.", "♀", "", "Seal Rocks, Phillip Id., Victoria.", "Skull + skeleton.", "Coll. Miss J.E. King.", "Shot Oct. 11th 1967. Pregnant ♀. JEK 18. Last years pup seen suckling after ♀ was shot. Length:- Tip of snout to tip of tail. 172cm. \" \" \" \" \" hind flips. 198cm. \" \" \" \" teats. ant. 111cm post. 137cm. Dist. between teats. ant. 22cm post. 12cm. Tip of snout to tip fore flip. 126cm. \" \" \" \" ant. corner eye. 10cm. \" \" \" \" tip ear. 25cm. Eye. 3cm. Ear. 3cm. Girth post to fore flips. 117cm. Oesophagus. 68cm. Small intestine. 2825cm. Caecum. 2cm. Large Intestine. 225cm. Weight R. Kidney 427.5gm. \" Heart. 656gm."]
            ]
        })
    }
]



# Few-shot Examples
# Mixed from both datasets
mixed_few_shot_examples = [
    {
        "image_path": str(DATA_DIR/"ior_htr_groundtruth"/"ior!p!242!38_Feb_1802_pp_406-24"/"0001_ior!p!242!38_Feb_1802_pp_406-24_f001r.jpg"),
        "transcription": "406 \n\nFort St George - February 1802 \n\nQuality of the Investment on a Salary of Pagodas 200 for Month.– \n\nFort St George 10th February 1802 \n\nI am &ca GG Keble Secry to Government \n\nReceived the following Letter To the Right Honble Lord Clive Governor in Council &ca &ca Fort St George. \n\nMy Lord, \n\nI have the honor of enclosing a report of the farther specimens of fine grained woods I have been able to procure for - - - the Honble the Court of Directors and have to request your Lordship's Orders for the few specified, in Number nine, being Shipped on Board one of the Indiamen now under dispatch – I have noticed in the report that I have about 100 specimens of small sized woods of which an Account will be given as soon as they are sufficiently \n\ndried \n\nLetter from Dr Berry-Sub mitting a Report of further Specimens of fine grained word procured for transmission to England request an order for their being received on Board of the India man–will hereafter submit an account of some smallsized word already procured - The amount  pence submitted with observations Requests instructions in Regard to the prosecution or otherwise of the researches for Specimens and repeat the accommodation \n\nPeon"
    },
    {
        "image_path": str(DATA_DIR/"ior_htr_groundtruth"/"ior!p!241!31_pp597-99_744-46"/"0004_P_241_31_004.jpg"),
        "transcription": "744 \n\nFort St George 13th March 1792 \n\nResolved that the Resident at Ingeram be informed that the Board's orders to him of the 5th Ultimo, authorized the interference of the Chief and Council as well with respect to the Price as to the distribution of all grain consigned to him or imported at Coringa on the Company's Account -- \n\norder thereon \n\nRead the following Letter from Doctor Anderson with the Paper accompanying it \n\n(Entered M.B. No 363) \n\nTo the Honble Sir Charles Oakeley Bart \n\nActing Governor and Council \n\nFort St George March 12th 1792 \n\nHonble Sirs \n\nBy Captain Simpson who brought the Tallow and Lacquer Trees in safety from China I have sent to the Mallabar Coast five Cart Load of Nopals chiefly of the sort that came from Kew garden, and hav-ing a perfect reliance on the integrity and attention of this Gentleman, I have the honor to enclose the Copy of his answer to me, which you will be pleas'd to transmit to the Government at Bombay with a requisition \n\non \n\nLetter from Dr Anderson Nopals delivered to Capt Simpson for the Malabar Coasts - requests that a Letter be written to Bombay relative to the care of them, as of great use on the receiving of the Cochineal Insect from America - no trees yet received from Sumatra - tho' promised. requests another application by the Asia, & for two kinds of Bread Fruit Trees -"
    },
    {
        "image_path": str(DATA_DIR/"Mammal_Osteology_held_out_compressed_3"/"Vol1"/"NHM-UK_A_DF_ZOO_218_12_1_014_M_1.jpg"),
        "transcription": json.dumps({
            "headers": ["Year + Collection Number", "Species", "Sex", "Life Stage", "Location", "Collector/Donor", "Collection", "Notes"], 
            "rows": [
                ["1914.V.14.1", "Mirounga leoninus", "♂", "ad.", "S.Georgia.", "Pres. Rupert Vallentin Esq. Carwinion Vean Mawnan, Falmouth.", "14.V.14.", "The os-penis bone."], 
                ["1914.V.18.1", "Rhinoceros bicornis", "♀", "juv", "Mombasa.", "H. Woosnam Esq.", "18.V.14.", "a foetus ♀ received in pickle."], 
                ["1914.VI.27.1", "Hippopotamus liberiensis. (this is 21 on the skeleton!)", "♀", "ad.", "Moa River, near Daru, Sierra Leone.", "Purch.", "", "an ad ♀ shot by R.M.S. Baynes Esq. Skull exhibited. Reg. no in Mammal Registr. = 14.6.21.1."], 
                ["1914.VII.4.1", "Otaria jubata", "♂", "", "Falkland Islands", "Mr. W. Harding Esq. the Falkland Isd. Co.", "20.XI.13.", ""], 
                ["1914.VII.5.1", "Meles meles.", "", "♀", "Midhurst, Sussex.", "H.A. Price Esq.", "28.3.13", ""], 
                ["1914.VII.10.1", "Tapirus americanus", "", "", "Brazil.", "Zoological Society.", "22.V.13.", "Newly born, in captivity."], 
                ["1914.VIII.15.1", "Dolichotis patachonica", "", "", "\"", "Horsham, Sussex.", "18.XII.13.", "Sir Edmund Loder, Bart. Leonardslee, Horsham. Bred in captivity."], 
                ["1914.VIII.17.1", "Ovis lervia", "", "♂", "Africa", "Leonardslee, Horsham, Sussex", "21.4.13", "Sir Edmund Loder Bart. Captive specimen."], 
                ["1914.VIII.20.1", "Equus burchelli chapmani", "", "♂", "", "Zoological Gardens.", "19.4.13.", "Died at birth."], 
                ["1914.VIII.22.1", "Antilope Cervi-Capra.", "", "♀", "", "Leonardslee, Horsham Sussex.", "7.1.13.", "Sir Edmund Loder, bred. A foetus taken from this specimen was preserved in spirit."], 
                ["1914.VIII.28.1", "Giraffa camelopardalis antiquorum", "", "♂", "", "Zoological Gardens London", "11.VI.13.", "Newly born, skin mounted for gallery."]]
        })
    }
]


######################
######################

# Post-Transcription Analysis & Processing

# Normalise 
# # Purpose: for normalising textual differences that don't have anything to do with actual transcription ability
def normalize(text):

    text = re.sub(r"^\d+\s+", "", text) # Find matches of a pattern in the text, and then replace them (here with a deletion). Strips a leading number (page number) followed by spaces or line breaks from the start of text 

    text = unicodedata.normalize("NFKC", text) # Normalization Form KC (Compatibility Composition) - K maps compatability characters to a standard equivalent and C combines base-diacritic sequences into single codepoints

    text = text.replace("\r\n", "\n")

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("“", '"').replace("”", '"') 
    text = text.replace("N°", "No.")
    text = text.replace("♂", "m").replace("♀", "f")
    text = text.lower()

    # join line-break hyphenation marked with "="
    text = re.sub(r"(\w+) =(\w+)", r"\1\2", text)

    # collapse spaces only
    text = re.sub(r"[ \t]+", " ", text)

    # remove trailing spaces
    text = re.sub(r" *\n *", "\n", text)

    # collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalising number of spaces around &
    text = re.sub(r" *& *", " & ", text)

    return text.strip()



# Alignment
# Purpose: to create some measure of structure accuracy

# First allowing the model to align blocks by searching if adding blocks to one another produces a match. Saves every intermediate span as a separate candidate
def merge_prediction_candidates(blocks, max_merge=6):
    candidates = []
    for start in range(len(blocks)): # For each possible starting block 
        merged = "" # Reset the accumulated text from this point
        for end in range(start, min(start + max_merge, len(blocks))): # Starting from the start block, extend up to the max number of blocks (capped at 6 here)
            if merged: # If merged isn't empty, add a newline separator before appending the next block
                merged += "\n" # This is the newline separator
            merged += blocks[end] # Appends the next block's text
            candidates.append({ # Records the span (start to end) as one candidate & store the start and end index & the merged text
                "start": start,
                "end": end,
                "text": merged
            })

    return candidates



# Aligning blocks
# Purpose: takes in the gt blocks & the model's predicted blocks, performs merging to find possible prediction candidates & then applies penalties for how many extra blocks were merged
def align_blocks(gt_blocks, pred_blocks):
    # Create possible merged prediction blocks & assign them to pred_candidates
    pred_candidates = merge_prediction_candidates(
        pred_blocks,
        max_merge=4
    )
    # Creating a new array called 'cost', with a shape matching the length of the GT blocks & the pred_candidates (currently filled with 0's)
    cost = np.zeros( 
        (
            len(gt_blocks),
            len(pred_candidates)
        )
    )
    for i, gt in enumerate(gt_blocks):        # Takes a collection (gt_blocks) & returns it as an enumerate object, adding a counter
        for j, candidate in enumerate(pred_candidates): # Same thing with the pred_candidates
            # Calculate the lev distance between the GT text & the model text (now merged into different possible blocks)
            distance = normalized_levenshtein(  
                gt,
                candidate["text"]
            )
            # Counts how many blocks got merged into the candidate. Adds 1 to correct for undercounting 
            n_blocks = (
                candidate["end"] - candidate["start"] + 1
            )
            penalty = 0.05 * (n_blocks - 1) # Applies a penality of 0.05 times the number of merged blocks minus 1. I.e., penalises for how many extra blocks beyond a single block were merged in
            cost[i, j] = distance + penalty # Create a cost list composed of the lev distance between the text transcriptions & the penalty score
    rows, cols = linear_sum_assignment(cost) # Uses the Hungarian algorithm to solve the linear alignment problem (finding minimum cost, one-to-one matching between sets) - here, finding the optimal match between GT & pred based on the cost function
    matches = []
    for r, c in zip(rows, cols): # returns a zip object, which is an iterator of tuples where the first item in each passed iterator is paired together. r and c pair with each other. Each pair means "ground-truth block r is optimally matched to prediction-candidate c"
        matches.append({
            "gt": int(r), # convert the GT block's row index to a plain Python int
            "pred_start": pred_candidates[c]["start"], # Find the start of the prediction candidate
            "pred_end": pred_candidates[c]["end"], # Find the end of the prediction candidate
            "alignment_cost": cost[r, c] # Look up the pre-computed cost (distance + merge penalty) for this specific GT-to-prediction match
        })

    return matches


# Layout
def layout_score(matches, n_gt_blocks, cost_threshold=0.5):
    """
    Order + completeness scoring for aligned blocks.
 
    Matches whose alignment_cost exceeds cost_threshold are treated as unreliable (likely a bad forced pairing from the Hungarian assignment, e.g. from over-segmentation) and excluded from the order calculation entirely — rather than silently contributing a possibly-wrong coordinate to Kendall's tau.
 
    Returns a dict rather than a bare float, since order and completeness are separate failure modes and collapsing them into one number hides which one actually happened.
    """

    # List comprehension (compact way of writing a loop to build a new list) to keep only matches whose cost is less than 0.6. Filters out matches that were likely forced by bad pairings by looking at alignment cost (i.e., how diff GT and predicted text are + a penalty fo rmerging multiple blocks)
    reliable = [m for m in matches if m["alignment_cost"] <= cost_threshold]

    # If the no ft blocks is not zero, calculate the ratio: what fraction of the GT got a reliable match. Guards against divide by 0 if there are 0 GT blocks
    completeness = len(reliable) / n_gt_blocks if n_gt_blocks else None

    # An early exit. If kendall's tau (rank correlation measure) is less than 2 reliable matches, then return with layout_score is none but still report completeness since that doesn't need ordering info
    if len(reliable) < 2:
        # Not enough reliable points to say anything about order
        return{
            "layout_order_score": None,
            "completeness": completeness,
            "n_reliable_matches": len(reliable),
            "n_gt_blocks": n_gt_blocks
        }
    
    # List of the GT block indices for reliable match. These are in order since matches was built by iterating GT blocks in order
    gt_order = [m["gt"] for m in reliable]

    # Converts raw pred_start positions into ranks rather thanindex values
    pred_order = [m["pred_start"] for m in reliable]

    # Convert prediction positions to ranks rather than index value
    pred_ranks = rankdata(pred_order)
    # Computes kendall's tau - correlation coefficient (-1-1), measuring how well the order of pre ranks matches the order of GT. "_" discards p-value 
    tau, _ = kendalltau(gt_order, pred_ranks)

    return {
        "layout_order_score": float(tau) if tau == tau else None, # nan guard
        "completeness": completeness,
        "n_reliable_matches": len(reliable),
        "n_gt_blocks": n_gt_blocks
    }

   



# Segmenter Functions
# For regular text with blocks - segmenting by text blocks/chunks
def split_blocks(text):                  # Takes in the text
    blocks = re.split(r"\n\s*\n+", text) # Splits on \n\n, including ones with trailing spaces/tabs (ex: "\n \n")
    if len(blocks) == 1:                 # If the length of the blocks is equal to 1 (i.e., no line breaks found at all):
        blocks = re.split("\n", text) # Then split on every single newline instead, treating each as their own block
    return [b.strip() for b in blocks if b.strip()] # list comprehension - loop over each block & only keep if it is non-empty after stripping white space. Then the value keeped is the stripped version without any trailing or leading whitespace



# Table Segmenter - segmenting by rows
def split_table_rows(text):
    rows = re.split("\n", text)  # Splits rows on \n
    return [b.strip() for b in rows if b.strip()] # Same list logic as previous



# Function for flattening table into string so it can be used in computing model metrics
def flatten_table(table, fallback_text="") -> str: # Take in the table & fallback_text (default = empty sting) as parameters. "-> str" is return type annotation saying this function returns a string

    # If the table parameter is empty or none and not a dictionary, return the text as is
    if not table or not isinstance(table, dict): 
        return fallback_text
    rows = table.get("rows", []) # Rows = the rows column in table. Return empty if rows not present
    flattened_rows = []  # Empty list for future values
    for row in rows:     # For a row in the gotten rows:
        clean_cells = [] # Another empty list
        for cell in row: # For a cell in the row
            cell = "" if not isinstance(cell, str) else cell # If the cell isn't a string (e.g. a stray float/NaN/None), treat it as empty instead of crashing
            clean_cells.append(cell.replace("\n", " ")) # Add to the empty list cells but repalce instances of \n with empty space
        flattened_rows.append(" | ".join(clean_cells)) # Join the cells into the flattened rows, using | to join the cells to each other (| is a separator between each item of the list)
    return "\n".join(flattened_rows) # Return the flattened rows, using \n to distinguish rows from each other (\n is a separator between each item of the list)





# Metrics calculated in evaluate function
# Character error rate (CER)
def character_error(gt, pred):
    return cer(gt, pred)

# Word error rate (WER)
def word_error(gt, pred):
    return wer(gt, pred)

# Token sort ratio
def similarity(gt, pred):
    return fuzz.token_sort_ratio(gt, pred)

# Normalized Levenshtein distance - normalised based on the size of the string so that values between strings are comparable
def normalized_levenshtein(gt, pred):
    """
    Returns a value between 0 and 1.
    0 = identical
    1 = completely different
    """
    return RapidLevenshtein.normalized_distance(gt, pred)

# Levenshtein distance - the exact amount of single-character edits required to transform one string into another
def edit_distance(gt, pred):
    return Levenshtein.distance(gt, pred)



## Evaluate function which combines all of the above
# Purpose: Calculating per page metrics
def evaluate(gt_text, pred_text, page_type, pred_table_text=None, layout_cost_threshold=0.5): # Layout cost threshold 
    # ----------------------------
    # 1. Normalize GT & predicted text 
    # ----------------------------
    gt_norm = normalize(gt_text)    
    pred_norm = normalize(pred_text)

    if page_type == "mixed" and pred_table_text:
        pred_norm = (pred_norm + "\n\n" + normalize(pred_table_text)).strip()

    # ----------------------------
    # 2. Split into blocks
    # ----------------------------
    # If the page type is text or mixed, use the split_blocks for text to split the GT & pred into blocks. Opposite if table
    if page_type == "text":
        gt_blocks = split_blocks(gt_norm)
        pred_blocks = split_blocks(pred_norm)
    elif page_type == "mixed":
        gt_blocks = split_blocks(gt_norm)
        text_blocks = split_blocks(normalize(pred_text)) if pred_text else []
        table_blocks = split_table_rows(normalize(pred_table_text)) if pred_table_text else []
        pred_blocks = text_blocks + table_blocks
    elif page_type == "table":
        gt_blocks = split_table_rows(gt_norm)
        pred_blocks = split_table_rows(pred_norm)

    # Calculate the total difference between the # of blocks in the GT vs the Predicted blocks
    block_count_diff = abs(len(gt_blocks) - len(pred_blocks))
    # Calculate the ratio between the # of blocks in both the GT & predicted if the length of the GT blocks is more than 0
    block_count_ratio = len(pred_blocks)/len(gt_blocks) if len(gt_blocks) > 0 else None

    # ----------------------------
    # 3. Align blocks
    # ----------------------------
    # Align the blocks (remember aligning based on the best possible matches of GT to predicted) & store in 'matches'
    matches = align_blocks(
        gt_blocks,
        pred_blocks
    )

    # ----------------------------
    # 4. Per block diagnostics (kept for inspectign matching quality but no longer aggregated into a weighted score)
    # ----------------------------
    block_scores = []

    # For a match in matches:
    for match in matches:
        gt_i = match["gt"] # Take the gt index # & assign it to gt_i
        # merge matched prediction span - use \n to separate the different blocks
        pred_text_aligned = "\n".join(
            pred_blocks[
                match["pred_start"]:  
                match["pred_end"] + 1 # Match the start & end of the prediction blocks...
            ]
        )
        gt_text_aligned = gt_blocks[gt_i] # Match the GT index pulled to the GT text 

        # Calculate per predicted block matches metrics
        block_scores.append({
            "gt": gt_i,
            "pred_start": match["pred_start"], # Extract the prediction start index & assign
            "pred_end": match["pred_end"], # Extract the prediction end index & assign
            "alignment_cost": match["alignment_cost"],
            "CER": character_error(gt_text_aligned, pred_text_aligned),
            "WER": word_error(gt_text_aligned, pred_text_aligned),
            "Lev": edit_distance(gt_text_aligned, pred_text_aligned),
        })

    # ----------------------------
    # 5. Raw page-level metrics - aware of structure
    # ----------------------------
    overall_cer = character_error(gt_norm,pred_norm)
    overall_wer = word_error(gt_norm,pred_norm)
    overall_lev = edit_distance(gt_norm,pred_norm)
    overall_token_sort = similarity(gt_norm, pred_norm)

    # ----------------------------
    # 6. Layout metric (order + completeness, gated on alignment cost)
    # ----------------------------
    # Calculates layour metrics from the matches. Also takes in the set parameter layout_cost_threshold as the cost_threshold & counts the total number of GT blocks for reference
    layout = layout_score(
        matches,
        n_gt_blocks=len(gt_blocks),
        cost_threshold=layout_cost_threshold)

    # ----------------------------
    # 7. Return results
    # ----------------------------
    return {
        # naive page comparison
        "overall_CER": overall_cer,
        "overall_WER": overall_wer,
        "overall_lev": overall_lev,
        "overall_token_sort": overall_token_sort,
        # structure preservation
        "layout_order_score": layout["layout_order_score"],
        "layout_completeness": layout["completeness"],
        "layout_n_reliable_matches": layout["n_reliable_matches"],
        # Segmentation diagnostics
        "gt_block_count": len(gt_blocks),
        "pred_block_count": len(pred_blocks),
        "block_count_diff": block_count_diff,
        "block_count_ratio": block_count_ratio,
        # detailed per-block diagnostics
        "block_scores": block_scores,
        "matches": matches,
    }



## Builder loop that combines flatten, evaulate & then recombines everything into a final json
def build_per_record_metrics(input_path, output_path, batch_size=10):
    if os.path.exists(output_path):
        print(f"Warning: {output_path} already exists. Delete it first to avoid duplicates.")
        return
    results_buffer = []
    batch_counter = 0
    # NB: encoding is the rule for convertin bytes into text characters & back. utf-8 is the most standard encoding, representing virtually all characters. Without it specific Python uses your OS's default encoding which can vary between machines & produce errors
    with open(input_path, "r", encoding="utf-8") as f:
        # This is all for one record, & then it will be repeated
        for line in f:
            record = json.loads(line.strip())
            # Pull what we need from that JSONL file
            page_type = record["page_type"]
            image_path = record["image_path"]
            document_id = record["document_id"]
            page_id = record["page_id"]
            gt_transcription = record["gt_transcription"]
            model_transcription = record["model_transcription"]
            table = record["table"]
            pred_table_text = None  # only gets set for "mixed" pages; stays None otherwise

            #If the page type is a table: flatten the table. If it's a table with no table key, return the model_transcription or nothing
            if page_type == "table":
                pred_text = flatten_table(table, fallback_text=record.get("model_transcription", ""))
            
            # If the page type if mixed: Flatten the table portion (with same return rule). If the model transcription in mixed is not empty or none, then add it to parts. If the table is not empty then also add it to parts. Separated the predicted parts by \n\n
            elif page_type == "mixed":
                pred_text = model_transcription if model_transcription else ""
                pred_table_text = flatten_table(table, fallback_text="")
            else:  # text
                pred_text = model_transcription
            usage = record.get("usage")
            output_record = evaluate(gt_transcription, pred_text, page_type, pred_table_text=pred_table_text)
            result_record = {
                "image_path": image_path,
                "document_id": document_id,
                "page_id": page_id,
                "page_type": page_type,
                "gt_transcription": gt_transcription,
                "model_transcription": model_transcription,
                "table": table,
                "usage": usage,
                **output_record # Take every key pair value inside output_record & spread them as individual entries into this record
            }
            results_buffer.append(result_record) # Add the results record (for each image) to the results buffer
            batch_counter += 1 # add a 1 to the batch count

            # If the batch counter is greater than or equal to the size, then save the results & restart the results buffer & batch counter
            if batch_counter >= batch_size:
                with open(output_path, "a", encoding="utf-8") as out_f:
                    for r in results_buffer:
                        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                results_buffer = []
                batch_counter = 0
    # If the length of the results buffer is greater than 0, save results & print that the final batch is saved
    if len(results_buffer) > 0:
        with open(output_path, "a", encoding="utf-8") as out_f:
            for r in results_buffer:
                # Ensure ascii keeps real characters as input instead of converting them. \n adds a new line after each JSON record
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("✔ Final batch saved")


# 95% CI
def bootstrap_ci(scores, n_bootstrap=2000, ci_level=95):
    if len(scores) < 2:
        return (float("nan"), float("nan"))
    
    scores = np.array(scores)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    lower_p = (100 - ci_level) / 2
    upper_p = 100 - lower_p
    ci_lower = np.percentile(bootstrap_means, lower_p)
    ci_upper = np.percentile(bootstrap_means, upper_p)
    return (ci_lower, ci_upper)
    

# Corpus-level metrics (averaged/median on the whole corpus)
# Purpose: 
def corpus_metrics(result_path, page_types_to_include=None):
    with open(result_path, "r", encoding="utf-8") as f:
        page_types = []
        metric_lists = {
                "overall_CER": [],
                "overall_WER": [],
                "overall_lev": [],
                "overall_token_sort": [],
                "layout_order_score": [],
                "layout_completeness": [],
                "block_count_ratio": []
            }
        token_list = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": [],
            "eval_duration_ns": []
        }
        # For one record in the json
        for line in f:
            record = json.loads(line.strip()) # Strip any trailing or leading white space
            # If page types to include is specified & the record page type does not equal it, skip it
            if page_types_to_include is not None and record["page_type"] not in page_types_to_include:
                continue
            # Add the page type to list
            page_types.append(record["page_type"])
            # For a key in the metrics list - if it is not none, add it to the list
            for key in metric_lists:
                if record[key] is not None:
                    metric_lists[key].append(record[key])
            # Extract usage from the record
            usage = record.get("usage", {})
            # For a key in the token list, extract the key value & if that value is not nine, then add it to the token list
            for key in token_list:
                val = usage.get(key)
                if val is not None:
                    token_list[key].append(val)
        # Count the number of page types
        distribution = Counter(page_types)
    summary = {}
    # For a key in the now completed record metrics list:
    for key in metric_lists:
        # Extract the keys and save to scores
        scores = metric_lists[key]
        # If scores is empty, print no data & skipping
        if not scores:
            print(f"No data for {key}, skipping.")
            continue
        ci_lower, ci_upper = bootstrap_ci(scores)
        # Summarise for all keys their mean, median & length & save
        summary[key] = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else float("nan"),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": len(scores)
        }
    # For a key (record) print these summary metrics, rounded to 3 decimal points
    for key in summary:
        s = summary[key]
        print(f"\n {key} (n={s['n']}): {s['mean']:.3f} (95% CI: {s['ci_lower']:.3f}–{s['ci_upper']:.3f}, median: {s['median']:.3f})")

    token_summary = {}
    # For the key & corresponding scores in the token list (opened as a view object with key-value pairs)
    for key, scores in token_list.items():
        # If there's no scores print no data & skipping
        if not scores:
            print(f"No data for {key}, skipping.")
            continue
        # For remaining keys, calculate the total, the mean & how many of each record was in it
        token_summary[key] = {
            "total": sum(scores),
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else float("nan"),
            "n": len(scores)
        }
    # Print the info to 1 decimal place
    for key, s in token_summary.items():
        print(f"\n {key} (n={s['n']}): total={s['total']}, mean={s['mean']:.1f} ± {s['std']:.3f}")

    print(distribution)


    

# JSONL to Excel Function
# Purpose: 
## B: need to pick one format over another


################
################

# Pipeline for Ollama Models

# Builder loop 
# Purpose:
def build_messages_ollama(user_prompt, examples, image_path, sys_prompt=system_prompt, schema=table_prompt_format_transcriptions):
    messages = [
        {
            'role': 'system',
            'content': sys_prompt
        }
    ]
    for example in examples:
        # append a user message with the image
        ex_image_path = example.get("image_path", "")
        transcription = example.get("transcription", "")
        # append an assistant message with the transcription
        messages.append({
                'role': 'user',
                'content': user_prompt, 
                'images': [ex_image_path]
            })
        messages.append({
                'role':'assistant',
                'content': transcription
            })
    messages.append({
        'role': 'user',
        'content': f"""
            {user_prompt}
            Provide the information following this JSON schema exactly and output JSON only: {schema}
            """,
        'images': [image_path]
        })
    return messages
    
################

# Main loop
# Purpose:
def run_ollama_pipeline(dataset, user_prompt, output_path, examples, temperature, sys_prompt=system_prompt, schema=table_prompt_format_transcriptions, model='qwen2.5vl:3b', batch_size=10, image_field="image_path"):
    start_time = time.time() # Start time of programme
    # Check if output file already exists to prevent duplicates
    if os.path.exists(output_path):
        print(f"Warning: {output_path} already exists. Delete it first to avoid duplicates.")
        return
    results_buffer = []   # in-memory storage until we flush
    batch_counter = 0     # For counting how many batches we've gone through
    for record in dataset:
        # First, the things being extracted from each record inside the loop
        image_path = record[image_field]                # Image path
        gt_transcription = record["transcription"]      # Main transcription
        document_id = record["document_id"]
        page_id = record["page_id"]                      # Page ID
        # Now we want to load in the image
        if image_path is None:
            print(f"No image path found for {page_id}, skipping")
            continue # skip to the next record
        # Then load in fll message structure using function parameters
        response = client.chat(
            model=model,
            messages=build_messages_ollama(
                user_prompt=user_prompt,
                examples=examples,
                image_path=image_path,
                sys_prompt=sys_prompt,
                schema=schema,
            ),
            options={"num_predict": 5000, "temperature": temperature}
        )


        raw_output = response['message']['content']
        # Creating a dictionary to log tokens
        usage = {
            "prompt_tokens": response.get("prompt_eval_count", None), # Returns the number of tokens process for input prompt
            "completion_tokens": response.get("eval_count", None), # Extracts number of tokens generated by the LLM
            "total_tokens": (response.get("prompt_eval_count", 0) or 0) + (response.get("eval_count", 0) or 0), # Calculates grand total of tokens used in LLM request including input prompt & outputs reponse
            "eval_duration_ns": response.get("eval_duration", None), # Extracts the time it takes the LLM to generate 
            "temperature": temperature
        }
        # Then pass string into function
        result = string_to_json(raw_output)
        # Handle case where result is a list instead of a dictionary
        if isinstance(result, list):
            result = result[0] if len(result) > 0 else None
        # Safety check
        if result is None:
            print(f"Failed to parse output for {page_id}, skipping")
            continue
        # print(result.keys())
        # Next: extract the model's main transcription & marginalia transcription from the result

        model_transcription = result.get("text_transcription", "")
        page_type = result.get("page_type", "text")
        table = result.get("table", None)

        
        result_record = {
            "image_path": image_path,
            "document_id": document_id,
            "page_id": page_id,
            "page_type": page_type,
            "gt_transcription": gt_transcription,
            "model_transcription": model_transcription,
            "table": table,
            "usage": usage
                }
        results_buffer.append(result_record)

        # Add batch counter
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
    elapsed = time.time() - start_time
    print(f"Finished in {elapsed/60:.2f} minutes")




################
################

# Pipeline for OpenRouter Models

# Builder loop
# Purpose: 
def build_messages_openrouter(user_prompt, image_path, examples, sys_prompt=system_prompt, schema=table_prompt_format_transcriptions):
    messages = [
        {
            'role': 'system',
            'content': sys_prompt
        }
    ]
    for example in examples:
        # append a user message with the image
        ex_image_path = example.get("image_path", "")
        transcription = example.get("transcription", "")
        with open(ex_image_path, "rb") as f:              # then encode the path
            ex_image_data = base64.b64encode(f.read()).decode("utf-8")
        # append an assistant message with the transcription
        messages.append({
            'role': 'user',
            'content': [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{ex_image_data}"}
                },
                {
                    'type': 'text',
                    'text': user_prompt
                }
            ]
        })
        messages.append({
            'role':'assistant',
            'content': transcription
        })
    with open(BASE_DIR / image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    messages.append({
        "role": "user",
        "content": [
            {
                'type': 'image_url',
                'image_url': {"url": f"data:image/jpeg;base64,{image_data}"}
            },
            {
                'type': 'text',
                'text': f"""
                    {user_prompt}
                    Provide the information following this JSON schema exactly and output JSON only: {schema}
                    """
            }
        ]
    })
    return messages

################

# Open router call function
@retry(
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True
)
def call_openrouter(image_path, user_prompt, model, examples, temperature, sys_prompt=system_prompt, schema=table_prompt_format_transcriptions):
    messages = build_messages_openrouter(user_prompt, image_path, examples, sys_prompt=sys_prompt, schema=schema)
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

################

# Main loop adapted for new schema to handle tables
def run_openrouter_pipeline(dataset, user_prompt, output_path, model, examples, temperature, sys_prompt=system_prompt, schema=table_prompt_format_transcriptions, batch_size=10, image_field="image_path"):
    start_time = time.time() # Start time of programme
    # Check if output file already exists to prevent duplicates
    if os.path.exists(output_path):
        print(f"Warning: {output_path} already exists. Delete it first to avoid duplicates.")
        return
    results_buffer = []   # in-memory storage until we flush
    batch_counter = 0     # For counting how many batches we've gone through
    for record in dataset:
        # First, the things being extracted from each record inside the loop
        image_path = record[image_field]                # Image path
        gt_transcription = record["transcription"]  # Main transcription
        document_id = record["document_id"]
        page_id = record["page_id"]                      # Page ID
        # Now we want to load in the image
        if image_path is None:
            print(f"No image path found for {page_id}, skipping")
            continue # skip to the next record
        # Call openrouter api with current image and prompt
        raw_output, usage = call_openrouter(
            image_path=image_path,
            user_prompt=user_prompt,
            sys_prompt=sys_prompt,
            schema=schema,
            model=model,
            examples=examples,
            temperature=temperature)

        # Then pass string into function
        result = string_to_json(raw_output)
        # Handle case where result is a list instead of a dictionary
        if isinstance(result, list):
            result = result[0] if len(result) > 0 else None
        # Safety check
        if result is None:
            print(f"Failed to parse output for {page_id}, skipping")
            continue  
        # print(result.keys())
        # Next: extract the model's main transcription & marginalia transcription from the result

        
        model_transcription = result.get("text_transcription", "")
        page_type = result.get("page_type", "text")
        table = result.get("table", None)



        result_record = {
            "image_path": image_path,
            "document_id": document_id,
            "page_id": page_id,
            "page_type": page_type,
            "gt_transcription": gt_transcription,
            "model_transcription": model_transcription,
            "table": table,
            "usage": usage
                }
        results_buffer.append(result_record)

        # Add batch counter
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
    elapsed = time.time() - start_time
    print(f"Finished in {elapsed/60:.2f} minutes")





