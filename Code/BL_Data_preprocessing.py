import os
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET


# Setting the data path
DATA_DIR = Path("data/ior_htr_groundtruth")

# Get all document folders (excluding the datacard)
doc_folders = sorted([            # Sorting alphabetically by name
    f for f in DATA_DIR.iterdir() # Loops through every item in the directory
    if f.is_dir()                 # Building a list of all document folders (is_dir checks whether a path points to a directory)
])

print(f"Found {len(doc_folders)} documents:") # Printing number of folders
for folder in doc_folders:      # Lops through each folder & extracts folder name
    print(f"  {folder.name}")


# Looking at the structure of one document
sample = doc_folders[0]

images = sorted(sample.glob("*.jpg"))
alto_xmls = sorted((sample / "alto").glob("*.xml"))

print(f"Document: {sample.name}")
print(f"  Images: {len(images)}")
print(f"  ALTO XMLs: {len(alto_xmls)}")
print(f"\nFirst image: {images[0].name}")
print(f"First XML:   {alto_xmls[0].name}")



# Looking at the HPOS values from across a few pages to find a threshold to help distinguish the main text from the marginal text. 

# HPOS = the horizontal position of a textblock on the page, measured in pixels from the left edge/
import os
import xml.etree.ElementTree as ET

ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}

# Look at first 3 documents, first 2 pages each
docs_to_sample = os.listdir(DATA_DIR)
docs_to_sample = [d for d in docs_to_sample if os.path.isdir(os.path.join(DATA_DIR, d))][:3]

for doc in docs_to_sample:
    alto_path = os.path.join(DATA_DIR, doc, "alto")
    xml_files = sorted([f for f in os.listdir(alto_path) if f.endswith(".xml") and f != "mets.xml"])[:2]
    
    print(f"\n{'='*60}")
    print(f"DOCUMENT: {doc}")
    print(f"{'='*60}")
    
    for xml_file in xml_files:
        print(f"\n  PAGE: {xml_file}")
        print(f"  {'-'*40}")
        
        tree = ET.parse(os.path.join(alto_path, xml_file))
        root = tree.getroot()
        
        # Get page width for context
        page = root.find(".//alto:Page", ns)
        page_width = int(page.attrib["WIDTH"])
        print(f"  Page width: {page_width}px")
        
        for block in root.findall(".//alto:TextBlock", ns):
            block_id = block.attrib["ID"]
            hpos = int(block.attrib["HPOS"])
            width = int(block.attrib["WIDTH"])
            vpos = int(block.attrib["VPOS"])
            right_edge = hpos + width
            
            # Get first line of text for context
            first_string = block.find(".//alto:String", ns)
            preview = first_string.attrib["CONTENT"][:40] if first_string is not None else "[empty]"
            
            print(f"  Block {block_id}: HPOS={hpos}, WIDTH={width}, right_edge={right_edge}, VPOS={vpos}")
            print(f"    Preview: '{preview}'")



xml_path = "data/ior_htr_groundtruth/ior!p!351!28_29_Mar_1854_nos_1894-1902/alto/0005_ior!p!351!28_29_Mar_1854_nos_1894-1902_f003r.xml"

tree = ET.parse(xml_path)
root = tree.getroot()

# First check what namespace this file actually uses
print(root.tag)
print(root.attrib)

# Count all TextBlock elements with no namespace
all_blocks = root.findall(".//{*}TextBlock")
print(f"\nTotal blocks found: {len(all_blocks)}")

# Print each block
for block in all_blocks:
    hpos = int(block.attrib.get("HPOS", 0))
    width = int(block.attrib.get("WIDTH", 0))
    vpos = int(block.attrib.get("VPOS", 0))
    print(f"Block HPOS={hpos}, WIDTH={width}, VPOS={vpos}")


for block in all_blocks:
    hpos = int(block.attrib.get("HPOS", 0))
    width = int(block.attrib.get("WIDTH", 0))
    vpos = int(block.attrib.get("VPOS", 0))
    
    lines = []
    for line in block.findall(".//{*}TextLine"):
        string_el = line.find("{*}String")
        if string_el is not None:
            lines.append(string_el.attrib.get("CONTENT", ""))
    
    content_preview = " ".join(lines)[:60]
    print(f"HPOS={hpos}, WIDTH={width}, VPOS={vpos}: '{content_preview}'")

 
# ALTO XML namespace
NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
 

# Separators
LINE_SEP = " "
BLOCK_SEP = "\n\n"               # Between main text blocks


# Function for extracting lines from the xml blocks
def extract_lines_from_block(block):
  lines = [] # Putting this inside function so a fresh empty list is created everytime the function is called
  for line in block.findall("alto:TextLine", NS):
    # find() navigates down to the String element inside this TextLine
    # "alto:String" uses NS dictionary to expand to the full namespace
    # so Python knows which String element to look for in the ALTO format
    string_el = line.find("alto:String", NS)
    if string_el is not None:
      content = string_el.attrib.get("CONTENT", "")
      lines.append(content)
  return lines

def extract_page_transcription_naturalsorting(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    blocks = []
    for block in root.findall(".//alto:TextBlock", NS):
        lines = extract_lines_from_block(block)
        blocks.append(lines)
    # No sorting — just join in XML order
    page_transcription_strings = [" ".join(lines) for lines in blocks]
    full_transcription = "\n\n".join(page_transcription_strings)
    return full_transcription

# Testing on one image
test_xml = "data/ior_htr_groundtruth/ior!p!351!28_29_Mar_1854_nos_1894-1902/alto/0005_ior!p!351!28_29_Mar_1854_nos_1894-1902_f003r.xml"
result_unsorted = extract_page_transcription_naturalsorting(test_xml)
print(result_unsorted)


# Now creating the full loop that will apply all of the above functions to the whole dataset file

DATA_DIR = Path("data/ior_htr_groundtruth")

OUTPUT_PATH = Path("data/ior_fullpage_transcriptions_naturalsorting")

def extract_dataset(DATA_DIR):
    records = []
    for item in os.listdir(DATA_DIR):              # loop through every item in the base folder one by one
        full_path = os.path.join(DATA_DIR, item)   # Sticks the base path & item name together
        if os.path.isdir(full_path):                # Now with the full address, we can check whether the address is a folder or file - if it's a folder we want to proceed
            alto_path = os.path.join(full_path, "alto")
            alto_files = os.listdir(alto_path)
            for xml_file in alto_files:
                if xml_file == "mets.xml":
                    continue
                xml_path = os.path.join(alto_path, xml_file) # Needs folder and filename
                transcription = extract_page_transcription_naturalsorting(xml_path)
                # Now we need to make a dictionary & append it to records
                record = {
                    "image_path":os.path.join(full_path, os.path.splitext(xml_file)[0] + ".jpg"),
                    "document_id": item,
                    "page_id": os.path.splitext(xml_file)[0],
                    "transcription": transcription
                }
                records.append(record)
# Open the output file for writing
    # OUTPUT_PATH: where to save the file (defined in config at top)
    # "w": write mode — creates the file if it doesn't exist, overwrites if it does
    # encoding="utf-8": ensures special characters are handled correctly
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:     
        # json.dump writes our records list into the file as formatted JSON
        # ensure_ascii=False: preserves special characters like & or accented letters as-is
        # indent=2: adds 2 space indentation so the JSON is human readable
        json.dump(records, f, ensure_ascii=False, indent = 2)
    return records


with open("data/ior_fullpage_transcriptions_naturalsorting", "r", encoding="utf-8") as f:
    ior_fullpage_transcriptions_naturalsorting = json.load(f)

print(f"Total records: {len(ior_fullpage_transcriptions_naturalsorting)}")

VALIDATION_PATH = "data/fullpage_validation_data.json"
TEST_PATH = "data/fullpage_test_data.json"

random.seed(42) # set seed for reproducability

# Sample 10% randomly for validation/development set
validation_set = random.sample(ior_fullpage_transcriptions_naturalsorting, int(len(ior_fullpage_transcriptions_naturalsorting) * 0.10))

# In order to check against a list of page_id values from the validation set (which is currently a dictionary)
validation_ids = [r["page_id"] for r in validation_set] # 'r' is a single record from validation_set with a dictionary that includes doc id, image path, transcription, etc
# To get a specific value out of a dictionary, you use square brackets with key name

# Remaining records become test set
test_set = [r for r in ior_fullpage_transcriptions_naturalsorting if r["page_id"] not in validation_ids]


with open(VALIDATION_PATH, "w", encoding="utf-8") as f:
    json.dump(validation_set, f, ensure_ascii=False, indent=2)

with open(TEST_PATH, "w", encoding="utf-8") as f:
    json.dump(test_set, f, ensure_ascii=False, indent=2)

# Checking this worked
print(f"Found {len(validation_set)} documents:") 
print(f"Found {len(test_set)} documents:")

# Checking if the two dataset lists share any values 
test_ids = [r["page_id"] for r in test_set] 
print(f"{set(validation_ids) & set(test_ids)}")

