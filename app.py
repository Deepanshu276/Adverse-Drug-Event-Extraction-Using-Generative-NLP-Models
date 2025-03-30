import os
import json
import logging
import requests # type: ignore
from typing import List, Dict, Any
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util # type: ignore
from langchain_community.llms import HuggingFacePipeline # type: ignore
from langchain.chains import LLMChain # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
from langchain_core.exceptions import OutputParserException # type: ignore


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

UMLS_API_KEY = " "
UMLS_AUTH_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
UMLS_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"

CADEC_ORIGINAL_FOLDER = "./CADEC.v1/Original/"
CADEC_AMT_SCT_FOLDER = "./CADEC.v1/AMT-SCT/"
CADEC_MEDDRA_FOLDER = "./CADEC.v1/MedDRA/"

MAX_RETRIES = 3


# Initialize Models
logger.info("Loading generative model google/flan-t5-base") # used facebook's bart large but accuracy was bad
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base") 
logger.info("Generative model loaded.")

logger.info("Loading Sentence Transformer model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2') # all-mpnet-base-v2 may work but heavier and slower and require large resources
logger.info("Sentence Transformer model loaded.")

llm = HuggingFacePipeline(pipeline=hf_pipeline)

#Prompt Template
prompt_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "Extract the following entities from the text. Return ONLY a valid JSON object without any additional text. "
        "The JSON object must have exactly these keys: drugs, ADEs, and symptoms, each mapping to a list of strings.\n"
        "For example, if the text is 'Aspirin causes nausea and headache', the output should be:\n"
        "{{\"drugs\": [\"Aspirin\"], \"ADEs\": [\"nausea\"], \"symptoms\": [\"headache\"]}}\n"
        "Now, extract the entities from the following text:\n"
        "Text: {text}"
    )
)

extraction_chain = LLMChain(llm=llm, prompt=prompt_template) #extract the entity

def query_umls(entity: str, vocab: str) -> dict:
    """
    Direct UMLS API call for standardization
    Works for RxNorm (drugs) and SNOMED CT (ADEs/symptoms)
    """
    url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    params = {
        "string": entity,
        "apiKey": UMLS_API_KEY, 
        "sabs": vocab,
        "pageSize": 1
    }
    
    try:
        response = requests.get(url, params=params)
        results = response.json().get("result", {}).get("results", [])
        print("original", entity)
        print(results[0]["name"])
        print("cui",results[0]["ui"])
        return {
            "original": entity,
            "standard_term": results[0]["name"] if results else None,
            "CUI": results[0]["ui"] if results else None
        }
    except Exception as e:
        return {"original": entity, "standard_term": None, "CUI": None}
    

# Parsing Functions for Annotation Files
def parse_original_file(text: str) -> List[Dict[str, Any]]:
    annotations = []
    print("Parsing Original file...")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        ann_id = parts[0]
        info_tokens = parts[1].split()
        entity_type = info_tokens[0]
        offsets = info_tokens[1:]
        entity_text = parts[2].strip()
        annotations.append({
            "id": ann_id,
            "entity_type": entity_type,
            "offsets": offsets,
            "text": entity_text
        })
        print(f"Parsed Original annotation: {ann_id}, {entity_type}, {offsets}, {entity_text}")
    return annotations

def parse_amt_sct_file(text: str) -> List[Dict[str, Any]]:
    annotations = []
    print("Parsing AMT-SCT file...")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        ann_id = parts[0]
        second_col = parts[1].strip()
        ann_text = parts[2].strip()
        if " or " in second_col:
            segments = second_col.split(" or ")
            last_seg_tokens = segments[-1].split("|")
            offsets = last_seg_tokens[-1].strip().split() if len(last_seg_tokens) >= 3 else []
            for seg in segments:
                seg = seg.strip(" |")
                tokens = seg.split("|")
                if len(tokens) >= 2:
                    concept_id = tokens[0].strip()
                    concept_name = tokens[1].strip()
                else:
                    concept_id = tokens[0].strip()
                    concept_name = ""
                annotations.append({
                    "id": ann_id,
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "offsets": offsets,
                    "text": ann_text
                })
                print(f"Parsed AMT-SCT annotation: {ann_id}, {concept_id}, {concept_name}, {offsets}, {ann_text}")
        else:
            tokens = second_col.split("|")
            if len(tokens) >= 3:
                concept_id = tokens[0].strip()
                concept_name = tokens[1].strip()
                offsets = tokens[2].strip().split()
            else:
                concept_id = tokens[0].strip()
                concept_name = ""
                offsets = []
            annotations.append({
                "id": ann_id,
                "concept_id": concept_id,
                "concept_name": concept_name,
                "offsets": offsets,
                "text": ann_text
            })
            print(f"Parsed AMT-SCT annotation: {ann_id}, {concept_id}, {concept_name}, {offsets}, {ann_text}")
    return annotations

def parse_meddra_file(text: str) -> List[Dict[str, Any]]:
    annotations = []
    print("Parsing MedDRA file...")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        ann_id = parts[0]
        second_col = parts[1].strip()
        tokens = second_col.split()
        if len(tokens) >= 3:
            meddra_code = tokens[0]
            offsets = tokens[1:]
        else:
            meddra_code = tokens[0]
            offsets = []
        ann_text = parts[2].strip()
        annotations.append({
            "id": ann_id,
            "meddra_code": meddra_code,
            "offsets": offsets,
            "text": ann_text
        })
        print(f"Parsed MedDRA annotation: {ann_id}, {meddra_code}, {offsets}, {ann_text}")
    return annotations

def load_annotations_from_folder(folder_path: str, parser_func) -> Dict[str, List[Dict[str, Any]]]:
    annotations = {}
    print(f"Loading annotations from folder: {folder_path}")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith('.ann'):
                continue
            full_path = os.path.join(root, file)
            print(f"Reading file: {full_path}")
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            print(f"Content from file (first 100 chars): {content[:100]}")
            parsed = parser_func(content)
            key = os.path.splitext(file)[0]
            annotations[key] = parsed
            print(f"Parsed {len(parsed)} annotations from file: {file}")
    logger.info(f"Loaded {len(annotations)} files from {folder_path}")
    return annotations

def merge_annotations(original_dict: Dict[str, Any],
                      amt_sct_dict: Dict[str, Any],
                      meddra_dict: Dict[str, Any]) -> Dict[str, Any]:
    print("Merging annotations...")
    merged = {}
    keys = set(original_dict.keys()) | set(amt_sct_dict.keys()) | set(meddra_dict.keys())
    for key in keys:
        merged[key] = {
            "original": original_dict.get(key, []),
            "amt_sct": amt_sct_dict.get(key, []),
            "meddra": meddra_dict.get(key, [])
        }
        print(f"Merged annotations for key: {key}")
    return merged

def get_forum_post_text(original_annotations: List[Dict[str, Any]]) -> str:
    texts = [ann["text"] for ann in original_annotations]
    combined_text = " ".join(texts)
    print(f"Combined forum post text (first 200 chars): {combined_text[:200]} ...")
    return combined_text

def get_ground_truth(original_annotations: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    gt = {"drugs": [], "ADEs": [], "symptoms": []}
    for ann in original_annotations:
        etype = ann.get("entity_type", "").lower()
        text = ann.get("text", "")
        if etype == "drug":
            gt["drugs"].append(text)
        elif etype == "adr":
            gt["ADEs"].append(text)
        elif etype in ["symptom", "disease"]:
            gt["symptoms"].append(text)
    print("Constructed ground truth from original annotations:")
    print("ground_truth",gt)
    return gt


# Generative NLP Pipeline Functions
def expand_abbreviations(text: str) -> str:
    prompt = f"Expand abbreviations in the following text:\n\n{text}"
    print("Prompting generative model for abbreviation expansion...")
    result = hf_pipeline(prompt, max_length=512)[0]['generated_text']
    print(f"Expanded text (first 200 chars): {result[:200]} ...")
    return result

def extract_entities(text: str) -> Dict[str, Any]:
    """
    Uses the LangChain LLMChain to extract entities as structured JSON.
    If the output parser fails, catch the exception and manually clean the output.
    """
    print("Prompting LangChain extraction chain for entity extraction...")
    print("Input text:", text)

    try:
        raw_response = extraction_chain.invoke({"text": text})
        print("Raw response from extraction chain:", raw_response)
    except OutputParserException as e:
        logger.error("OutputParserException encountered. Raw output:")
        print(e.llm_output)
        raw_response = e.llm_output

    if isinstance(raw_response, dict):
        if "text" in raw_response:
            raw_text = raw_response["text"]
        else:
            raw_text = str(raw_response)
    else:
        raw_text = raw_response

    print("Raw generative model response (first 300 chars):")
    print(raw_text[:300])
    
    if "{" not in raw_text:
        if "ADE" in raw_text:
            extracted = {"drugs": [], "ADEs": [raw_text.strip()], "symptoms": []}
        else:
            extracted = {"drugs": [], "ADEs": [], "symptoms": [raw_text.strip()]}
    else:
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = raw_text[start:end+1]
        else:
            json_str = raw_text  

        try:
            extracted = json.loads(json_str)
        except Exception as e:
            logger.error("Entity extraction failed during manual JSON parsing: " + str(e))
            extracted = {"drugs": [], "ADEs": [], "symptoms": []}

    print("Extracted entities:")
    print(extracted)
    return extracted


def standardize_entities(extracted: Dict[str, Any]) -> Dict[str, Any]:
    standardized = {
        "drugs": [query_umls(drug, "RXNORM") for drug in extracted["drugs"]],
        "ADEs": [query_umls(ade, "SNOMEDCT_US") for ade in extracted["ADEs"]],
        "symptoms": [query_umls(sym, "SNOMEDCT_US") for sym in extracted["symptoms"]]
    }
    return standardized

def verify_format(data: Dict[str, Any]) -> bool:
    """
    Verifies that the standardized data contains the required keys and that values are lists.
    """
    expected_keys = ['drugs', 'ADEs', 'symptoms']
    for key in expected_keys:
        if key not in data or not isinstance(data[key], list):
            logger.error(f"Format verification failed: Missing or invalid '{key}' key.")
            return False
    return True

def verify_completeness(extracted: Dict[str, Any], ground_truth: Dict[str, Any]) -> bool:
    """
    Checks that each extracted entity (per type) appears in at least one ground truth annotation.
    """
    complete = True
    for key in ['drugs', 'ADEs', 'symptoms']:
        gt_items = ground_truth.get(key, [])
        extracted_items = extracted.get(key, [])
        for item in extracted_items:
            item_text = item["original"] if isinstance(item, dict) and "original" in item else item
            if not any(item_text.lower() in gt.lower() for gt in gt_items):
                logger.warning(f"Completeness check: Extracted '{item_text}' not found in ground truth for {key}.")
                complete = False
    return complete

def average_similarity(list1, list2):
    if not list1 or not list2:
        return 0.0
    sims = []
    for item1 in list1:
        item1_text = item1["original"] if isinstance(item1, dict) and "original" in item1 else str(item1)
        emb1 = sbert_model.encode(item1_text, convert_to_tensor=True)
        for item2 in list2:
            item2_text = item2["original"] if isinstance(item2, dict) and "original" in item2 else str(item2)
            emb2 = sbert_model.encode(item2_text, convert_to_tensor=True)
            sims.append(util.pytorch_cos_sim(emb1, emb2).item())
    return sum(sims) / len(sims) if sims else 0.0



def verify_semantic_similarity(extracted: Dict[str, Any], ground_truth: Dict[str, Any], threshold: float = 0.7) -> bool:
    
    overall = True
    for key in ['drugs', 'ADEs', 'symptoms']:
        ext_items = extracted.get(key, [])
        gt_items = ground_truth.get(key, [])
        avg_sim = average_similarity(ext_items, gt_items)
        logger.info(f"Semantic similarity for {key}: {avg_sim:.2f}")
        if avg_sim < threshold:
            overall = False
    return overall

def process_post(post: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n=== Processing post: {post['id']} ===")
    print("Original post text:")
    print(post["text"])
    text_expanded = expand_abbreviations(post["text"])
    logger.info("Abbreviation expansion completed.")

    attempt = 0
    final_output = None

    while attempt < MAX_RETRIES:
        logger.info(f"Extraction attempt {attempt+1} for post {post['id']}.")
        extracted_entities = extract_entities(text_expanded)
        standardized = standardize_entities(extracted_entities)
        format_ok = verify_format(standardized)
        completeness_ok = verify_completeness(standardized, ground_truth)
        semantic_ok = verify_semantic_similarity(standardized, ground_truth)
        
        print(f"Verification results for attempt {attempt+1}: Format: {format_ok}, Completeness: {completeness_ok}, Semantic: {semantic_ok}")
        if format_ok and completeness_ok and semantic_ok:
            final_output = standardized
            logger.info("Verification successful. Extraction complete.")
            break
        else:
            logger.warning("Verification failed. Retrying extraction with feedback.")
            attempt += 1

    if final_output is None:
        logger.error(f"Failed to extract valid entities after {MAX_RETRIES} attempts for post {post['id']}.")
        final_output = standardized 

    return final_output

def main():
    print("Starting main pipeline...")
    original_annotations = load_annotations_from_folder(CADEC_ORIGINAL_FOLDER, parse_original_file)
    amt_sct_annotations = load_annotations_from_folder(CADEC_AMT_SCT_FOLDER, parse_amt_sct_file)
    meddra_annotations = load_annotations_from_folder(CADEC_MEDDRA_FOLDER, parse_meddra_file)
    merged_annotations = merge_annotations(original_annotations, amt_sct_annotations, meddra_annotations)
    logger.info("Annotations merged.")

    extraction_results = {}
    for post_id, ann_data in merged_annotations.items():
        if ann_data["original"]:
            forum_text = get_forum_post_text(ann_data["original"])
            gt = get_ground_truth(ann_data["original"])
            post = {"id": post_id, "text": forum_text}
            logger.info(f"Processing post: {post_id}")
            result = process_post(post, gt)
            extraction_results[post_id] = result
        else:
            logger.warning(f"No original annotations for post {post_id}; skipping extraction.")

    with open("merged_cadec_annotations.json", "w", encoding="utf-8") as f:
        json.dump(merged_annotations, f, indent=4)
    with open("extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(extraction_results, f, indent=4)

    logger.info("Processing complete. Merged annotations and extraction results saved.")
    print("Main pipeline completed.")

if __name__ == "__main__":
    main()
