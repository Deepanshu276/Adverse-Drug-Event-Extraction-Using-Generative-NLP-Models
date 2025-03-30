# Entity Extraction & Standardization Pipeline

This project implements an NLP pipeline that parses annotated files, expands abbreviations, extracts entities using a generative model, and standardizes them via the UMLS API. The code leverages Hugging Face transformers, Sentence Transformers, and LangChain to perform end-to-end processing of forum posts extracted from CADEC datasets.

## Overview

The pipeline performs the following key steps:

1. **Data Loading & Parsing:**  
   - Loads annotation files from three folders (`CADEC.v1/Original/`, `CADEC.v1/AMT-SCT/`, and `CADEC.v1/MedDRA/`).
   - Parses these files using dedicated functions to extract annotation details.

2. **Annotation Merging:**  
   - Merges the parsed annotations based on file basename, creating a consolidated view of each forum post.

3. **Pre-processing:**  
   - Combines the text annotations from the original files to simulate a forum post.
   - Expands any abbreviations present in the text using a generative model.

4. **Entity Extraction & Standardization:**  
   - Uses a LangChain LLMChain to extract structured entities (drugs, ADEs, and symptoms) from the expanded text.
   - Standardizes the extracted entities using the UMLS API (with direct API key usage) for mapping to standard vocabularies (RxNorm and SNOMED CT).

5. **Verification & Retry:**  
   - The pipeline performs format, completeness, and semantic similarity verifications on the standardized entities.
   - It retries extraction up to a configurable maximum number of attempts (`MAX_RETRIES`) if verification fails.

6. **Output:**  
   - Merged annotations and final extraction results are saved as JSON files (`merged_cadec_annotations.json` and `extraction_results.json`).

## Prerequisites

- **Python Version:** Python 3.7+
- **Libraries:**  
  - `os`, `json`, `logging`, `re`, `requests`
  - [Transformers](https://github.com/huggingface/transformers)  
  - [Sentence Transformers](https://www.sbert.net/)
  - [LangChain](https://github.com/langchain-ai/langchain)
  - [Pydantic](https://pydantic-docs.helpmanual.io/)

Install required packages using pip. For example, you might use a requirements file:

```bash
pip install transformers sentence-transformers langchain pydantic requests
```

## Setup & Configuration

1. **UMLS API Key:**  
   Replace the placeholder UMLS API key (`UMLS_API_KEY`) in the code with your actual key from the UMLS portal.

2. **Folder Paths:**  
   Ensure that the paths to the CADEC annotation folders (`CADEC.v1/Original/`, `CADEC.v1/AMT-SCT/`, and `CADEC.v1/MedDRA/`) are correct or adjust them as needed.

3. **Models:**  
   - The generative model used is `google/flan-t5-base` via Hugging Face's pipeline.
   - Sentence Transformer model `all-MiniLM-L6-v2` is used for computing semantic similarity.

## Usage

Run the main pipeline by executing the Python script:

```bash
python <script_name>.py
```

The script will:
- Load and merge annotations from the designated folders.
- Process each forum post by expanding abbreviations, extracting and standardizing entities.
- Save the merged annotations and extraction results to JSON files.

## Code Structure

- **Logging Configuration:**  
  Sets up logging to output key process steps at the INFO level.

- **Global Configurations:**  
  Contains API credentials, folder paths, and retry configurations.

- **Model Initialization:**  
  Loads both the generative model (via Hugging Face's pipeline) and the Sentence Transformer model.

- **Annotation Parsing Functions:**  
  Dedicated functions (`parse_original_file`, `parse_amt_sct_file`, `parse_meddra_file`) for reading and parsing annotation files from each folder.

- **Annotation Merging:**  
  The `merge_annotations` function combines parsed data based on file basenames.

- **Text Pre-processing:**  
  Functions such as `get_forum_post_text` and `expand_abbreviations` prepare the forum post text for entity extraction.

- **Entity Extraction & Standardization:**  
  Uses a LangChain `LLMChain` with a prompt template to extract entities. The `standardize_entities` function maps these entities to standardized terms using the UMLS API.

- **Verification Functions:**  
  Verifies output format, completeness, and semantic similarity with helper functions (`verify_format`, `verify_completeness`, and `verify_semantic_similarity`).

- **Pipeline Execution:**  
  The `process_post` function orchestrates the extraction, verification, and retry logic. The `main` function manages the entire workflow.

## Customization & Extension

- **Abbreviation Expansion:**  
  Modify the prompt or model settings in `expand_abbreviations` if your text contains domain-specific abbreviations.

- **Entity Extraction:**  
  Adjust the prompt template in the LangChain chain to better fit your extraction needs.

- **Verification Logic:**  
  Tune thresholds and verification methods in the semantic similarity functions to match your dataset's characteristics.

## Logging & Debugging

The script prints detailed logs and intermediate outputs to help trace the pipeline’s execution. Adjust the logging level if you require more or less verbosity.

