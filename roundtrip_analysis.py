#!/usr/bin/env python3
"""
Round-Trip Consistency Analysis for Translation Quality Prediction.

This module investigates whether round-trip translation consistency can predict
translation quality and serve as an unsupervised signal for pivot language selection.

Core Hypothesis:
    Translations with high round-trip consistency are higher quality because
    they land in "stable" semantic regions in the model's representation space.

RT Metrics Computed:
    1. RT_source:     EN → X → EN'        (Does EN survive through pivot?)
    2. RT_hop2:       X → ES' → X'        (Does pivot survive through target?)
    3. RT_output:     ES' → X' → ES''     (Is output in stable region?)
    4. RT_target_ref: ES_ref → X → ES'    (Ground-truth pivot reliability)

Usage:
    # Run full experiment
    python roundtrip_analysis.py --flores data/flores_plus.csv --target swh --n-sentences 100
    
    # Quick test with 3 sentences
    python roundtrip_analysis.py --flores data/flores_plus.csv --target swh --n-sentences 3 --test

Requirements:
    pip install pandas openai tqdm sacrebleu rapidfuzz langdetect
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Import dependencies
try:
    import pandas as pd
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pandas tqdm")
    sys.exit(1)

# Optional: language detection
try:
    from langdetect import detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not available. Install with: pip install langdetect")
    print("Language validation will be disabled.")

# Import from existing modules
from translate_with_llm import Translator, LANGUAGE_NAMES, MODELS
from evaluate_translations import (
    compute_sentence_bleu, 
    compute_sentence_chrf,
    compute_levenshtein
)


# =============================================================================
# Configuration
# =============================================================================

# Default pivot languages for experiments (strategically chosen)
DEFAULT_PIVOTS = {
    # High-resource, different families
    "fra": "French",
    "deu": "German", 
    "spa": "Spanish",
    "cmn": "Chinese (Mandarin)",
    "rus": "Russian",
    "arb": "Arabic",
    # Medium resource, potentially helpful for some targets
    "por": "Portuguese",
    "ita": "Italian",
    "hin": "Hindi",
    "tur": "Turkish",
    # Distant (control)
    "jpn": "Japanese",
    "kor": "Korean",
}

# FLORES+ language code mapping (ISO 639-3 to our internal codes)
# Note: FLORES uses ISO 639-3, we may need to map some codes
# Internal codes are typically ISO 639-1 (2-letter) used by langdetect and LANGUAGE_NAMES
FLORES_TO_INTERNAL = {
    # Major European Languages
    "eng": "en",
    "fra": "fr",
    "deu": "de",
    "spa": "es",
    "por": "pt",
    "ita": "it",
    "rus": "ru",
    "nld": "nl",  # Dutch
    "pol": "pl",  # Polish
    "ukr": "uk",  # Ukrainian
    "ces": "cs",  # Czech
    "ron": "ro",  # Romanian
    "hun": "hu",  # Hungarian
    "ell": "el",  # Greek
    "bul": "bg",  # Bulgarian
    "hrv": "hr",  # Croatian
    "srp": "sr",  # Serbian
    "slk": "sk",  # Slovak
    "slv": "sl",  # Slovenian
    "lit": "lt",  # Lithuanian
    "lav": "lv",  # Latvian
    "est": "et",  # Estonian
    "fin": "fi",  # Finnish
    "swe": "sv",  # Swedish
    "dan": "da",  # Danish
    "nob": "no",  # Norwegian Bokmål
    "isl": "is",  # Icelandic
    "mlt": "mt",  # Maltese
    "sqi": "sq",  # Albanian
    "mkd": "mk",  # Macedonian
    "bos": "bs",  # Bosnian
    "bel": "be",  # Belarusian
    
    # Celtic Languages
    "cym": "cy",  # Welsh
    "gle": "ga",  # Irish
    "gla": "gd",  # Scottish Gaelic
    "bre": "br",  # Breton
    
    # Other European
    "eus": "eu",  # Basque
    "cat": "ca",  # Catalan
    "glg": "gl",  # Galician
    
    # East Asian Languages
    "cmn": "zh",  # Chinese (Mandarin)
    "jpn": "ja",  # Japanese
    "kor": "ko",  # Korean
    "vie": "vi",  # Vietnamese
    "tha": "th",  # Thai
    "lao": "lo",  # Lao
    "khm": "km",  # Khmer
    "mya": "my",  # Burmese
    "mon": "mn",  # Mongolian
    
    # South Asian Languages
    "hin": "hi",  # Hindi
    "ben": "bn",  # Bengali
    "urd": "ur",  # Urdu
    "tam": "ta",  # Tamil
    "tel": "te",  # Telugu
    "mar": "mr",  # Marathi
    "guj": "gu",  # Gujarati
    "kan": "kn",  # Kannada
    "mal": "ml",  # Malayalam
    "pan": "pa",  # Punjabi
    "npi": "ne",  # Nepali - CRITICAL FIX for low-resource experiment
    "sin": "si",  # Sinhala
    "ori": "or",  # Odia
    "asm": "as",  # Assamese
    
    # Southeast Asian Languages
    "ind": "id",  # Indonesian
    "zsm": "ms",  # Malay (Standard)
    "msa": "ms",  # Malay (alternate code)
    "tgl": "tl",  # Tagalog
    "jav": "jv",  # Javanese
    "sun": "su",  # Sundanese
    "ceb": "ceb", # Cebuano
    
    # Middle Eastern Languages
    "arb": "ar",  # Arabic (Standard)
    "arz": "ar",  # Arabic (Egyptian)
    "heb": "he",  # Hebrew
    "fas": "fa",  # Persian
    "tur": "tr",  # Turkish
    "kur": "ku",  # Kurdish
    "pus": "ps",  # Pashto
    "aze": "az",  # Azerbaijani
    
    # African Languages
    "swh": "sw",  # Swahili
    "hau": "ha",  # Hausa
    "yor": "yo",  # Yoruba
    "ibo": "ig",  # Igbo
    "amh": "am",  # Amharic
    "orm": "om",  # Oromo
    "som": "so",  # Somali
    "zul": "zu",  # Zulu
    "xho": "xh",  # Xhosa
    "afr": "af",  # Afrikaans
    "sna": "sn",  # Shona
    "nya": "ny",  # Chichewa
    "kin": "rw",  # Kinyarwanda
    "mlg": "mg",  # Malagasy
    "tir": "ti",  # Tigrinya
    "wol": "wo",  # Wolof
    "ful": "ff",  # Fulah
    "twi": "tw",  # Twi
    "lin": "ln",  # Lingala
    "lug": "lg",  # Luganda
    
    # Central Asian Languages
    "kaz": "kk",  # Kazakh
    "uzb": "uz",  # Uzbek
    "kir": "ky",  # Kyrgyz
    "tgk": "tg",  # Tajik
    "tat": "tt",  # Tatar
    
    # Caucasian Languages
    "kat": "ka",  # Georgian
    "hye": "hy",  # Armenian
}

# Reverse mapping
INTERNAL_TO_FLORES = {v: k for k, v in FLORES_TO_INTERNAL.items()}

# Mapping from langdetect codes (ISO 639-1) to our internal codes
# langdetect returns codes like 'en', 'fr', 'sw', etc.
# Note: langdetect has limited language support, so some low-resource languages may not be detected
LANGDETECT_TO_INTERNAL = {
    # European languages
    'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
    'pt': 'pt', 'nl': 'nl', 'pl': 'pl', 'ru': 'ru', 'uk': 'uk',
    'cs': 'cs', 'ro': 'ro', 'hu': 'hu', 'el': 'el', 'bg': 'bg',
    'hr': 'hr', 'sr': 'sr', 'sk': 'sk', 'sl': 'sl', 'da': 'da',
    'no': 'no', 'sv': 'sv', 'fi': 'fi', 'et': 'et', 'lv': 'lv',
    'lt': 'lt', 'is': 'is', 'mt': 'mt', 'mk': 'mk', 'sq': 'sq',
    'bs': 'bs', 'be': 'be',
    # Celtic
    'cy': 'cy', 'ga': 'ga', 'gd': 'gd',
    # Other European
    'eu': 'eu', 'ca': 'ca', 'gl': 'gl',
    # East Asian
    'zh-cn': 'zh', 'zh-tw': 'zh', 'ja': 'ja', 'ko': 'ko',
    'vi': 'vi', 'th': 'th', 'lo': 'lo', 'km': 'km', 'my': 'my',
    # South Asian
    'hi': 'hi', 'bn': 'bn', 'ur': 'ur', 'ta': 'ta', 'te': 'te',
    'mr': 'mr', 'gu': 'gu', 'kn': 'kn', 'ml': 'ml', 'pa': 'pa',
    'ne': 'ne', 'si': 'si', 'or': 'or', 'as': 'as',
    # Southeast Asian
    'id': 'id', 'ms': 'ms', 'tl': 'tl', 'jv': 'jv', 'su': 'su',
    # Middle Eastern
    'ar': 'ar', 'he': 'he', 'fa': 'fa', 'tr': 'tr', 'ku': 'ku', 'az': 'az',
    # African
    'sw': 'sw', 'af': 'af', 'so': 'so', 'am': 'am',
    'yo': 'yo', 'ig': 'ig', 'ha': 'ha', 'zu': 'zu', 'xh': 'xh',
    # Central Asian / Caucasian
    'ka': 'ka', 'hy': 'hy', 'kk': 'kk', 'uz': 'uz', 'ky': 'ky', 'tg': 'tg',
}

# Known confusable language pairs (langdetect sometimes confuses these)
CONFUSABLE_LANGUAGES = {
    ('sw', 'sv'): "Swahili vs Swedish",  # Both use 'sw'/'sv' codes
    ('no', 'da'): "Norwegian vs Danish",
    ('hr', 'sr'): "Croatian vs Serbian",
    ('id', 'ms'): "Indonesian vs Malay",
}


def detect_language(text: str, logger: Optional[logging.Logger] = None) -> Tuple[str, float, str]:
    """
    Detect the language of text.
    
    Returns:
        Tuple of (detected_code, confidence, detected_name)
        - detected_code: Internal language code (e.g., 'sw', 'sv')
        - confidence: Detection confidence (0.0 to 1.0)
        - detected_name: Full language name (e.g., 'Swahili')
    """
    if not LANGDETECT_AVAILABLE:
        return ('unknown', 0.0, 'Unknown')
    
    try:
        results = detect_langs(text)
        if results:
            top_result = results[0]
            lang_code = str(top_result.lang)
            confidence = top_result.prob
            
            # Map to internal code
            internal_code = LANGDETECT_TO_INTERNAL.get(lang_code, lang_code)
            
            # Get full name
            lang_name = LANGUAGE_NAMES.get(internal_code, internal_code)
            
            if logger:
                logger.debug(f"    Language detected: {lang_name} ({internal_code}) [{confidence:.2%}]")
            
            return (internal_code, confidence, lang_name)
    except Exception as e:
        if logger:
            logger.debug(f"    Language detection failed: {e}")
    
    return ('unknown', 0.0, 'Unknown')


def check_language_match(
    detected_code: str, 
    expected_code: str, 
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str]:
    """
    Check if detected language matches expected language.
    
    Returns:
        Tuple of (is_match, warning_message)
    """
    if detected_code == 'unknown':
        return (True, "")  # Can't verify, assume OK
    
    if detected_code == expected_code:
        return (True, "")
    
    # Check for known confusable pairs
    pair = tuple(sorted([detected_code, expected_code]))
    if pair in CONFUSABLE_LANGUAGES:
        warning = f"LANGUAGE MISMATCH: Expected {LANGUAGE_NAMES.get(expected_code, expected_code)}, " \
                  f"got {LANGUAGE_NAMES.get(detected_code, detected_code)} " \
                  f"({CONFUSABLE_LANGUAGES[pair]})"
    else:
        warning = f"LANGUAGE MISMATCH: Expected {LANGUAGE_NAMES.get(expected_code, expected_code)}, " \
                  f"got {LANGUAGE_NAMES.get(detected_code, detected_code)}"
    
    if logger:
        logger.warning(f"    {warning}")
    
    return (False, warning)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: Path, experiment_name: str) -> logging.Logger:
    """Set up logging to both file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"{experiment_name}.log"
    
    # Create logger
    logger = logging.getLogger("roundtrip_analysis")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler (DEBUG level - everything) - USE UTF-8!
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler (INFO level - important stuff only)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Data Loading
# =============================================================================

class FLORESDataset:
    """Handler for FLORES+ dataset."""
    
    def __init__(self, csv_path: Path, logger: Optional[logging.Logger] = None):
        self.csv_path = Path(csv_path)
        self.logger = logger or logging.getLogger("roundtrip_analysis")
        self.df = None
        self._load()
    
    def _load(self):
        """Load the FLORES+ CSV file."""
        self.logger.info(f"Loading FLORES+ from {self.csv_path}...")
        
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        
        # Log dataset stats
        n_rows = len(self.df)
        languages = self.df['iso_639_3'].nunique()
        sentence_ids = self.df['id'].nunique()
        
        self.logger.info(f"  Loaded {n_rows:,} rows")
        self.logger.info(f"  {languages} unique languages")
        self.logger.info(f"  {sentence_ids} unique sentence IDs")
        
        # Check available languages
        self.available_langs = set(self.df['iso_639_3'].unique())
        self.logger.debug(f"  Available languages: {sorted(self.available_langs)[:20]}...")
    
    def get_sentence(self, sentence_id: int, lang_code: str) -> Optional[str]:
        """Get a specific sentence by ID and language code."""
        mask = (self.df['id'] == sentence_id) & (self.df['iso_639_3'] == lang_code)
        matches = self.df[mask]
        
        if len(matches) == 0:
            return None
        
        return matches.iloc[0]['text']
    
    def get_parallel_sentences(
        self, 
        source_lang: str, 
        target_lang: str, 
        n_sentences: Optional[int] = None,
        split: str = "dev"
    ) -> pd.DataFrame:
        """Get parallel sentences for source-target pair."""
        # Filter by split if specified
        df = self.df[self.df['split'] == split] if 'split' in self.df.columns else self.df
        
        # Get source sentences
        source_df = df[df['iso_639_3'] == source_lang][['id', 'text']].copy()
        source_df = source_df.rename(columns={'text': 'source_text'})
        
        # Get target sentences
        target_df = df[df['iso_639_3'] == target_lang][['id', 'text']].copy()
        target_df = target_df.rename(columns={'text': 'target_text'})
        
        # Merge on ID
        parallel = source_df.merge(target_df, on='id', how='inner')
        
        self.logger.info(f"  Found {len(parallel)} parallel pairs for {source_lang}-{target_lang}")
        
        if n_sentences and len(parallel) > n_sentences:
            parallel = parallel.head(n_sentences)
            self.logger.info(f"  Using first {n_sentences} sentences")
        
        return parallel
    
    def check_language_available(self, lang_code: str) -> bool:
        """Check if a language is available in the dataset."""
        return lang_code in self.available_langs
    
    def list_available_languages(self) -> list:
        """List all available language codes."""
        return sorted(self.available_langs)


# =============================================================================
# Similarity Metrics
# =============================================================================

def compute_similarity(text1: str, text2: str, metric: str = "chrf") -> float:
    """
    Compute similarity between two texts.
    
    Returns a score in range 0-100 where higher is more similar.
    """
    if not text1 or not text2:
        return 0.0
    
    if metric == "bleu":
        return compute_sentence_bleu(text1, text2)
    elif metric == "chrf":
        return compute_sentence_chrf(text1, text2)
    elif metric == "levenshtein":
        _, _, similarity = compute_levenshtein(text1, text2)
        return similarity
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def compute_composite_metrics(
    rt_source: float,
    rt_hop2: float, 
    rt_output: float,
    rt_target_ref: float
) -> dict:
    """Compute all composite RT metrics."""
    # All four metrics
    all_metrics = [rt_source, rt_hop2, rt_output, rt_target_ref]
    # Source-target chain only (without reference-based)
    st_metrics = [rt_source, rt_hop2, rt_output]
    
    # Handle edge cases
    def safe_geometric_mean(values):
        if any(v <= 0 for v in values):
            return 0.0
        return math.exp(sum(math.log(v) for v in values) / len(values))
    
    def safe_harmonic_mean(values):
        if any(v <= 0 for v in values):
            return 0.0
        return len(values) / sum(1/v for v in values)
    
    def safe_product(values):
        # Scale to 0-100 range
        result = 1.0
        for v in values:
            result *= (v / 100)
        return result * 100
    
    return {
        "rt_min": min(all_metrics),
        "rt_geometric": safe_geometric_mean(all_metrics),
        "rt_harmonic": safe_harmonic_mean(all_metrics),
        "rt_product": safe_product(all_metrics),
        "rt_min_st": min(st_metrics),
        "rt_geometric_st": safe_geometric_mean(st_metrics),
    }


# =============================================================================
# Core RT Computation
# =============================================================================

class RoundTripAnalyzer:
    """Main class for computing round-trip consistency metrics."""
    
    def __init__(
        self,
        translator: Translator,
        similarity_metric: str = "chrf",
        logger: Optional[logging.Logger] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        self.translator = translator
        self.similarity_metric = similarity_metric
        self.logger = logger or logging.getLogger("roundtrip_analysis")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Track API calls for logging
        self.api_calls = 0
    
    def _translate(self, text: str, source: str, target: str) -> str:
        """Wrapper for translation with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self.api_calls += 1
                result = self.translator.translate_direct(text, source, target)
                
                # Check for None response
                if result is None or result.translation is None:
                    raise ValueError("API returned None response")
                
                translation = result.translation.strip()
                if not translation:
                    raise ValueError("API returned empty translation")
                
                return translation
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"    Translation attempt {attempt + 1} failed: {e}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"    Translation failed after {self.max_retries} attempts: {e}")
        
        # All retries failed
        raise last_error
    
    def compute_full_rt(
        self,
        source_text: str,
        target_reference: str,
        source_lang: str,
        pivot_lang: str,
        target_lang: str,
    ) -> dict:
        """
        Compute all RT metrics for a single sentence through a pivot.
        
        Translation chain: source_lang → pivot_lang → target_lang
        
        API calls made (7 total):
            1. source → pivot (forward hop 1)
            2. pivot → target (forward hop 2)  
            3. pivot → source (back for rt_source)
            4. target_output → pivot (back for rt_hop2)
            5. pivot_back → target (forward for rt_output)
            6. target_ref → pivot (for rt_target_ref)
            7. pivot_from_ref → target (for rt_target_ref)
        
        Returns dict with all translations and metrics.
        """
        start_time = time.time()
        
        # Map to internal language codes if needed
        src = FLORES_TO_INTERNAL.get(source_lang, source_lang)
        pvt = FLORES_TO_INTERNAL.get(pivot_lang, pivot_lang)
        tgt = FLORES_TO_INTERNAL.get(target_lang, target_lang)
        
        self.logger.debug(f"  Computing RT: {src} -> {pvt} -> {tgt}")
        
        # === Forward chain ===
        # Call 1: source → pivot
        pivot_text = self._translate(source_text, src, pvt)
        
        # Call 2: pivot → target (this is our output)
        output_text = self._translate(pivot_text, pvt, tgt)
        
        # === Language Detection (validate output is in correct language) ===
        detected_code, detection_confidence, detected_name = detect_language(output_text, self.logger)
        language_match, language_warning = check_language_match(detected_code, tgt, self.logger)
        
        if not language_match:
            self.logger.warning(f"    Output may be in wrong language: expected {LANGUAGE_NAMES.get(tgt, tgt)}, detected {detected_name}")
        
        # === RT_source: EN → X → EN' ===
        # Call 3: pivot → source
        back_to_source = self._translate(pivot_text, pvt, src)
        rt_source = compute_similarity(source_text, back_to_source, self.similarity_metric)
        
        # === RT_hop2: X → ES' → X' ===
        # Call 4: output → pivot
        back_to_pivot = self._translate(output_text, tgt, pvt)
        rt_hop2 = compute_similarity(pivot_text, back_to_pivot, self.similarity_metric)
        
        # === RT_output: ES' → X' → ES'' ===
        # Call 5: back_to_pivot → target
        output_roundtrip = self._translate(back_to_pivot, pvt, tgt)
        rt_output = compute_similarity(output_text, output_roundtrip, self.similarity_metric)
        
        # === RT_target_ref: ES_ref → X → ES_ref' ===
        # Call 6: reference → pivot
        ref_to_pivot = self._translate(target_reference, tgt, pvt)
        # Call 7: pivot_from_ref → target
        ref_via_pivot = self._translate(ref_to_pivot, pvt, tgt)
        rt_target_ref = compute_similarity(target_reference, ref_via_pivot, self.similarity_metric)
        
        # Compute composites
        composites = compute_composite_metrics(rt_source, rt_hop2, rt_output, rt_target_ref)
        
        # Compute quality metrics (output vs reference)
        quality_bleu = compute_sentence_bleu(output_text, target_reference)
        quality_chrf = compute_sentence_chrf(output_text, target_reference)
        _, _, quality_lev = compute_levenshtein(output_text, target_reference)
        
        elapsed = time.time() - start_time
        
        return {
            # Translations
            "pivot_text": pivot_text,
            "output_text": output_text,
            "back_to_source": back_to_source,
            "back_to_pivot": back_to_pivot,
            "output_roundtrip": output_roundtrip,
            "ref_to_pivot": ref_to_pivot,
            "ref_via_pivot": ref_via_pivot,
            # RT metrics
            "rt_source": rt_source,
            "rt_hop2": rt_hop2,
            "rt_output": rt_output,
            "rt_target_ref": rt_target_ref,
            **composites,
            # Quality metrics
            "quality_bleu": quality_bleu,
            "quality_chrf": quality_chrf,
            "quality_levenshtein": quality_lev,
            # Language detection
            "detected_lang_code": detected_code,
            "detected_lang_name": detected_name,
            "detection_confidence": detection_confidence,
            "language_match": language_match,
            "language_warning": language_warning if not language_match else "",
            # Timing
            "elapsed_time": elapsed,
        }


# =============================================================================
# Experiment Runner
# =============================================================================

class RTExperiment:
    """Runner for round-trip consistency experiments."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.results = []
        
        # Set up output paths
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"rt_{config['source_lang']}_{config['target_lang']}_{timestamp}"
        
        self.results_file = output_dir / f"{self.experiment_name}_results.jsonl"
        self.summary_file = output_dir / f"{self.experiment_name}_summary.json"
        self.checkpoint_file = output_dir / f"{self.experiment_name}_checkpoint.json"
        
        # Initialize components
        self.translator = None
        self.dataset = None
        self.analyzer = None
        
        # Track progress
        self.completed = set()  # (sentence_id, pivot_lang) pairs
    
    def initialize(self):
        """Initialize translator, dataset, and analyzer."""
        self.logger.info("=" * 60)
        self.logger.info("ROUND-TRIP CONSISTENCY EXPERIMENT")
        self.logger.info("=" * 60)
        self.logger.info(f"Source: {self.config['source_lang']}")
        self.logger.info(f"Target: {self.config['target_lang']}")
        self.logger.info(f"Pivots: {', '.join(self.config['pivot_langs'])}")
        self.logger.info(f"Sentences: {self.config['n_sentences']}")
        self.logger.info(f"Model: {self.config['model']}")
        self.logger.info(f"Output: {self.config['output_dir']}")
        self.logger.info("=" * 60)
        
        # Initialize translator
        self.logger.info("\nInitializing translator...")
        self.translator = Translator(model_key=self.config['model'])
        
        # Load dataset
        self.logger.info("\nLoading FLORES+ dataset...")
        self.dataset = FLORESDataset(self.config['flores_path'], self.logger)
        
        # Validate languages
        self._validate_languages()
        
        # Initialize analyzer
        self.analyzer = RoundTripAnalyzer(self.translator, logger=self.logger)
        
        # Load checkpoint if resuming
        if self.config.get('resume', True):
            self._load_checkpoint()
        
        # Save config
        self._save_config()
    
    def _validate_languages(self):
        """Validate that all required languages are in the dataset."""
        self.logger.info("\nValidating language availability...")
        
        required = [self.config['source_lang'], self.config['target_lang']] + self.config['pivot_langs']
        missing = []
        
        for lang in required:
            if not self.dataset.check_language_available(lang):
                missing.append(lang)
                self.logger.warning(f"  [X] {lang} NOT FOUND in dataset")
            else:
                self.logger.info(f"  [OK] {lang} available")
        
        if missing:
            available = self.dataset.list_available_languages()
            self.logger.error(f"\nMissing languages: {missing}")
            self.logger.error(f"Available languages include: {available[:30]}...")
            raise ValueError(f"Languages not found in dataset: {missing}")
    
    def _load_checkpoint(self):
        """Load checkpoint to resume interrupted experiment."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            self.completed = set(tuple(x) for x in checkpoint.get('completed', []))
            self.logger.info(f"Loaded checkpoint: {len(self.completed)} pairs already completed")
    
    def _save_checkpoint(self):
        """Save checkpoint for resumption."""
        checkpoint = {
            'completed': list(self.completed),
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f)
    
    def _save_config(self):
        """Save experiment configuration."""
        config_file = Path(self.config['output_dir']) / f"{self.experiment_name}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def _save_result(self, result: dict):
        """Append a result to the JSONL file."""
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def run(self):
        """Run the full experiment."""
        self.initialize()
        
        # Get parallel sentences
        self.logger.info("\nFetching parallel sentences...")
        parallel_df = self.dataset.get_parallel_sentences(
            self.config['source_lang'],
            self.config['target_lang'],
            self.config['n_sentences']
        )
        
        total_pairs = len(parallel_df) * len(self.config['pivot_langs'])
        remaining = total_pairs - len(self.completed)
        
        self.logger.info(f"\nTotal pairs to process: {total_pairs}")
        self.logger.info(f"Already completed: {len(self.completed)}")
        self.logger.info(f"Remaining: {remaining}")
        self.logger.info(f"Estimated API calls: {remaining * 7}")
        
        if remaining == 0:
            self.logger.info("All pairs already completed!")
            return
        
        self.logger.info("\n" + "-" * 60)
        self.logger.info("Starting experiment...")
        self.logger.info("-" * 60 + "\n")
        
        # Create progress bar
        pbar = tqdm(total=remaining, desc="Processing", unit="pair")
        
        errors = []
        start_time = time.time()
        
        for _, row in parallel_df.iterrows():
            sentence_id = row['id']
            source_text = row['source_text']
            target_ref = row['target_text']
            
            for pivot in self.config['pivot_langs']:
                pair_key = (sentence_id, pivot)
                
                # Skip if already done
                if pair_key in self.completed:
                    continue
                
                try:
                    # Compute RT metrics
                    result = self.analyzer.compute_full_rt(
                        source_text=source_text,
                        target_reference=target_ref,
                        source_lang=self.config['source_lang'],
                        pivot_lang=pivot,
                        target_lang=self.config['target_lang'],
                    )
                    
                    # Add metadata
                    result.update({
                        'sentence_id': int(sentence_id),
                        'source_lang': self.config['source_lang'],
                        'pivot_lang': pivot,
                        'target_lang': self.config['target_lang'],
                        'source_text': source_text,
                        'target_reference': target_ref,
                        'model': self.config['model'],
                        'timestamp': datetime.now().isoformat(),
                    })
                    
                    # Save result
                    self._save_result(result)
                    self.results.append(result)
                    
                    # Mark as completed
                    self.completed.add(pair_key)
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({
                        'sent': sentence_id,
                        'pivot': pivot,
                        'bleu': f"{result['quality_bleu']:.1f}",
                        'rt_src': f"{result['rt_source']:.1f}",
                    })
                    
                    # Periodic checkpoint
                    if len(self.completed) % 10 == 0:
                        self._save_checkpoint()
                    
                except Exception as e:
                    error_msg = f"Error on sentence {sentence_id}, pivot {pivot}: {e}"
                    self.logger.error(error_msg)
                    errors.append({'sentence_id': sentence_id, 'pivot': pivot, 'error': str(e)})
                    pbar.update(1)
        
        pbar.close()
        
        # Final checkpoint
        self._save_checkpoint()
        
        # Generate summary
        elapsed = time.time() - start_time
        self._generate_summary(elapsed, errors)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Results saved to: {self.results_file}")
        self.logger.info(f"Summary saved to: {self.summary_file}")
    
    def _generate_summary(self, elapsed_time: float, errors: list):
        """Generate experiment summary with statistics."""
        if not self.results:
            self.logger.warning("No results to summarize")
            return
        
        df = pd.DataFrame(self.results)
        
        # Language mismatch statistics
        lang_mismatches = []
        if 'language_match' in df.columns:
            mismatch_df = df[df['language_match'] == False]
            n_mismatches = len(mismatch_df)
            mismatch_rate = n_mismatches / len(df) * 100 if len(df) > 0 else 0
            
            # Get details of mismatches
            if n_mismatches > 0:
                for _, row in mismatch_df.iterrows():
                    lang_mismatches.append({
                        'sentence_id': int(row['sentence_id']),
                        'pivot': row['pivot_lang'],
                        'expected': self.config['target_lang'],
                        'detected': row.get('detected_lang_code', 'unknown'),
                        'detected_name': row.get('detected_lang_name', 'Unknown'),
                        'confidence': float(row.get('detection_confidence', 0)),
                        'warning': row.get('language_warning', ''),
                    })
        else:
            n_mismatches = 0
            mismatch_rate = 0
        
        # Overall statistics
        summary = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'timing': {
                'total_seconds': elapsed_time,
                'total_minutes': elapsed_time / 60,
                'seconds_per_pair': elapsed_time / len(self.results) if self.results else 0,
                'api_calls': self.analyzer.api_calls,
            },
            'counts': {
                'sentences': int(df['sentence_id'].nunique()),
                'pivots': int(df['pivot_lang'].nunique()),
                'total_pairs': len(df),
                'errors': len(errors),
                'language_mismatches': n_mismatches,
                'language_mismatch_rate': mismatch_rate,
            },
            'quality_metrics': {
                'mean_bleu': float(df['quality_bleu'].mean()),
                'std_bleu': float(df['quality_bleu'].std()),
                'mean_chrf': float(df['quality_chrf'].mean()),
                'std_chrf': float(df['quality_chrf'].std()),
            },
            'rt_metrics': {
                'mean_rt_source': float(df['rt_source'].mean()),
                'mean_rt_hop2': float(df['rt_hop2'].mean()),
                'mean_rt_output': float(df['rt_output'].mean()),
                'mean_rt_target_ref': float(df['rt_target_ref'].mean()),
                'mean_rt_min': float(df['rt_min'].mean()),
                'mean_rt_geometric': float(df['rt_geometric'].mean()),
            },
            'per_pivot_summary': {},
            'language_mismatches': lang_mismatches,
            'errors': errors,
        }
        
        # Per-pivot statistics
        for pivot in self.config['pivot_langs']:
            pivot_df = df[df['pivot_lang'] == pivot]
            if len(pivot_df) > 0:
                # Count mismatches for this pivot
                pivot_mismatches = 0
                if 'language_match' in pivot_df.columns:
                    pivot_mismatches = int((pivot_df['language_match'] == False).sum())
                
                summary['per_pivot_summary'][pivot] = {
                    'n_sentences': len(pivot_df),
                    'mean_bleu': float(pivot_df['quality_bleu'].mean()),
                    'mean_chrf': float(pivot_df['quality_chrf'].mean()),
                    'mean_rt_source': float(pivot_df['rt_source'].mean()),
                    'mean_rt_hop2': float(pivot_df['rt_hop2'].mean()),
                    'mean_rt_output': float(pivot_df['rt_output'].mean()),
                    'mean_rt_target_ref': float(pivot_df['rt_target_ref'].mean()),
                    'mean_rt_min': float(pivot_df['rt_min'].mean()),
                    'mean_rt_geometric': float(pivot_df['rt_geometric'].mean()),
                    'language_mismatches': pivot_mismatches,
                }
        
        # Save summary
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Log summary
        self.logger.info("\n" + "-" * 60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("-" * 60)
        self.logger.info(f"Total pairs processed: {len(df)}")
        self.logger.info(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        self.logger.info(f"API calls made: {self.analyzer.api_calls}")
        
        # Log language mismatch warning
        if n_mismatches > 0:
            self.logger.warning(f"\n[!] LANGUAGE MISMATCHES: {n_mismatches}/{len(df)} ({mismatch_rate:.1f}%)")
            self.logger.warning(f"    Some outputs may be in the wrong language!")
        
        self.logger.info(f"\nQuality Metrics (mean +/- std):")
        self.logger.info(f"  BLEU: {df['quality_bleu'].mean():.2f} +/- {df['quality_bleu'].std():.2f}")
        self.logger.info(f"  chrF: {df['quality_chrf'].mean():.2f} +/- {df['quality_chrf'].std():.2f}")
        self.logger.info(f"\nRT Metrics (mean):")
        self.logger.info(f"  RT_source:     {df['rt_source'].mean():.2f}")
        self.logger.info(f"  RT_hop2:       {df['rt_hop2'].mean():.2f}")
        self.logger.info(f"  RT_output:     {df['rt_output'].mean():.2f}")
        self.logger.info(f"  RT_target_ref: {df['rt_target_ref'].mean():.2f}")
        self.logger.info(f"\nPer-Pivot Quality (BLEU):")
        for pivot, stats in summary['per_pivot_summary'].items():
            mismatch_note = f" [!{stats.get('language_mismatches', 0)} lang errors]" if stats.get('language_mismatches', 0) > 0 else ""
            self.logger.info(f"  {pivot}: {stats['mean_bleu']:.2f}{mismatch_note}")


# =============================================================================
# Analysis Functions  
# =============================================================================

def load_results(results_file: Path) -> pd.DataFrame:
    """Load results from JSONL file into DataFrame."""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return pd.DataFrame(results)


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between RT metrics and quality metrics."""
    try:
        from scipy import stats
    except ImportError:
        print("scipy not installed, skipping correlation analysis")
        return pd.DataFrame()
    
    rt_cols = ['rt_source', 'rt_hop2', 'rt_output', 'rt_target_ref', 
               'rt_min', 'rt_geometric', 'rt_harmonic', 'rt_min_st', 'rt_geometric_st']
    quality_cols = ['quality_bleu', 'quality_chrf', 'quality_levenshtein']
    
    results = []
    for rt in rt_cols:
        for qual in quality_cols:
            if rt in df.columns and qual in df.columns:
                r, p = stats.pearsonr(df[rt], df[qual])
                results.append({
                    'rt_metric': rt,
                    'quality_metric': qual,
                    'pearson_r': r,
                    'p_value': p,
                    'significant': p < 0.05,
                })
    
    return pd.DataFrame(results)


def analyze_pivot_prediction_accuracy(df: pd.DataFrame) -> dict:
    """Analyze how well RT metrics predict the best pivot for each sentence."""
    results = {
        'per_sentence': [],
        'overall': {},
    }
    
    rt_metrics = ['rt_source', 'rt_hop2', 'rt_output', 'rt_target_ref', 
                  'rt_min', 'rt_geometric', 'rt_min_st']
    
    for sentence_id in df['sentence_id'].unique():
        sent_df = df[df['sentence_id'] == sentence_id]
        
        if len(sent_df) < 2:
            continue
        
        # Actual best pivot (by BLEU)
        actual_best_idx = sent_df['quality_bleu'].idxmax()
        actual_best_pivot = sent_df.loc[actual_best_idx, 'pivot_lang']
        best_bleu = sent_df['quality_bleu'].max()
        
        sent_result = {
            'sentence_id': sentence_id,
            'actual_best_pivot': actual_best_pivot,
            'best_bleu': best_bleu,
        }
        
        # Predicted best pivot for each RT metric
        for rt in rt_metrics:
            if rt in sent_df.columns:
                pred_idx = sent_df[rt].idxmax()
                pred_pivot = sent_df.loc[pred_idx, 'pivot_lang']
                pred_bleu = sent_df.loc[pred_idx, 'quality_bleu']
                regret = best_bleu - pred_bleu
                
                sent_result[f'{rt}_pred_pivot'] = pred_pivot
                sent_result[f'{rt}_correct'] = pred_pivot == actual_best_pivot
                sent_result[f'{rt}_regret'] = regret
        
        results['per_sentence'].append(sent_result)
    
    # Aggregate accuracy for each RT metric
    per_sent_df = pd.DataFrame(results['per_sentence'])
    
    for rt in rt_metrics:
        correct_col = f'{rt}_correct'
        regret_col = f'{rt}_regret'
        
        if correct_col in per_sent_df.columns:
            results['overall'][rt] = {
                'accuracy': float(per_sent_df[correct_col].mean()),
                'mean_regret': float(per_sent_df[regret_col].mean()),
                'max_regret': float(per_sent_df[regret_col].max()),
            }
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Round-Trip Consistency Analysis for Translation Quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment with Swahili as target
  python roundtrip_analysis.py --flores data/flores.csv --target swh --n-sentences 100
  
  # Quick test with 3 sentences
  python roundtrip_analysis.py --flores data/flores.csv --target swh --n-sentences 3 --test
  
  # Custom pivots
  python roundtrip_analysis.py --flores data/flores.csv --target swh --pivots fra,deu,arb
  
  # Analyze existing results
  python roundtrip_analysis.py --analyze results/rt_experiment_results.jsonl
        """
    )
    
    # Input/output
    parser.add_argument("--flores", type=str, help="Path to FLORES+ CSV file")
    parser.add_argument("--output-dir", type=str, default="./data/roundtrip_results",
                        help="Output directory for results")
    
    # Experiment settings
    parser.add_argument("--source", type=str, default="eng",
                        help="Source language (ISO 639-3 code)")
    parser.add_argument("--target", type=str, required=False,
                        help="Target language (ISO 639-3 code)")
    parser.add_argument("--pivots", type=str, default=None,
                        help="Comma-separated pivot languages (default: use standard set)")
    parser.add_argument("--n-sentences", type=int, default=100,
                        help="Number of sentences to process")
    
    # Model settings
    parser.add_argument("--model", type=str, default="oss120",
                        choices=list(MODELS.keys()),
                        help="Model to use for translations")
    
    # Execution options
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, don't resume from checkpoint")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: use fewer pivots for quick validation")
    
    # Analysis mode
    parser.add_argument("--analyze", type=str,
                        help="Analyze existing results file instead of running experiment")
    
    args = parser.parse_args()
    
    # Analysis mode
    if args.analyze:
        print(f"Analyzing results from: {args.analyze}")
        df = load_results(Path(args.analyze))
        
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)
        corr_df = compute_correlations(df)
        if len(corr_df) > 0:
            print(corr_df.to_string(index=False))
        
        print("\n" + "=" * 60)
        print("PIVOT PREDICTION ACCURACY")
        print("=" * 60)
        pred_results = analyze_pivot_prediction_accuracy(df)
        for rt, stats in pred_results['overall'].items():
            print(f"{rt}:")
            print(f"  Accuracy: {stats['accuracy']*100:.1f}%")
            print(f"  Mean Regret: {stats['mean_regret']:.2f} BLEU")
        
        return
    
    # Experiment mode
    if not args.flores or not args.target:
        parser.error("--flores and --target are required for experiment mode")
    
    # Parse pivot languages
    if args.pivots:
        pivot_langs = [p.strip() for p in args.pivots.split(',')]
    elif args.test:
        # Minimal set for testing
        pivot_langs = ["fra", "deu", "arb"]
    else:
        # Default set
        pivot_langs = list(DEFAULT_PIVOTS.keys())
    
    # Set up logging
    output_dir = Path(args.output_dir)
    experiment_name = f"rt_{args.source}_{args.target}"
    logger = setup_logging(output_dir, experiment_name)
    
    # Create config
    config = {
        'source_lang': args.source,
        'target_lang': args.target,
        'pivot_langs': pivot_langs,
        'n_sentences': args.n_sentences,
        'model': args.model,
        'output_dir': str(output_dir),
        'flores_path': args.flores,
        'resume': not args.no_resume,
    }
    
    # Run experiment
    experiment = RTExperiment(config, logger)
    experiment.run()


if __name__ == "__main__":
    main()