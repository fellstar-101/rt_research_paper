#!/usr/bin/env python3
"""
Semantic Analysis for Round-Trip Translation Consistency.

This script computes semantic similarity metrics using embeddings (SONAR/LaBSE)
and tests hypotheses about RT-quality relationships.

Version 4.2 Updates:
    - Added per-sample CSV export for post-hoc analysis (scatter plots, etc.)
    - Added direct_source_output as a predictor in ROC and correlation analyses
    - Added combined metrics: combined_rt_direct_geometric, combined_rt_direct_mean, combined_rt_direct_min
    - Extended rank_correlation to compute Spearman for all quality metrics (not just BLEU)
    - Added granularity_analysis: within-band correlations (quartile/decile splits by quality)
    - Tests whether RT can do fine-grained ranking or only coarse classification
    - Added pivot filtering: --pivots and --exclude-pivots for fair cross-language comparison
    - Added cross-pivot comparison summary: ranks pivots by AUC, correlation, smoke detector strength

Version 4.1 Updates:
    - Split four_way_breakdown into four_way_breakdown_chrf and four_way_breakdown_semantic
    - Split lift_analysis into lift_analysis_chrf and lift_analysis_semantic
    - Semantic versions use rt_output_semantic when stratifying by rt_source_semantic

Version 4 Updates:
    - Added pure RT composites (rt_min_st, rt_geometric_st) excluding target_ref
    - Added semantic pure composites (rt_min_semantic_st, rt_geometric_semantic_st)
    - Added direct_source_output metric (cross-lingual meaning preservation)
    - Updated quality_semantic thresholds to [0.92, 0.90, 0.85]
    - Added conditional_analysis for paradox investigation
    - Added cross-tabulation and four-way breakdown analysis
    - Added lift analysis for rt_output in different strata

Version 3 Updates:
    - Per-pivot decomposition for ALL analyses (enabled by default)
    - Bootstrap confidence intervals for ROC
    - All RT metrics (surface + semantic) included in per-pivot summaries
    - Semantic RT metrics included in stratified correlations

Hypotheses tested:
    A: RT can detect catastrophic translation failures
    E: Semantic quality metrics may correlate better with RT than surface metrics
    F: Semantic RT metrics may predict quality better than surface RT metrics

Usage:
    # Basic analysis with auto-detected embedding model
    python semantic_analysis.py results.jsonl
    
    # Specify output directory
    python semantic_analysis.py results.jsonl --output-dir analysis/
    
    # Use specific model
    python semantic_analysis.py results.jsonl --model labse
    
    # Use both models for comparison
    python semantic_analysis.py results.jsonl --model both
    
    # With COMET quality scoring and COMET-QE drift detection
    python semantic_analysis.py results.jsonl \
        --comet-path my_comet_models/models--Unbabel--wmt22-comet-da \
        --comet-qe-path my_comet_models/models--Unbabel--wmt22-cometkiwi-da
    
    # Disable per-pivot analysis (faster but less detailed)
    python semantic_analysis.py results.jsonl --no-per-pivot
    
    # Analyze only shared pivots (for fair cross-language comparison)
    python semantic_analysis.py results.jsonl --pivots fra,deu,cmn
    
    # Exclude specific pivots (e.g., exclude Hindi from Nepali analysis)
    python semantic_analysis.py results.jsonl --exclude-pivots hin

Requirements:
    pip install numpy scipy pandas
    # Plus at least one of:
    pip install sentence-transformers  # For LaBSE
    pip install fairseq2 sonar-space   # For SONAR
    # Optional (for COMET quality metrics):
    pip install unbabel-comet
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Check for scipy
try:
    from scipy import stats
    from scipy.spatial.distance import cosine as cosine_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")


# =============================================================================
# Embedding Models
# =============================================================================

class EmbeddingModel:
    """Base class for embedding models."""
    
    def __init__(self, name: str):
        self.name = name
        self.dimension = None
    
    def embed(self, text: str, lang: str = None) -> np.ndarray:
        """Embed a single text. Returns 1D numpy array."""
        raise NotImplementedError
    
    def embed_batch(self, texts: List[str], lang: str = None) -> np.ndarray:
        """Embed multiple texts. Returns 2D numpy array (n_texts, dimension)."""
        raise NotImplementedError


class LaBSEModel(EmbeddingModel):
    """LaBSE embedding model via sentence-transformers."""
    
    def __init__(self):
        super().__init__("labse")
        self.model = None
        self.dimension = 768
    
    def _load(self):
        """Lazy-load the model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print("  Loading LaBSE model...")
            self.model = SentenceTransformer('sentence-transformers/LaBSE')
            print("  LaBSE loaded successfully")
    
    def embed(self, text: str, lang: str = None) -> np.ndarray:
        """Embed a single text. Lang is ignored (auto-detected)."""
        self._load()
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    
    def embed_batch(self, texts: List[str], lang: str = None) -> np.ndarray:
        """Embed multiple texts."""
        self._load()
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class SONARModel(EmbeddingModel):
    """SONAR embedding model from Meta."""
    
    def __init__(self):
        super().__init__("sonar")
        self.encoder = None
        self.dimension = 1024
        
        # SONAR language code mapping (to FLORES-200 codes)
        self.lang_map = {
            'en': 'eng_Latn', 'eng': 'eng_Latn',
            'es': 'spa_Latn', 'spa': 'spa_Latn',
            'fr': 'fra_Latn', 'fra': 'fra_Latn',
            'de': 'deu_Latn', 'deu': 'deu_Latn',
            'it': 'ita_Latn', 'ita': 'ita_Latn',
            'pt': 'por_Latn', 'por': 'por_Latn',
            'ru': 'rus_Cyrl', 'rus': 'rus_Cyrl',
            'zh': 'zho_Hans', 'zho': 'zho_Hans', 'cmn': 'zho_Hans',
            'ja': 'jpn_Jpan', 'jpn': 'jpn_Jpan',
            'ko': 'kor_Hang', 'kor': 'kor_Hang',
            'ar': 'arb_Arab', 'arb': 'arb_Arab',
            'hi': 'hin_Deva', 'hin': 'hin_Deva',
            'sw': 'swh_Latn', 'swh': 'swh_Latn',
            'tr': 'tur_Latn', 'tur': 'tur_Latn',
            'vi': 'vie_Latn', 'vie': 'vie_Latn',
            'th': 'tha_Thai', 'tha': 'tha_Thai',
            'nl': 'nld_Latn', 'nld': 'nld_Latn',
            'pl': 'pol_Latn', 'pol': 'pol_Latn',
            'uk': 'ukr_Cyrl', 'ukr': 'ukr_Cyrl',
            'ro': 'ron_Latn', 'ron': 'ron_Latn',
            'el': 'ell_Grek', 'ell': 'ell_Grek',
            'he': 'heb_Hebr', 'heb': 'heb_Hebr',
            'cs': 'ces_Latn', 'ces': 'ces_Latn',
            'hu': 'hun_Latn', 'hun': 'hun_Latn',
            'id': 'ind_Latn', 'ind': 'ind_Latn',
            'ms': 'zsm_Latn', 'zsm': 'zsm_Latn',
            'bn': 'ben_Beng', 'ben': 'ben_Beng',
            'ta': 'tam_Taml', 'tam': 'tam_Taml',
            'te': 'tel_Telu', 'tel': 'tel_Telu',
            'mr': 'mar_Deva', 'mar': 'mar_Deva',
            'fa': 'pes_Arab', 'pes': 'pes_Arab',
        }
    
    def _load(self):
        """Lazy-load the model."""
        if self.encoder is None:
            print("  Loading SONAR model...")
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
            self.encoder = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder"
            )
            print("  SONAR loaded successfully")
    
    def _get_sonar_lang(self, lang: str) -> str:
        """Convert language code to SONAR format."""
        if lang in self.lang_map:
            return self.lang_map[lang]
        # Try direct use
        if '_' in lang:
            return lang
        # Default to English
        return 'eng_Latn'
    
    def embed(self, text: str, lang: str = 'eng') -> np.ndarray:
        """Embed a single text."""
        self._load()
        sonar_lang = self._get_sonar_lang(lang)
        embeddings = self.encoder.predict([text], source_lang=sonar_lang)
        return embeddings[0].cpu().numpy()
    
    def embed_batch(self, texts: List[str], lang: str = 'eng') -> np.ndarray:
        """Embed multiple texts."""
        self._load()
        sonar_lang = self._get_sonar_lang(lang)
        embeddings = self.encoder.predict(texts, source_lang=sonar_lang)
        return embeddings.cpu().numpy()


def get_available_models() -> Dict[str, bool]:
    """Check which embedding models are available."""
    available = {'labse': False, 'sonar': False}
    
    try:
        from sentence_transformers import SentenceTransformer
        available['labse'] = True
    except ImportError:
        pass
    
    try:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        available['sonar'] = True
    except ImportError:
        pass
    
    return available


def load_embedding_model(model_name: str) -> EmbeddingModel:
    """Load a specific embedding model."""
    if model_name == 'labse':
        return LaBSEModel()
    elif model_name == 'sonar':
        return SONARModel()
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: labse, sonar")


# =============================================================================
# COMET Models for Quality Scoring
# =============================================================================

def _suppress_all_comet_output():
    """Suppress all verbose output from COMET and its dependencies."""
    import os
    import logging as log_module
    import warnings
    
    # Environment variables to suppress PyTorch Lightning output
    os.environ["PYTORCH_LIGHTNING_SILENT"] = "1"
    os.environ["PL_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warnings
    
    # Suppress all relevant loggers
    for logger_name in [
        "pytorch_lightning",
        "pytorch_lightning.utilities.rank_zero",
        "pytorch_lightning.accelerators.cuda",
        "pytorch_lightning.utilities.distributed",
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
        "lightning.fabric",
        "transformers",
        "transformers.modeling_utils",
        "comet",
        "comet.models",
        "comet.encoders",
        "urllib3",
        "filelock",
        "torch",
        "sentencepiece",
        "sentence_transformers",
    ]:
        logger = log_module.getLogger(logger_name)
        logger.setLevel(log_module.ERROR)
        logger.propagate = False
    
    # Also suppress root logger's existing handlers from printing low-level messages
    root_logger = log_module.getLogger()
    root_logger.setLevel(log_module.WARNING)
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*progress_bar.*")
    warnings.filterwarnings("ignore", message=".*GPU available.*")
    warnings.filterwarnings("ignore", message=".*TPU available.*")
    warnings.filterwarnings("ignore", message=".*checkpoint.*")


class COMETModel:
    """COMET reference-based quality model for computing quality_comet."""
    
    def __init__(self, model_path: str = "my_comet_models/models--Unbabel--wmt22-comet-da"):
        self.model_path = model_path
        self.model = None
        self.name = "comet"
    
    def _load(self):
        """Lazy-load the model."""
        if self.model is None:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"COMET model not found at: {self.model_path}\n"
                    f"Please ensure the model is downloaded to this location."
                )
            
            # Suppress ALL output before importing comet
            _suppress_all_comet_output()
            
            from comet import load_from_checkpoint
            print(f"  Loading COMET model from {self.model_path}...")
            checkpoint_path = self._find_checkpoint(model_path)
            self.model = load_from_checkpoint(checkpoint_path)
            print("  COMET loaded successfully")
    
    def _find_checkpoint(self, model_path: Path) -> str:
        """Find the checkpoint file in the model directory."""
        ckpt_files = list(model_path.rglob("*.ckpt"))
        if ckpt_files:
            return str(ckpt_files[0])
        return str(model_path)
    
    def score(self, sources: List[str], hypotheses: List[str], 
              references: List[str], batch_size: int = 8) -> List[float]:
        """
        Score translations with references.
        
        Args:
            sources: Source texts
            hypotheses: Translation outputs (MT)
            references: Reference translations
            batch_size: Batch size for processing
            
        Returns:
            List of quality scores
        """
        self._load()
        data = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(sources, hypotheses, references)
        ]
        
        # Suppress output again in case it was re-enabled
        _suppress_all_comet_output()
        
        # Run prediction with all progress/logging disabled
        # Note: Using accelerator="cpu" handles device selection
        output = self.model.predict(
            data, 
            batch_size=batch_size, 
            accelerator="cpu",
            num_workers=0,
            progress_bar=False,
        )
        return list(output.scores)


class COMETQEModel:
    """COMET-QE reference-free quality model for computing direct_source_output_comet."""
    
    def __init__(self, model_path: str = "my_comet_models/models--Unbabel--wmt22-cometkiwi-da"):
        self.model_path = model_path
        self.model = None
        self.name = "comet_qe"
    
    def _load(self):
        """Lazy-load the model."""
        if self.model is None:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"COMET-QE model not found at: {self.model_path}\n"
                    f"Please ensure the model is downloaded to this location."
                )
            
            # Suppress ALL output before importing comet
            _suppress_all_comet_output()
            
            from comet import load_from_checkpoint
            print(f"  Loading COMET-QE model from {self.model_path}...")
            checkpoint_path = self._find_checkpoint(model_path)
            self.model = load_from_checkpoint(checkpoint_path)
            print("  COMET-QE loaded successfully")
    
    def _find_checkpoint(self, model_path: Path) -> str:
        """Find the checkpoint file in the model directory."""
        ckpt_files = list(model_path.rglob("*.ckpt"))
        if ckpt_files:
            return str(ckpt_files[0])
        return str(model_path)
    
    def score(self, sources: List[str], hypotheses: List[str], 
              batch_size: int = 8) -> List[float]:
        """
        Score translations without references (quality estimation).
        
        Args:
            sources: Source texts
            hypotheses: Translation outputs (MT)
            batch_size: Batch size for processing
            
        Returns:
            List of quality scores
        """
        self._load()
        data = [
            {"src": src, "mt": mt}
            for src, mt in zip(sources, hypotheses)
        ]
        
        # Suppress output again in case it was re-enabled
        _suppress_all_comet_output()
        
        # Run prediction with all progress/logging disabled
        # Note: Using accelerator="cpu" handles device selection
        output = self.model.predict(
            data, 
            batch_size=batch_size, 
            accelerator="cpu",
            num_workers=0,
            progress_bar=False,
        )
        return list(output.scores)


# =============================================================================
# Similarity Functions
# =============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    
    # Flatten if needed
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    # Compute cosine similarity
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# =============================================================================
# Semantic Metrics Computer
# =============================================================================

class SemanticMetricsComputer:
    """Computes semantic similarity metrics using embeddings."""
    
    def __init__(self, model: EmbeddingModel, logger: Optional[logging.Logger] = None,
                 comet_model: Optional['COMETModel'] = None,
                 comet_qe_model: Optional['COMETQEModel'] = None,
                 comet_batch_size: int = 8):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self._embedding_cache = {}
        self.comet_model = comet_model
        self.comet_qe_model = comet_qe_model
        self.comet_batch_size = comet_batch_size
    
    def _get_embedding(self, text: str, lang: str) -> np.ndarray:
        """Get embedding with caching."""
        cache_key = (text, lang, self.model.name)
        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self.model.embed(text, lang)
        return self._embedding_cache[cache_key]
    
    def compute_for_row(self, row: dict) -> dict:
        """Compute semantic metrics for a single result row."""
        src_lang = row.get('source_lang', 'eng')
        tgt_lang = row.get('target_lang', 'jpn')
        pivot_lang = row.get('pivot_lang', 'fra')
        
        metrics = {}
        
        # === Quality: output vs reference ===
        embed_output = self._get_embedding(row['output_text'], tgt_lang)
        embed_reference = self._get_embedding(row['target_reference'], tgt_lang)
        
        metrics['quality_semantic'] = cosine_similarity(embed_output, embed_reference)
        
        # Magnitude ratio (should be ~1 for good embeddings)
        metrics['quality_magnitude_ratio'] = float(
            np.linalg.norm(embed_output) / np.linalg.norm(embed_reference)
        ) if np.linalg.norm(embed_reference) > 0 else 0.0
        
        # === RT Source Semantic: source vs back_to_source ===
        embed_source = self._get_embedding(row['source_text'], src_lang)
        embed_back_source = self._get_embedding(row['back_to_source'], src_lang)
        
        metrics['rt_source_semantic'] = cosine_similarity(embed_source, embed_back_source)
        
        # === RT Hop2 Semantic: pivot vs back_to_pivot ===
        embed_pivot = self._get_embedding(row['pivot_text'], pivot_lang)
        embed_back_pivot = self._get_embedding(row['back_to_pivot'], pivot_lang)
        
        metrics['rt_hop2_semantic'] = cosine_similarity(embed_pivot, embed_back_pivot)
        
        # === RT Output Semantic: output vs output_roundtrip ===
        embed_output_rt = self._get_embedding(row['output_roundtrip'], tgt_lang)
        
        metrics['rt_output_semantic'] = cosine_similarity(embed_output, embed_output_rt)
        
        # === RT Target Ref Semantic: reference vs ref_via_pivot ===
        embed_ref_via = self._get_embedding(row['ref_via_pivot'], tgt_lang)
        
        metrics['rt_target_ref_semantic'] = cosine_similarity(embed_reference, embed_ref_via)
        
        # === Composite semantic metrics (all four) ===
        semantic_rts = [
            metrics['rt_source_semantic'],
            metrics['rt_hop2_semantic'],
            metrics['rt_output_semantic'],
            metrics['rt_target_ref_semantic']
        ]
        
        metrics['rt_min_semantic'] = min(semantic_rts)
        metrics['rt_geometric_semantic'] = float(np.exp(np.mean(np.log(np.clip(semantic_rts, 1e-10, 1)))))
        
        # === Pure semantic composites (excluding target_ref - true RT only) ===
        pure_semantic_rts = [
            metrics['rt_source_semantic'],
            metrics['rt_hop2_semantic'],
            metrics['rt_output_semantic']
        ]
        
        metrics['rt_min_semantic_st'] = min(pure_semantic_rts)
        metrics['rt_geometric_semantic_st'] = float(np.exp(np.mean(np.log(np.clip(pure_semantic_rts, 1e-10, 1)))))
        
        # === Direct source-to-output similarity (cross-lingual meaning preservation) ===
        # LaBSE version - kept for comparison with COMET-QE version
        metrics['direct_source_output_labse'] = cosine_similarity(embed_source, embed_output)
        
        # NOTE: direct_source_output_comet and combined metrics are computed in batch
        # in compute_for_dataframe() for efficiency
        
        return metrics
    
    def compute_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute semantic metrics for all rows in a DataFrame.
        
        Uses:
        - LaBSE for RT metrics (rt_*_semantic)
        - COMET (with reference) for quality_comet
        - COMET-QE for direct_source_output_comet
        """
        import time
        
        n_rows = len(df)
        self.logger.info(f"Computing semantic metrics for {n_rows} rows...")
        
        # === Step 1: Compute COMET quality scores in batch (if model available) ===
        quality_comet_scores = None
        if self.comet_model is not None:
            n_batches = (n_rows + self.comet_batch_size - 1) // self.comet_batch_size
            self.logger.info(f"  Computing quality_comet (COMET with references)...")
            self.logger.info(f"    Processing {n_rows} samples in {n_batches} batches (batch_size={self.comet_batch_size})")
            start_comet = time.time()
            
            try:
                sources = df['source_text'].tolist()
                outputs = df['output_text'].tolist()
                references = df['target_reference'].tolist()
                
                quality_comet_scores = self.comet_model.score(
                    sources, outputs, references, 
                    batch_size=self.comet_batch_size
                )
                
                comet_time = time.time() - start_comet
                self.logger.info(f"    COMET scoring completed in {comet_time:.1f}s ({comet_time/n_rows:.2f}s per row)")
            except Exception as e:
                self.logger.error(f"    COMET scoring failed: {e}")
                self.logger.info("    Continuing without quality_comet...")
                quality_comet_scores = None
        
        # === Step 2: Compute COMET-QE direct_source_output in batch (if model available) ===
        direct_source_output_comet_scores = None
        if self.comet_qe_model is not None:
            n_batches = (n_rows + self.comet_batch_size - 1) // self.comet_batch_size
            self.logger.info("  Computing direct_source_output_comet (COMET-QE)...")
            self.logger.info(f"    Processing {n_rows} samples in {n_batches} batches (batch_size={self.comet_batch_size})")
            start_qe = time.time()
            
            try:
                sources = df['source_text'].tolist()
                outputs = df['output_text'].tolist()
                
                direct_source_output_comet_scores = self.comet_qe_model.score(
                    sources, outputs,
                    batch_size=self.comet_batch_size
                )
                
                qe_time = time.time() - start_qe
                self.logger.info(f"    COMET-QE scoring completed in {qe_time:.1f}s ({qe_time/n_rows:.2f}s per row)")
            except Exception as e:
                self.logger.error(f"    COMET-QE scoring failed: {e}")
                self.logger.info("    Continuing without direct_source_output_comet...")
                direct_source_output_comet_scores = None
        
        # === Step 3: Compute LaBSE-based metrics row by row ===
        self.logger.info(f"  Computing LaBSE-based semantic metrics...")
        self.logger.info(f"    (This may take a few minutes - ~2-3 sec per row)")
        
        all_metrics = []
        start_time = time.time()
        
        for idx, row in df.iterrows():
            row_num = len(all_metrics) + 1
            if row_num % 10 == 0 or row_num == 1:
                elapsed = time.time() - start_time
                if row_num > 1:
                    rate = elapsed / (row_num - 1)
                    remaining = rate * (n_rows - row_num + 1)
                    self.logger.info(f"    Processing row {row_num}/{n_rows} ({100*row_num/n_rows:.0f}%) - ~{remaining:.0f}s remaining...")
                else:
                    self.logger.info(f"    Processing row {row_num}/{n_rows}...")
            
            try:
                metrics = self.compute_for_row(row.to_dict())
                all_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Error computing semantics for row {idx}: {e}")
                all_metrics.append({
                    'quality_semantic': np.nan,
                    'quality_magnitude_ratio': np.nan,
                    'rt_source_semantic': np.nan,
                    'rt_hop2_semantic': np.nan,
                    'rt_output_semantic': np.nan,
                    'rt_target_ref_semantic': np.nan,
                    'rt_min_semantic': np.nan,
                    'rt_geometric_semantic': np.nan,
                    'rt_min_semantic_st': np.nan,
                    'rt_geometric_semantic_st': np.nan,
                    'direct_source_output_labse': np.nan,
                })
        
        total_time = time.time() - start_time
        self.logger.info(f"  LaBSE metrics completed in {total_time:.1f}s ({total_time/n_rows:.2f}s per row)")
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # === Step 4: Add COMET scores ===
        if quality_comet_scores is not None:
            metrics_df['quality_comet'] = quality_comet_scores
        
        if direct_source_output_comet_scores is not None:
            metrics_df['direct_source_output_comet'] = direct_source_output_comet_scores
        
        # === Step 5: Compute combined metrics (LaBSE RT + COMET-QE direct) ===
        if direct_source_output_comet_scores is not None and 'rt_geometric_semantic_st' in metrics_df.columns:
            self.logger.info("  Computing combined metrics (LaBSE RT + COMET-QE)...")
            
            rt_geom = metrics_df['rt_geometric_semantic_st'].values
            direct_comet = metrics_df['direct_source_output_comet'].values
            
            # Geometric mean
            metrics_df['combined_rt_direct_geometric'] = np.sqrt(rt_geom * direct_comet)
            # Arithmetic mean  
            metrics_df['combined_rt_direct_mean'] = (rt_geom + direct_comet) / 2
            # Min (conservative)
            metrics_df['combined_rt_direct_min'] = np.minimum(rt_geom, direct_comet)
            
            self.logger.info("    Note: combined metrics mix LaBSE RT (~0.95+) with COMET-QE (~0.85)")
        
        # Combine with original dataframe
        result = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)
        
        self.logger.info("Semantic metrics computed successfully")
        return result


# =============================================================================
# Hypothesis A: Failure Detection Analysis
# =============================================================================

class HypothesisAAnalyzer:
    """Analyzes whether RT metrics can detect translation failures."""
    
    def __init__(self, df: pd.DataFrame, logger: Optional[logging.Logger] = None):
        self.df = df
        self.logger = logger or logging.getLogger(__name__)
    
    def roc_analysis(self, rt_metric: str, quality_metric: str = 'quality_chrf',
                     thresholds: List[float] = None, bootstrap_n: int = 1000) -> dict:
        """
        ROC analysis at multiple quality thresholds with bootstrap confidence intervals.
        
        For each quality threshold, compute AUC for using rt_metric to predict failure.
        
        Args:
            rt_metric: RT metric to use as predictor
            quality_metric: Quality metric to define failure
            thresholds: Quality thresholds (failure = below threshold)
            bootstrap_n: Number of bootstrap samples for CI (0 to disable)
        """
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}
        
        # Set default thresholds based on quality metric
        if thresholds is None:
            if quality_metric == 'quality_chrf':
                thresholds = [50, 55, 60]
            elif quality_metric == 'quality_semantic':
                thresholds = [0.92, 0.90, 0.85]
            elif quality_metric == 'quality_comet':
                thresholds = [0.7, 0.8, 0.85]
            else:
                thresholds = [0.7, 0.8, 0.85]  # Default to COMET-like
        
        results = {}
        
        for threshold in thresholds:
            # Filter out rows with NaN in either quality or RT metric
            valid_mask = self.df[quality_metric].notna() & self.df[rt_metric].notna()
            valid_df = self.df[valid_mask]
            
            if len(valid_df) < 5:
                results[f'threshold_{threshold}'] = {
                    'auc': None,
                    'n_failures': 0,
                    'n_successes': 0,
                    'note': 'Insufficient valid data after removing NaN'
                }
                continue
            
            # Define failure as quality below threshold
            labels = (valid_df[quality_metric] < threshold).astype(int)
            
            n_failures = labels.sum()
            n_successes = len(labels) - n_failures
            
            if n_failures == 0 or n_successes == 0:
                results[f'threshold_{threshold}'] = {
                    'auc': None,
                    'n_failures': int(n_failures),
                    'n_successes': int(n_successes),
                    'note': 'Cannot compute AUC: only one class present'
                }
                continue
            
            # RT scores (higher RT should predict success, so we negate for failure prediction)
            rt_scores = -valid_df[rt_metric].values
            
            # Compute ROC AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(labels, rt_scores)
                
                result = {
                    'auc': float(auc),
                    'n_failures': int(n_failures),
                    'n_successes': int(n_successes),
                    'interpretation': self._interpret_auc(auc)
                }
                
                # Bootstrap confidence interval if requested and sufficient samples
                if bootstrap_n > 0 and min(n_failures, n_successes) >= 5:
                    ci_lower, ci_upper = self._bootstrap_auc_ci(
                        labels.values, rt_scores, n_bootstrap=bootstrap_n
                    )
                    result['ci_lower'] = float(ci_lower)
                    result['ci_upper'] = float(ci_upper)
                    result['ci_width'] = float(ci_upper - ci_lower)
                
                results[f'threshold_{threshold}'] = result
                
            except ImportError:
                # Fallback without sklearn
                auc = self._compute_auc_manual(labels.values, rt_scores)
                results[f'threshold_{threshold}'] = {
                    'auc': float(auc),
                    'n_failures': int(n_failures),
                    'n_successes': int(n_successes),
                    'interpretation': self._interpret_auc(auc)
                }
        
        return results
    
    def _bootstrap_auc_ci(self, labels: np.ndarray, scores: np.ndarray, 
                          n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for AUC."""
        from sklearn.metrics import roc_auc_score
        
        n = len(labels)
        aucs = []
        
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = rng.choice(n, size=n, replace=True)
            boot_labels = labels[indices]
            boot_scores = scores[indices]
            
            # Skip if only one class in bootstrap sample
            if len(np.unique(boot_labels)) < 2:
                continue
            
            try:
                auc = roc_auc_score(boot_labels, boot_scores)
                aucs.append(auc)
            except:
                continue
        
        if len(aucs) < 100:
            # Not enough valid bootstrap samples
            return (np.nan, np.nan)
        
        ci_lower = np.percentile(aucs, 100 * alpha / 2)
        ci_upper = np.percentile(aucs, 100 * (1 - alpha / 2))
        
        return (ci_lower, ci_upper)
    
    def _compute_auc_manual(self, labels: np.ndarray, scores: np.ndarray) -> float:
        """Compute AUC manually (Mann-Whitney U statistic)."""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        # Count pairs where positive score > negative score
        count = 0
        for ps in pos_scores:
            count += np.sum(ps > neg_scores) + 0.5 * np.sum(ps == neg_scores)
        
        return count / (n_pos * n_neg)
    
    def _interpret_auc(self, auc: float) -> str:
        """Interpret AUC value."""
        if auc is None:
            return "Cannot compute"
        if auc >= 0.9:
            return "Excellent discriminative power"
        if auc >= 0.8:
            return "Good discriminative power"
        if auc >= 0.7:
            return "Moderate discriminative power"
        if auc >= 0.6:
            return "Weak discriminative power"
        return "Poor discriminative power (near random)"
    
    def stratified_correlations(self, rt_metrics: List[str], 
                                stratify_by: str = 'quality_chrf',
                                correlate_with: List[str] = None) -> dict:
        """
        Compute correlations within quality strata.
        
        Args:
            rt_metrics: RT metrics to correlate
            stratify_by: Quality metric to stratify by
            correlate_with: Quality metrics to correlate with (default: all available)
        """
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}
        
        # Default: correlate with all quality metrics
        if correlate_with is None:
            correlate_with = ['quality_chrf', 'quality_semantic', 'quality_comet']
            correlate_with = [c for c in correlate_with if c in self.df.columns]
        
        median_quality = self.df[stratify_by].median()
        
        low_df = self.df[self.df[stratify_by] < median_quality]
        high_df = self.df[self.df[stratify_by] >= median_quality]
        
        results = {
            'stratify_by': stratify_by,
            'median_threshold': float(median_quality),
            'low_stratum': {'n': len(low_df), 'correlations': {}},
            'high_stratum': {'n': len(high_df), 'correlations': {}},
        }
        
        # Track skipped correlations due to constant input
        skipped_constant = []
        
        for rt in rt_metrics:
            if rt not in self.df.columns:
                continue
            
            results['low_stratum']['correlations'][rt] = {}
            results['high_stratum']['correlations'][rt] = {}
            
            for quality in correlate_with:
                if quality not in self.df.columns:
                    continue
                
                # Low stratum - drop rows where EITHER column is NaN
                if len(low_df) > 2:
                    try:
                        valid_low = low_df[[rt, quality]].dropna()
                        if len(valid_low) > 2:
                            # Check for constant input (near-zero variance)
                            rt_std = valid_low[rt].std()
                            quality_std = valid_low[quality].std()
                            
                            if rt_std < 1e-10:
                                results['low_stratum']['correlations'][rt][quality] = {
                                    'r': float('nan'), 'p': float('nan'), 
                                    'note': f'RT metric is constant in low stratum'
                                }
                                skipped_constant.append(f"low:{rt}:{quality}(RT constant)")
                            elif quality_std < 1e-10:
                                results['low_stratum']['correlations'][rt][quality] = {
                                    'r': float('nan'), 'p': float('nan'), 
                                    'note': f'Quality metric is constant in low stratum'
                                }
                                skipped_constant.append(f"low:{rt}:{quality}(quality constant)")
                            else:
                                r, p = stats.pearsonr(valid_low[rt], valid_low[quality])
                                results['low_stratum']['correlations'][rt][quality] = {
                                    'r': float(r), 'p': float(p), 'significant': str(p < 0.05)
                                }
                        else:
                            results['low_stratum']['correlations'][rt][quality] = {'error': 'Insufficient valid data'}
                    except Exception as e:
                        results['low_stratum']['correlations'][rt][quality] = {'error': str(e)}
                
                # High stratum - drop rows where EITHER column is NaN
                if len(high_df) > 2:
                    try:
                        valid_high = high_df[[rt, quality]].dropna()
                        if len(valid_high) > 2:
                            # Check for constant input (near-zero variance)
                            rt_std = valid_high[rt].std()
                            quality_std = valid_high[quality].std()
                            
                            if rt_std < 1e-10:
                                results['high_stratum']['correlations'][rt][quality] = {
                                    'r': float('nan'), 'p': float('nan'), 
                                    'note': f'RT metric is constant in high stratum'
                                }
                                skipped_constant.append(f"high:{rt}:{quality}(RT constant)")
                            elif quality_std < 1e-10:
                                results['high_stratum']['correlations'][rt][quality] = {
                                    'r': float('nan'), 'p': float('nan'), 
                                    'note': f'Quality metric is constant in high stratum'
                                }
                                skipped_constant.append(f"high:{rt}:{quality}(quality constant)")
                            else:
                                r, p = stats.pearsonr(valid_high[rt], valid_high[quality])
                                results['high_stratum']['correlations'][rt][quality] = {
                                    'r': float(r), 'p': float(p), 'significant': str(p < 0.05)
                                }
                        else:
                            results['high_stratum']['correlations'][rt][quality] = {'error': 'Insufficient valid data'}
                    except Exception as e:
                        results['high_stratum']['correlations'][rt][quality] = {'error': str(e)}
        
        # Print warning if correlations were skipped due to constant input
        if skipped_constant:
            # Group by reason for cleaner output
            quality_constant = [s for s in skipped_constant if 'quality constant' in s]
            rt_constant = [s for s in skipped_constant if 'RT constant' in s]
            
            if quality_constant:
                # Extract unique quality metrics that were constant
                constant_metrics = set()
                for s in quality_constant:
                    parts = s.split(':')
                    if len(parts) >= 3:
                        constant_metrics.add(parts[2].split('(')[0])
                print(f"  [!] Skipped {len(quality_constant)} correlations: quality metric(s) {constant_metrics} constant in stratum")
            
            if rt_constant:
                print(f"  [!] Skipped {len(rt_constant)} correlations: RT metric constant in stratum")
        
        return results
    
    def outlier_agreement(self, rt_metric: str, quality_metric: str = 'quality_chrf',
                          n_std: float = 1.5) -> dict:
        """
        Check agreement between RT outliers and quality outliers.
        """
        # Define outliers as values below mean - n_std * std
        rt_mean = self.df[rt_metric].mean()
        rt_std = self.df[rt_metric].std()
        rt_threshold = rt_mean - n_std * rt_std
        
        quality_mean = self.df[quality_metric].mean()
        quality_std = self.df[quality_metric].std()
        quality_threshold = quality_mean - n_std * quality_std
        
        rt_outliers = self.df[rt_metric] < rt_threshold
        quality_outliers = self.df[quality_metric] < quality_threshold
        
        n_rt_outliers = rt_outliers.sum()
        n_quality_outliers = quality_outliers.sum()
        n_overlap = (rt_outliers & quality_outliers).sum()
        
        # Precision: of RT outliers, how many are quality outliers?
        precision = n_overlap / n_rt_outliers if n_rt_outliers > 0 else 0
        
        # Recall: of quality outliers, how many are RT outliers?
        recall = n_overlap / n_quality_outliers if n_quality_outliers > 0 else 0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'n_std': n_std,
            'rt_threshold': float(rt_threshold),
            'quality_threshold': float(quality_threshold),
            'n_rt_outliers': int(n_rt_outliers),
            'n_quality_outliers': int(n_quality_outliers),
            'n_overlap': int(n_overlap),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    
    def rank_correlation(self, rt_metrics: List[str], 
                         quality_metric: str = 'quality_chrf') -> dict:
        """
        Compute Spearman rank correlation between RT and quality.
        """
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}
        
        results = {}
        
        for rt in rt_metrics:
            if rt not in self.df.columns:
                continue
            
            try:
                # Drop rows where EITHER column is NaN
                valid = self.df[[rt, quality_metric]].dropna()
                if len(valid) > 2:
                    rho, p = stats.spearmanr(valid[rt], valid[quality_metric])
                    results[rt] = {
                        'spearman_rho': float(rho),
                        'p_value': float(p),
                        'significant': str(p < 0.05)
                    }
                else:
                    results[rt] = {'error': 'Insufficient valid data'}
            except Exception as e:
                results[rt] = {'error': str(e)}
        
        return results
    
    def granularity_analysis(self, rt_metrics: List[str], 
                             quality_metric: str = 'quality_semantic',
                             n_bands: int = 4) -> dict:
        """
        Analyze RT prediction granularity by computing correlations within quality bands.
        
        This tests whether RT can distinguish within quality bands (e.g., 0.90 vs 0.95)
        or only do coarse classification (poor vs good).
        
        Args:
            rt_metrics: RT metrics to test
            quality_metric: Quality metric to stratify by
            n_bands: Number of quality bands (4 = quartiles, 10 = deciles)
        
        Returns:
            Dict with correlation and ranking metrics within each quality band
        """
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}
        
        if quality_metric not in self.df.columns:
            return {'error': f'{quality_metric} not in columns'}
        
        results = {
            'quality_metric': quality_metric,
            'n_bands': n_bands,
            'bands': {}
        }
        
        # Compute band boundaries using quantiles
        quantiles = [i / n_bands for i in range(n_bands + 1)]
        boundaries = self.df[quality_metric].quantile(quantiles).tolist()
        
        for i in range(n_bands):
            band_low = boundaries[i]
            band_high = boundaries[i + 1]
            
            # Get samples in this band
            if i == n_bands - 1:  # Include upper boundary for last band
                band_df = self.df[(self.df[quality_metric] >= band_low) & 
                                  (self.df[quality_metric] <= band_high)]
            else:
                band_df = self.df[(self.df[quality_metric] >= band_low) & 
                                  (self.df[quality_metric] < band_high)]
            
            band_key = f'band_{i+1}_of_{n_bands}'
            band_label = f'Q{i+1}' if n_bands == 4 else f'D{i+1}' if n_bands == 10 else f'B{i+1}'
            
            results['bands'][band_key] = {
                'label': band_label,
                'quality_range': [float(band_low), float(band_high)],
                'n': len(band_df),
                'correlations': {}
            }
            
            for rt in rt_metrics:
                if rt not in self.df.columns:
                    continue
                
                try:
                    valid = band_df[[rt, quality_metric]].dropna()
                    if len(valid) > 5:  # Need sufficient samples
                        # Check for constant input
                        rt_std = valid[rt].std()
                        quality_std = valid[quality_metric].std()
                        
                        if rt_std < 1e-10 or quality_std < 1e-10:
                            results['bands'][band_key]['correlations'][rt] = {
                                'pearson_r': None,
                                'spearman_rho': None,
                                'note': 'Constant input in band'
                            }
                        else:
                            # Pearson (linear correlation)
                            r, p_pearson = stats.pearsonr(valid[rt], valid[quality_metric])
                            # Spearman (rank correlation - better for granularity)
                            rho, p_spearman = stats.spearmanr(valid[rt], valid[quality_metric])
                            
                            results['bands'][band_key]['correlations'][rt] = {
                                'pearson_r': float(r),
                                'pearson_p': float(p_pearson),
                                'spearman_rho': float(rho),
                                'spearman_p': float(p_spearman),
                                'n_valid': len(valid)
                            }
                    else:
                        results['bands'][band_key]['correlations'][rt] = {
                            'error': f'Insufficient data (n={len(valid)})'
                        }
                except Exception as e:
                    results['bands'][band_key]['correlations'][rt] = {'error': str(e)}
        
        # Compute summary: average correlation across bands
        summary = {}
        for rt in rt_metrics:
            if rt not in self.df.columns:
                continue
            
            pearson_vals = []
            spearman_vals = []
            for band_key, band_data in results['bands'].items():
                if rt in band_data['correlations']:
                    corr = band_data['correlations'][rt]
                    if 'pearson_r' in corr and corr['pearson_r'] is not None:
                        pearson_vals.append(corr['pearson_r'])
                    if 'spearman_rho' in corr and corr['spearman_rho'] is not None:
                        spearman_vals.append(corr['spearman_rho'])
            
            if pearson_vals:
                summary[rt] = {
                    'mean_pearson_r': float(np.mean(pearson_vals)),
                    'mean_spearman_rho': float(np.mean(spearman_vals)) if spearman_vals else None,
                    'n_bands_with_data': len(pearson_vals)
                }
        
        results['summary'] = summary
        
        return results
    
    def full_analysis(self, bootstrap_n: int = 1000) -> dict:
        """Run all Hypothesis A analyses."""
        # Define RT metrics to test
        surface_rt = ['rt_source', 'rt_hop2', 'rt_output', 'rt_target_ref', 
                      'rt_min', 'rt_geometric', 'rt_min_st', 'rt_geometric_st']
        semantic_rt = ['rt_source_semantic', 'rt_hop2_semantic', 'rt_output_semantic', 
                       'rt_target_ref_semantic', 'rt_min_semantic', 'rt_geometric_semantic',
                       'rt_min_semantic_st', 'rt_geometric_semantic_st']
        
        # Filter to available columns
        surface_rt = [c for c in surface_rt if c in self.df.columns]
        semantic_rt = [c for c in semantic_rt if c in self.df.columns]
        all_rt = surface_rt + semantic_rt
        
        # Add direct_source_output variants as predictors
        for dso in ['direct_source_output_labse', 'direct_source_output_comet']:
            if dso in self.df.columns:
                all_rt.append(dso)
        
        # Add combined RT + direct_source_output metrics
        combined_metrics = ['combined_rt_direct_geometric', 'combined_rt_direct_mean', 'combined_rt_direct_min']
        for cm in combined_metrics:
            if cm in self.df.columns:
                all_rt.append(cm)
        
        results = {
            'roc_analysis': {},
            'stratified_correlations': {},
            'outlier_agreement': {},
            'rank_correlation': {},
        }
        
        # === ROC analysis for ALL RT metrics with multiple failure definitions ===
        self.logger.info("  Running ROC analysis...")
        
        # Define quality metrics and their thresholds (BLEU removed)
        quality_definitions = {
            'quality_chrf': [50, 55, 60],
            'quality_semantic': [0.92, 0.90, 0.85],
            'quality_comet': [0.7, 0.8, 0.85]
        }
        
        for quality_metric, thresholds in quality_definitions.items():
            if quality_metric not in self.df.columns:
                continue
            
            results['roc_analysis'][quality_metric] = {}
            
            for rt in all_rt:
                results['roc_analysis'][quality_metric][rt] = self.roc_analysis(
                    rt, quality_metric=quality_metric, thresholds=thresholds,
                    bootstrap_n=bootstrap_n
                )
        
        # === Stratified correlations with multiple stratification metrics ===
        # Now includes BOTH surface and semantic RT in BOTH analyses
        self.logger.info("  Running stratified correlation analysis...")
        
        for stratify_by in ['quality_chrf', 'quality_semantic', 'quality_comet']:
            if stratify_by not in self.df.columns:
                continue
            
            results['stratified_correlations'][f'stratified_by_{stratify_by}'] = {
                'surface': self.stratified_correlations(surface_rt, stratify_by=stratify_by),
                'semantic': self.stratified_correlations(semantic_rt, stratify_by=stratify_by) if semantic_rt else {}
            }
        
        # === Outlier agreement ===
        for rt in ['rt_min', 'rt_min_st', 'rt_min_semantic', 'rt_min_semantic_st']:
            if rt in self.df.columns:
                results['outlier_agreement'][rt] = self.outlier_agreement(rt)
        
        # === Rank correlations (Spearman) for all quality metrics ===
        # Spearman is more appropriate for ranking ability than Pearson
        results['rank_correlation'] = {}
        for quality_metric in ['quality_chrf', 'quality_semantic', 'quality_comet']:
            if quality_metric in self.df.columns:
                results['rank_correlation'][quality_metric] = self.rank_correlation(all_rt, quality_metric=quality_metric)
        
        # === Granularity Analysis (within-band correlations) ===
        # Tests whether RT can rank within quality bands (fine-grained) or only classify (coarse)
        self.logger.info("  Running granularity analysis...")
        results['granularity_analysis'] = {}
        for quality_metric in ['quality_chrf', 'quality_semantic', 'quality_comet']:
            if quality_metric in self.df.columns:
                # Quartile analysis (4 bands)
                results['granularity_analysis'][f'{quality_metric}_quartiles'] = \
                    self.granularity_analysis(all_rt, quality_metric=quality_metric, n_bands=4)
                # Decile analysis (10 bands) - finer granularity test
                results['granularity_analysis'][f'{quality_metric}_deciles'] = \
                    self.granularity_analysis(all_rt, quality_metric=quality_metric, n_bands=10)
        
        # === Conditional Analysis (paradox analysis) ===
        self.logger.info("  Running conditional analysis (paradox analysis)...")
        results['conditional_analysis'] = self.conditional_analysis()
        
        return results
    
    def conditional_analysis(self) -> dict:
        """
        Analyze how predictive power of RT metrics changes based on stratification.
        
        This implements the "paradox analysis" showing rt_output is more predictive
        when rt_source is low than when rt_source is high.
        
        Returns:
            Dict with results stratified by rt_source and rt_source_semantic
        """
        results = {}
        
        # Stratify by both surface and semantic rt_source
        for stratify_by in ['rt_source', 'rt_source_semantic']:
            if stratify_by not in self.df.columns:
                continue
            
            stratify_results = self._conditional_analysis_single(stratify_by)
            results[f'stratified_by_{stratify_by}'] = stratify_results
        
        return results
    
    def _conditional_analysis_single(self, stratify_by: str) -> dict:
        """Run conditional analysis for a single stratification metric."""
        if stratify_by not in self.df.columns:
            return {'error': f'{stratify_by} not in columns'}
        
        # Define predict metrics based on stratify_by
        if 'semantic' in stratify_by:
            predict_metrics = ['rt_hop2_semantic', 'rt_output_semantic']
            # Also include surface for comparison
            predict_metrics += ['rt_hop2', 'rt_output']
        else:
            predict_metrics = ['rt_hop2', 'rt_output']
            # Also include semantic for comparison
            if 'rt_hop2_semantic' in self.df.columns:
                predict_metrics += ['rt_hop2_semantic', 'rt_output_semantic']
        
        predict_metrics = [m for m in predict_metrics if m in self.df.columns]
        
        quality_metrics = ['quality_chrf', 'quality_semantic']
        quality_metrics = [m for m in quality_metrics if m in self.df.columns]
        
        # Compute thresholds
        stratify_values = self.df[stratify_by].dropna()
        q1 = float(np.percentile(stratify_values, 25))
        median = float(np.median(stratify_values))
        q3 = float(np.percentile(stratify_values, 75))
        
        # Quality thresholds
        quality_thresholds = {}
        for qm in quality_metrics:
            qvals = self.df[qm].dropna()
            quality_thresholds[f'{qm}_median'] = float(np.median(qvals))
        
        results = {
            'thresholds': {
                'stratify_q1': q1,
                'stratify_median': median,
                'stratify_q3': q3,
                **quality_thresholds
            },
            'median_split': {},
            'quartile_split': {},
            'cross_tabulation_chrf': {},
            'cross_tabulation_semantic': {},
            'four_way_breakdown_chrf': {},
            'four_way_breakdown_semantic': {},
            'lift_analysis_chrf': {},
            'lift_analysis_semantic': {}
        }
        
        # Define strata
        low_median = self.df[stratify_by] <= median
        high_median = self.df[stratify_by] > median
        low_q1 = self.df[stratify_by] <= q1
        high_q3 = self.df[stratify_by] >= q3
        
        # === Median split correlations ===
        for pred in predict_metrics:
            for qual in quality_metrics:
                key = f'{pred}_vs_{qual}'
                try:
                    # High stratum
                    high_data = self.df[high_median]
                    r_high = np.corrcoef(
                        high_data[pred].fillna(0), 
                        high_data[qual].fillna(0)
                    )[0, 1]
                    
                    # Low stratum
                    low_data = self.df[low_median]
                    r_low = np.corrcoef(
                        low_data[pred].fillna(0), 
                        low_data[qual].fillna(0)
                    )[0, 1]
                    
                    results['median_split'][key] = {
                        'r_high': float(r_high) if not np.isnan(r_high) else None,
                        'r_low': float(r_low) if not np.isnan(r_low) else None,
                        'diff': float(r_low - r_high) if not (np.isnan(r_high) or np.isnan(r_low)) else None
                    }
                except Exception as e:
                    results['median_split'][key] = {'error': str(e)}
        
        # === Quartile split correlations ===
        for pred in predict_metrics:
            for qual in quality_metrics:
                key = f'{pred}_vs_{qual}'
                try:
                    # High stratum (Q3)
                    high_data = self.df[high_q3]
                    r_high = np.corrcoef(
                        high_data[pred].fillna(0), 
                        high_data[qual].fillna(0)
                    )[0, 1] if len(high_data) > 5 else np.nan
                    
                    # Low stratum (Q1)
                    low_data = self.df[low_q1]
                    r_low = np.corrcoef(
                        low_data[pred].fillna(0), 
                        low_data[qual].fillna(0)
                    )[0, 1] if len(low_data) > 5 else np.nan
                    
                    results['quartile_split'][key] = {
                        'r_high_q3': float(r_high) if not np.isnan(r_high) else None,
                        'r_low_q1': float(r_low) if not np.isnan(r_low) else None,
                        'diff': float(r_low - r_high) if not (np.isnan(r_high) or np.isnan(r_low)) else None
                    }
                except Exception as e:
                    results['quartile_split'][key] = {'error': str(e)}
        
        # === Cross-tabulation for chrF ===
        if 'quality_chrf' in self.df.columns:
            results['cross_tabulation_chrf'] = self._cross_tabulation(
                stratify_by, 'rt_output', 'quality_chrf', q1, q3
            )
        
        # === Cross-tabulation for semantic ===
        if 'quality_semantic' in self.df.columns:
            # Determine which rt_output to use
            rt_out = 'rt_output_semantic' if 'semantic' in stratify_by and 'rt_output_semantic' in self.df.columns else 'rt_output'
            results['cross_tabulation_semantic'] = self._cross_tabulation(
                stratify_by, rt_out, 'quality_semantic', q1, q3
            )
        
        # === Four-way breakdown for chrF ===
        if 'quality_chrf' in self.df.columns and 'rt_output' in self.df.columns:
            results['four_way_breakdown_chrf'] = self._four_way_breakdown(
                stratify_by, 'rt_output', 'quality_chrf', q1, q3
            )
        
        # === Four-way breakdown for semantic ===
        if 'quality_semantic' in self.df.columns:
            # Determine which rt_output to use - semantic when stratifying by semantic
            rt_out_sem = 'rt_output_semantic' if 'semantic' in stratify_by and 'rt_output_semantic' in self.df.columns else 'rt_output'
            if rt_out_sem in self.df.columns:
                results['four_way_breakdown_semantic'] = self._four_way_breakdown(
                    stratify_by, rt_out_sem, 'quality_semantic', q1, q3
                )
        
        # === Lift analysis for chrF ===
        if 'quality_chrf' in self.df.columns and 'rt_output' in self.df.columns:
            results['lift_analysis_chrf'] = self._lift_analysis(
                stratify_by, 'rt_output', 'quality_chrf', q1, q3
            )
        
        # === Lift analysis for semantic ===
        if 'quality_semantic' in self.df.columns:
            # Determine which rt_output to use - semantic when stratifying by semantic
            rt_out_sem = 'rt_output_semantic' if 'semantic' in stratify_by and 'rt_output_semantic' in self.df.columns else 'rt_output'
            if rt_out_sem in self.df.columns:
                results['lift_analysis_semantic'] = self._lift_analysis(
                    stratify_by, rt_out_sem, 'quality_semantic', q1, q3
                )
        
        return results
    
    def _cross_tabulation(self, stratify_by: str, rt_output_col: str, 
                          quality_col: str, q1: float, q3: float) -> dict:
        """Compute cross-tabulation: stratify_by stratum × rt_output level → % good quality."""
        if rt_output_col not in self.df.columns or quality_col not in self.df.columns:
            return {'error': 'Required columns not found'}
        
        rt_output_median = float(np.median(self.df[rt_output_col].dropna()))
        quality_median = float(np.median(self.df[quality_col].dropna()))
        
        low_src_q1 = self.df[stratify_by] <= q1
        high_src_q3 = self.df[stratify_by] >= q3
        low_rt_out = self.df[rt_output_col] <= rt_output_median
        high_rt_out = self.df[rt_output_col] > rt_output_median
        
        results = {
            'rt_output_median': rt_output_median,
            'quality_median': quality_median
        }
        
        for src_name, src_mask in [('low_src_q1', low_src_q1), ('high_src_q3', high_src_q3)]:
            for out_name, out_mask in [('low_rt_out', low_rt_out), ('high_rt_out', high_rt_out)]:
                combined = src_mask & out_mask
                n = int(combined.sum())
                if n > 0:
                    n_good = int((self.df.loc[combined, quality_col] > quality_median).sum())
                    pct_good = 100 * n_good / n
                else:
                    n_good = 0
                    pct_good = 0
                
                results[f'{src_name}_{out_name}'] = {
                    'n': n,
                    'n_good': n_good,
                    'pct_good': float(pct_good)
                }
        
        return results
    
    def _four_way_breakdown(self, stratify_by: str, rt_output_col: str,
                            quality_col: str, q1: float, q3: float) -> dict:
        """Compute four-way breakdown: good/bad × stable/unstable for each stratum."""
        if rt_output_col not in self.df.columns or quality_col not in self.df.columns:
            return {'error': 'Required columns not found'}
        
        rt_output_median = float(np.median(self.df[rt_output_col].dropna()))
        quality_median = float(np.median(self.df[quality_col].dropna()))
        
        results = {}
        
        for stratum_name, stratum_mask in [('low_stratum_q1', self.df[stratify_by] <= q1),
                                            ('high_stratum_q3', self.df[stratify_by] >= q3)]:
            stratum_df = self.df[stratum_mask]
            n_total = len(stratum_df)
            
            if n_total == 0:
                results[stratum_name] = {'n_total': 0}
                continue
            
            good = stratum_df[quality_col] > quality_median
            stable = stratum_df[rt_output_col] > rt_output_median
            
            good_stable = good & stable
            bad_stable = ~good & stable
            good_unstable = good & ~stable
            bad_unstable = ~good & ~stable
            
            results[stratum_name] = {
                'n_total': n_total,
                'good_and_stable': {'n': int(good_stable.sum()), 'pct': float(100 * good_stable.sum() / n_total)},
                'bad_but_stable': {'n': int(bad_stable.sum()), 'pct': float(100 * bad_stable.sum() / n_total)},
                'good_but_unstable': {'n': int(good_unstable.sum()), 'pct': float(100 * good_unstable.sum() / n_total)},
                'bad_and_unstable': {'n': int(bad_unstable.sum()), 'pct': float(100 * bad_unstable.sum() / n_total)}
            }
        
        return results
    
    def _lift_analysis(self, stratify_by: str, rt_output_col: str,
                       quality_col: str, q1: float, q3: float) -> dict:
        """Compute lift from high rt_output in each stratum."""
        if rt_output_col not in self.df.columns or quality_col not in self.df.columns:
            return {'error': 'Required columns not found'}
        
        quality_median = float(np.median(self.df[quality_col].dropna()))
        
        results = {}
        
        for stratum_name, stratum_mask in [('low_stratum_q1', self.df[stratify_by] <= q1),
                                            ('high_stratum_q3', self.df[stratify_by] >= q3)]:
            stratum_df = self.df[stratum_mask]
            
            if len(stratum_df) < 5:
                results[stratum_name] = {'error': 'Insufficient samples'}
                continue
            
            # Compute rt_output median within stratum
            rt_out_median_within = float(np.median(stratum_df[rt_output_col].dropna()))
            
            low_rt_out = stratum_df[rt_output_col] <= rt_out_median_within
            high_rt_out = stratum_df[rt_output_col] > rt_out_median_within
            
            pct_good_low = 100 * (stratum_df.loc[low_rt_out, quality_col] > quality_median).mean() if low_rt_out.sum() > 0 else 0
            pct_good_high = 100 * (stratum_df.loc[high_rt_out, quality_col] > quality_median).mean() if high_rt_out.sum() > 0 else 0
            
            lift_pp = pct_good_high - pct_good_low
            lift_ratio = pct_good_high / pct_good_low if pct_good_low > 0 else float('inf')
            
            results[stratum_name] = {
                'rt_output_median_within': rt_out_median_within,
                'pct_good_when_low_rt_out': float(pct_good_low),
                'pct_good_when_high_rt_out': float(pct_good_high),
                'lift_pp': float(lift_pp),
                'lift_ratio': float(lift_ratio) if lift_ratio != float('inf') else None
            }
        
        return results


# =============================================================================
# Investigation Analysis (Extreme Cases, Characterization, Embedding Alignment)
# =============================================================================

class InvestigationAnalyzer:
    """Analyzes extreme cases and patterns in RT-quality relationships."""
    
    def __init__(self, df: pd.DataFrame, logger: Optional[logging.Logger] = None):
        self.df = df
        self.logger = logger or logging.getLogger(__name__)
    
    def find_extreme_cases(self, rt_metric: str, quality_metric: str, 
                           stratum: str = 'high', n_cases: int = 5) -> dict:
        """
        Find extreme cases where RT and quality diverge.
        """
        # Get relevant stratum
        median = self.df[quality_metric].median()
        if stratum == 'high':
            stratum_df = self.df[self.df[quality_metric] >= median].copy()
        else:
            stratum_df = self.df[self.df[quality_metric] < median].copy()
        
        if len(stratum_df) == 0:
            return {'error': 'No data in stratum'}
        
        # Normalize metrics for comparison
        rt_std = stratum_df[rt_metric].std()
        quality_std = stratum_df[quality_metric].std()
        
        if rt_std == 0 or quality_std == 0:
            return {'error': 'No variance in metrics'}
        
        rt_norm = (stratum_df[rt_metric] - stratum_df[rt_metric].mean()) / rt_std
        quality_norm = (stratum_df[quality_metric] - stratum_df[quality_metric].mean()) / quality_std
        
        # Divergence score: high RT, low quality = positive; low RT, high quality = negative
        stratum_df['divergence'] = rt_norm - quality_norm
        
        # High RT, Low Quality cases
        high_rt_low_quality = stratum_df.nlargest(n_cases, 'divergence')
        
        # Low RT, High Quality cases
        low_rt_high_quality = stratum_df.nsmallest(n_cases, 'divergence')
        
        def extract_case_info(row):
            return {
                'sentence_id': int(row.get('sentence_id', -1)),
                'pivot_lang': row.get('pivot_lang', ''),
                'source_text': row.get('source_text', '')[:100] + '...' if len(row.get('source_text', '')) > 100 else row.get('source_text', ''),
                'output_text': row.get('output_text', '')[:100] + '...' if len(row.get('output_text', '')) > 100 else row.get('output_text', ''),
                'target_reference': row.get('target_reference', '')[:100] + '...' if len(row.get('target_reference', '')) > 100 else row.get('target_reference', ''),
                rt_metric: float(row[rt_metric]),
                quality_metric: float(row[quality_metric]),
                'divergence': float(row['divergence']),
            }
        
        return {
            'stratum': stratum,
            'median_threshold': float(median),
            'n_in_stratum': len(stratum_df),
            'high_rt_low_quality': [extract_case_info(row) for _, row in high_rt_low_quality.iterrows()],
            'low_rt_high_quality': [extract_case_info(row) for _, row in low_rt_high_quality.iterrows()],
        }
    
    def automatic_characterization(self, df_subset: pd.DataFrame = None) -> dict:
        """
        Compute automatic text statistics for characterizing translations.
        """
        if df_subset is None:
            df_subset = self.df
        
        results = {
            'n_samples': len(df_subset),
            'statistics': []
        }
        
        for idx, row in df_subset.iterrows():
            output = row.get('output_text', '')
            reference = row.get('target_reference', '')
            
            if not output or not reference:
                continue
            
            # Word-level analysis
            output_words = set(output.lower().split())
            ref_words = set(reference.lower().split())
            
            # Word overlap
            overlap = output_words & ref_words
            word_overlap_ratio = len(overlap) / len(ref_words) if ref_words else 0
            
            # Jaccard similarity
            union = output_words | ref_words
            jaccard = len(overlap) / len(union) if union else 0
            
            # Length ratio
            len_ratio = len(output.split()) / len(reference.split()) if reference.split() else 1
            
            results['statistics'].append({
                'sentence_id': row.get('sentence_id', idx),
                'pivot_lang': row.get('pivot_lang', ''),
                'word_overlap_ratio': float(word_overlap_ratio),
                'jaccard_similarity': float(jaccard),
                'length_ratio': float(len_ratio),
                'output_unique_words': len(output_words),
                'reference_unique_words': len(ref_words),
                'common_words': len(overlap),
            })
        
        return results
    
    def embedding_word_alignment(self, row: dict, embedding_model: EmbeddingModel) -> dict:
        """
        Compute word-level alignment using embeddings.
        """
        output = row.get('output_text', '')
        reference = row.get('target_reference', '')
        tgt_lang = row.get('target_lang', 'jpn')
        
        if not output or not reference:
            return {'error': 'Missing text'}
        
        # Split into words
        output_words = output.split()
        ref_words = reference.split()
        
        if not output_words or not ref_words:
            return {'error': 'No words after splitting'}
        
        # Get embeddings for each word
        try:
            output_embeds = [embedding_model.embed(w, tgt_lang) for w in output_words[:20]]
            ref_embeds = [embedding_model.embed(w, tgt_lang) for w in ref_words[:20]]
        except Exception as e:
            return {'error': f'Embedding error: {str(e)}'}
        
        # For each output word, find closest reference word
        alignments = []
        for i, (ow, oe) in enumerate(zip(output_words[:20], output_embeds)):
            best_sim = -1
            best_match = None
            for j, (rw, re) in enumerate(zip(ref_words[:20], ref_embeds)):
                sim = cosine_similarity(oe, re)
                if sim > best_sim:
                    best_sim = sim
                    best_match = rw
            
            # Classify match
            if ow == best_match:
                match_type = 'exact'
            elif best_sim > 0.8:
                match_type = 'close'
            else:
                match_type = 'different'
            
            alignments.append({
                'output_word': ow,
                'best_match': best_match,
                'similarity': float(best_sim),
                'match_type': match_type,
            })
        
        # Aggregate statistics
        n_exact = sum(1 for a in alignments if a['match_type'] == 'exact')
        n_close = sum(1 for a in alignments if a['match_type'] == 'close')
        n_different = sum(1 for a in alignments if a['match_type'] == 'different')
        
        return {
            'n_output_words': len(output_words[:20]),
            'n_ref_words': len(ref_words[:20]),
            'pct_exact_match': float(n_exact / len(alignments)) if alignments else 0,
            'pct_close_match': float(n_close / len(alignments)) if alignments else 0,
            'pct_different': float(n_different / len(alignments)) if alignments else 0,
            'mean_alignment_similarity': float(np.mean([a['similarity'] for a in alignments])) if alignments else 0,
            'alignments': alignments[:10],  # Only keep first 10 for output
        }
    
    def full_investigation(self, embedding_model: EmbeddingModel = None) -> dict:
        """Run all investigation analyses."""
        self.logger.info("  Finding extreme cases...")
        
        results = {}
        
        # Surface RT vs chrF
        results['extreme_cases_surface_rt_vs_chrf_high'] = self.find_extreme_cases(
            'rt_geometric', 'quality_chrf', stratum='high'
        )
        
        # Semantic RT vs chrF (if available)
        if 'rt_geometric_semantic' in self.df.columns:
            results['extreme_cases_semantic_rt_vs_chrf_high'] = self.find_extreme_cases(
                'rt_geometric_semantic', 'quality_chrf', stratum='high'
            )
        
        # Semantic RT vs Semantic Quality
        if 'rt_geometric_semantic' in self.df.columns and 'quality_semantic' in self.df.columns:
            results['extreme_cases_semantic_rt_vs_semantic_quality_high'] = self.find_extreme_cases(
                'rt_geometric_semantic', 'quality_semantic', stratum='high'
            )
        
        # Semantic RT vs COMET Quality (if available)
        if 'rt_geometric_semantic' in self.df.columns and 'quality_comet' in self.df.columns:
            results['extreme_cases_semantic_rt_vs_comet_quality_high'] = self.find_extreme_cases(
                'rt_geometric_semantic', 'quality_comet', stratum='high'
            )
        
        # Automatic characterization
        self.logger.info("  Running automatic characterization...")
        results['characterization_high_stratum'] = self.automatic_characterization()
        
        # Embedding word alignment on extreme cases (if model provided)
        if embedding_model is not None:
            self.logger.info("  Running embedding word alignment on extreme cases...")
            
            extreme_cases = results.get('extreme_cases_surface_rt_vs_chrf_high', {})
            high_rt_cases = extreme_cases.get('high_rt_low_quality', [])
            
            if high_rt_cases:
                alignment_results = []
                for case in high_rt_cases[:3]:  # Only first 3 to save time
                    # Find the original row
                    mask = (self.df['sentence_id'] == case['sentence_id']) & \
                           (self.df['pivot_lang'] == case['pivot_lang'])
                    if mask.any():
                        row = self.df[mask].iloc[0].to_dict()
                        alignment = self.embedding_word_alignment(row, embedding_model)
                        alignment['sentence_id'] = case['sentence_id']
                        alignment['pivot_lang'] = case['pivot_lang']
                        alignment_results.append(alignment)
                
                results['alignment_high_rt_low_chrf'] = alignment_results
        
        return results


# =============================================================================
# Metrics Summary Computer
# =============================================================================

def compute_metrics_summary(df: pd.DataFrame) -> dict:
    """Compute summary statistics for all metrics."""
    
    # Surface RT metrics
    surface_rt = ['rt_source', 'rt_hop2', 'rt_output', 'rt_target_ref', 
                  'rt_min', 'rt_geometric', 'rt_min_st', 'rt_geometric_st']
    
    # Semantic RT metrics
    semantic_rt = ['rt_source_semantic', 'rt_hop2_semantic', 'rt_output_semantic',
                   'rt_target_ref_semantic', 'rt_min_semantic', 'rt_geometric_semantic',
                   'rt_min_semantic_st', 'rt_geometric_semantic_st']
    
    # Quality metrics (BLEU removed, COMET added)
    quality_metrics = ['quality_chrf', 'quality_semantic', 'quality_comet',
                       'direct_source_output_labse', 'direct_source_output_comet']
    
    # Combined metrics
    combined_metrics = ['combined_rt_direct_geometric', 'combined_rt_direct_mean', 'combined_rt_direct_min']
    
    results = {
        'n_samples': len(df),
        'quality_metrics': {},
        'surface_rt_metrics': {},
        'semantic_rt_metrics': {},
        'combined_metrics': {},
    }
    
    # Quality metrics summary
    for metric in quality_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            results['quality_metrics'][metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median()),
            }
    
    # Surface RT metrics summary
    for metric in surface_rt:
        if metric in df.columns:
            values = df[metric].dropna()
            results['surface_rt_metrics'][metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
            }
    
    # Semantic RT metrics summary
    for metric in semantic_rt:
        if metric in df.columns:
            values = df[metric].dropna()
            results['semantic_rt_metrics'][metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
            }
    
    # Combined metrics summary
    for metric in combined_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            results['combined_metrics'][metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
            }
    
    return results


def compute_full_correlation_matrix(df: pd.DataFrame) -> dict:
    """Compute correlations between all RT metrics and all quality metrics."""
    if not SCIPY_AVAILABLE:
        return {'error': 'scipy not available'}
    
    surface_rt = ['rt_source', 'rt_hop2', 'rt_output', 'rt_target_ref', 
                  'rt_min', 'rt_geometric', 'rt_min_st', 'rt_geometric_st']
    semantic_rt = ['rt_source_semantic', 'rt_hop2_semantic', 'rt_output_semantic',
                   'rt_target_ref_semantic', 'rt_min_semantic', 'rt_geometric_semantic',
                   'rt_min_semantic_st', 'rt_geometric_semantic_st']
    # Include direct_source_output variants and combined metrics as predictors
    additional_predictors = ['direct_source_output_labse',
                            'direct_source_output_comet',
                            'combined_rt_direct_geometric', 
                            'combined_rt_direct_mean', 
                            'combined_rt_direct_min']
    quality_metrics = ['quality_chrf', 'quality_semantic', 'quality_comet']
    
    # Filter to available columns
    surface_rt = [c for c in surface_rt if c in df.columns]
    semantic_rt = [c for c in semantic_rt if c in df.columns]
    additional_predictors = [c for c in additional_predictors if c in df.columns]
    quality_metrics = [c for c in quality_metrics if c in df.columns]
    
    all_rt = surface_rt + semantic_rt + additional_predictors
    
    results = {}
    
    skipped_constant = []
    
    for rt in all_rt:
        for quality in quality_metrics:
            try:
                valid_mask = df[rt].notna() & df[quality].notna()
                if valid_mask.sum() > 2:
                    rt_std = df.loc[valid_mask, rt].std()
                    quality_std = df.loc[valid_mask, quality].std()
                    
                    if rt_std < 1e-10:
                        results[f'{rt}_vs_{quality}'] = {
                            'pearson_r': float('nan'),
                            'p_value': float('nan'),
                            'note': 'RT metric is constant'
                        }
                        skipped_constant.append(f"{rt}:{quality}(RT constant)")
                    elif quality_std < 1e-10:
                        results[f'{rt}_vs_{quality}'] = {
                            'pearson_r': float('nan'),
                            'p_value': float('nan'),
                            'note': 'Quality metric is constant'
                        }
                        skipped_constant.append(f"{rt}:{quality}(quality constant)")
                    else:
                        r, p = stats.pearsonr(df.loc[valid_mask, rt], df.loc[valid_mask, quality])
                        results[f'{rt}_vs_{quality}'] = {
                            'pearson_r': float(r),
                            'p_value': float(p),
                            'significant': str(p < 0.05)
                        }
                else:
                    results[f'{rt}_vs_{quality}'] = {'error': 'Insufficient data'}
            except Exception as e:
                results[f'{rt}_vs_{quality}'] = {'error': str(e)}
    
    if skipped_constant:
        print(f"  [!] Skipped {len(skipped_constant)} correlations due to constant input")
    
    return results


# =============================================================================
# Per-Pivot Analysis
# =============================================================================

class PerPivotAnalyzer:
    """Runs all analyses decomposed by pivot language."""
    
    def __init__(self, df: pd.DataFrame, embedding_model: EmbeddingModel = None,
                 logger: Optional[logging.Logger] = None):
        self.df = df
        self.embedding_model = embedding_model
        self.logger = logger or logging.getLogger(__name__)
        self.pivots = df['pivot_lang'].unique().tolist() if 'pivot_lang' in df.columns else []
    
    def analyze_all_pivots(self, bootstrap_n: int = 500, include_investigation: bool = True) -> dict:
        """
        Run full analysis for each pivot language.
        
        Args:
            bootstrap_n: Number of bootstrap samples (reduced for per-pivot to save time)
            include_investigation: Whether to include investigation analysis (extreme cases)
        
        Returns:
            Dict with results for each pivot
        """
        if not self.pivots:
            return {'error': 'No pivot_lang column found'}
        
        results = {}
        
        for pivot in self.pivots:
            self.logger.info(f"    Analyzing pivot: {pivot}")
            pivot_df = self.df[self.df['pivot_lang'] == pivot].copy()
            
            if len(pivot_df) < 10:
                results[pivot] = {'error': f'Insufficient samples: {len(pivot_df)}'}
                continue
            
            # Metrics summary (includes all RT metrics - surface and semantic)
            metrics_summary = compute_metrics_summary(pivot_df)
            
            # Correlation matrix
            correlation_matrix = compute_full_correlation_matrix(pivot_df)
            
            # Hypothesis A analysis (ROC, stratified correlations, etc.)
            hyp_a = HypothesisAAnalyzer(pivot_df, self.logger)
            hypothesis_a = hyp_a.full_analysis(bootstrap_n=bootstrap_n)
            
            pivot_result = {
                'n_samples': len(pivot_df),
                'metrics_summary': metrics_summary,
                'correlation_matrix': correlation_matrix,
                'hypothesis_a': hypothesis_a,
            }
            
            # Investigation analysis (extreme cases, characterization)
            if include_investigation and len(pivot_df) >= 20:
                investigator = InvestigationAnalyzer(pivot_df, self.logger)
                # Don't do word alignment per-pivot to save time
                pivot_result['investigation'] = investigator.full_investigation(
                    embedding_model=None  # Skip word alignment per pivot
                )
            
            results[pivot] = pivot_result
        
        # Add cross-pivot comparison summary
        results['_pivot_comparison'] = self._compare_across_pivots(results)
        
        return results
    
    def _compare_across_pivots(self, pivot_results: dict) -> dict:
        """
        Compare key metrics across pivot languages.
        
        Extracts and compares:
        - ROC AUC for failure detection
        - RT-quality correlations
        - Smoke detector pattern strength (Δ)
        - Failure rates
        - Granularity (within-band correlations)
        """
        comparison = {
            'pivots': [],
            'n_samples': {},
            'failure_rates': {},
            'roc_auc': {},
            'correlations': {},
            'smoke_detector_delta': {},
            'granularity': {},
            'rankings': {},
        }
        
        # Key metrics to extract
        rt_metric = 'rt_geometric_semantic_st'
        quality_metric = 'quality_semantic'
        
        for pivot, results in pivot_results.items():
            if pivot.startswith('_') or 'error' in results:
                continue
            
            comparison['pivots'].append(pivot)
            comparison['n_samples'][pivot] = results.get('n_samples', 0)
            
            # 1. Failure rate (% below threshold)
            metrics = results.get('metrics_summary', {})
            quality_data = metrics.get('quality_metrics', {}).get(quality_metric, {})
            # Estimate failure rate from mean/std (rough approximation)
            # Better: would need to compute from raw data
            comparison['failure_rates'][pivot] = {
                'mean_quality': quality_data.get('mean'),
                'std_quality': quality_data.get('std'),
            }
            
            # 2. ROC AUC for semantic failure detection (threshold 0.85)
            hyp_a = results.get('hypothesis_a', {})
            roc = hyp_a.get('roc_analysis', {}).get(quality_metric, {})
            rt_roc = roc.get(rt_metric, {})
            auc_085 = rt_roc.get('threshold_0.85', {}).get('auc')
            auc_090 = rt_roc.get('threshold_0.9', {}).get('auc')
            comparison['roc_auc'][pivot] = {
                'auc_0.85': auc_085,
                'auc_0.90': auc_090,
            }
            
            # 3. RT-quality correlation
            corr_matrix = results.get('correlation_matrix', {})
            corr_key = f'{rt_metric}_vs_{quality_metric}'
            corr_data = corr_matrix.get(corr_key, {})
            comparison['correlations'][pivot] = {
                'pearson_r': corr_data.get('pearson_r'),
            }
            
            # 4. Smoke detector pattern (Δ = low stratum - high stratum correlation)
            strat_corr = hyp_a.get('stratified_correlations', {})
            strat_by_quality = strat_corr.get(f'stratified_by_{quality_metric}', {})
            semantic_strat = strat_by_quality.get('semantic', {})
            
            low_r = None
            high_r = None
            if 'low_stratum' in semantic_strat and 'high_stratum' in semantic_strat:
                low_corrs = semantic_strat['low_stratum'].get('correlations', {})
                high_corrs = semantic_strat['high_stratum'].get('correlations', {})
                
                if rt_metric in low_corrs and quality_metric in low_corrs[rt_metric]:
                    low_r = low_corrs[rt_metric][quality_metric].get('r')
                if rt_metric in high_corrs and quality_metric in high_corrs[rt_metric]:
                    high_r = high_corrs[rt_metric][quality_metric].get('r')
            
            delta = None
            if low_r is not None and high_r is not None:
                delta = low_r - high_r
            
            comparison['smoke_detector_delta'][pivot] = {
                'low_stratum_r': low_r,
                'high_stratum_r': high_r,
                'delta': delta,
            }
            
            # 5. Granularity (mean correlation in top quartile)
            gran = hyp_a.get('granularity_analysis', {})
            gran_quartiles = gran.get(f'{quality_metric}_quartiles', {})
            top_quartile = gran_quartiles.get('bands', {}).get('band_4_of_4', {})
            top_q_corr = top_quartile.get('correlations', {}).get(rt_metric, {})
            
            comparison['granularity'][pivot] = {
                'top_quartile_spearman': top_q_corr.get('spearman_rho'),
                'top_quartile_n': top_q_corr.get('n_valid'),
            }
        
        # Generate rankings
        pivots = comparison['pivots']
        
        # Rank by ROC AUC (higher is better)
        auc_values = [(p, comparison['roc_auc'].get(p, {}).get('auc_0.85')) 
                      for p in pivots]
        auc_values = [(p, v) for p, v in auc_values if v is not None]
        auc_ranking = sorted(auc_values, key=lambda x: x[1], reverse=True)
        comparison['rankings']['by_roc_auc_0.85'] = [p for p, _ in auc_ranking]
        
        # Rank by correlation (higher absolute value is better)
        corr_values = [(p, comparison['correlations'].get(p, {}).get('pearson_r')) 
                       for p in pivots]
        corr_values = [(p, v) for p, v in corr_values if v is not None]
        corr_ranking = sorted(corr_values, key=lambda x: abs(x[1]), reverse=True)
        comparison['rankings']['by_correlation'] = [p for p, _ in corr_ranking]
        
        # Rank by smoke detector delta (higher is stronger smoke detector)
        delta_values = [(p, comparison['smoke_detector_delta'].get(p, {}).get('delta')) 
                        for p in pivots]
        delta_values = [(p, v) for p, v in delta_values if v is not None]
        delta_ranking = sorted(delta_values, key=lambda x: x[1], reverse=True)
        comparison['rankings']['by_smoke_detector'] = [p for p, _ in delta_ranking]
        
        # Summary: best pivot for each criterion
        comparison['best_pivot'] = {
            'roc_auc': auc_ranking[0][0] if auc_ranking else None,
            'correlation': corr_ranking[0][0] if corr_ranking else None,
            'smoke_detector': delta_ranking[0][0] if delta_ranking else None,
        }
        
        return comparison


# =============================================================================
# Main Analysis Class
# =============================================================================

class SemanticAnalyzer:
    """Main class for running semantic analysis on RT results."""
    
    def __init__(self, model_name: str = 'auto', logger: Optional[logging.Logger] = None,
                 comet_path: Optional[str] = None,
                 comet_qe_path: Optional[str] = None,
                 comet_batch_size: int = 8):
        """
        Initialize the analyzer.
        
        Args:
            model_name: 'labse', 'sonar', 'both', or 'auto' (uses best available)
            comet_path: Path to COMET model for quality_comet (optional)
            comet_qe_path: Path to COMET-QE model for direct_source_output_comet (optional)
            comet_batch_size: Batch size for COMET processing
        """
        self.logger = logger or self._setup_logger()
        self.model_name = model_name
        self.models = {}
        self.comet_model = None
        self.comet_qe_model = None
        self.comet_batch_size = comet_batch_size
        
        # Suppress verbose logging from COMET dependencies BEFORE loading
        if comet_path or comet_qe_path:
            self._suppress_comet_logging()
        
        # Load COMET model if path provided
        if comet_path:
            self.logger.info(f"Initializing COMET model from: {comet_path}")
            self.comet_model = COMETModel(comet_path)
        
        # Load COMET-QE model if path provided
        if comet_qe_path:
            self.logger.info(f"Initializing COMET-QE model from: {comet_qe_path}")
            self.comet_qe_model = COMETQEModel(comet_qe_path)
        
        # Determine which embedding models to use
        available = get_available_models()
        self.logger.info(f"Available embedding models: {available}")
        
        if model_name == 'auto':
            # Prefer SONAR, fall back to LaBSE
            if available['sonar']:
                self.models['sonar'] = load_embedding_model('sonar')
            elif available['labse']:
                self.models['labse'] = load_embedding_model('labse')
            else:
                raise RuntimeError("No embedding models available. Run setup_embeddings.py first.")
        elif model_name == 'both':
            if available['labse']:
                self.models['labse'] = load_embedding_model('labse')
            if available['sonar']:
                self.models['sonar'] = load_embedding_model('sonar')
            if not self.models:
                raise RuntimeError("No embedding models available. Run setup_embeddings.py first.")
        else:
            if not available.get(model_name, False):
                raise RuntimeError(f"Model {model_name} not available. Run setup_embeddings.py first.")
            self.models[model_name] = load_embedding_model(model_name)
        
        self.logger.info(f"Using models: {list(self.models.keys())}")
    
    def _suppress_comet_logging(self):
        """Suppress verbose logging from COMET and its dependencies."""
        _suppress_all_comet_output()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up basic logging."""
        logger = logging.getLogger("semantic_analysis")
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate messages - don't propagate to root logger
        # (PyTorch Lightning adds handlers to root logger which causes duplicates)
        logger.propagate = False
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
        
        return logger
    
    def analyze(self, results_file: Path, output_dir: Optional[Path] = None,
                per_pivot: bool = True, bootstrap_n: int = 1000,
                include_pivots: Optional[List[str]] = None,
                exclude_pivots: Optional[List[str]] = None) -> dict:
        """
        Run full semantic analysis on results file.
        
        Args:
            results_file: Path to JSONL results file
            output_dir: Directory for output (defaults to same as results_file)
            per_pivot: Whether to run per-pivot analysis (default: True)
            bootstrap_n: Number of bootstrap samples for CI
            include_pivots: If provided, only analyze these pivot languages
            exclude_pivots: If provided, exclude these pivot languages
        
        Returns:
            Analysis report dictionary
        """
        results_file = Path(results_file)
        output_dir = Path(output_dir) if output_dir else results_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.logger.info(f"\nLoading results from: {results_file}")
        df = self._load_results(results_file)
        self.logger.info(f"Loaded {len(df)} results")
        
        # Get pivot languages before filtering
        all_pivots = df['pivot_lang'].unique().tolist() if 'pivot_lang' in df.columns else []
        self.logger.info(f"Pivot languages in file: {all_pivots}")
        
        # Apply pivot filtering
        if include_pivots or exclude_pivots:
            original_len = len(df)
            
            if include_pivots:
                df = df[df['pivot_lang'].isin(include_pivots)]
                self.logger.info(f"Filtering to pivots: {include_pivots}")
            
            if exclude_pivots:
                df = df[~df['pivot_lang'].isin(exclude_pivots)]
                self.logger.info(f"Excluding pivots: {exclude_pivots}")
            
            filtered_pivots = df['pivot_lang'].unique().tolist() if 'pivot_lang' in df.columns else []
            self.logger.info(f"After filtering: {len(df)} results ({original_len - len(df)} removed)")
            self.logger.info(f"Analyzing pivots: {filtered_pivots}")
        
        # Get pivot languages after filtering
        pivots = df['pivot_lang'].unique().tolist() if 'pivot_lang' in df.columns else []
        
        # Create filename suffix for filtered results
        pivot_filter_suffix = ""
        if include_pivots:
            pivot_filter_suffix = f"_pivots_{'_'.join(sorted(include_pivots))}"
        elif exclude_pivots:
            pivot_filter_suffix = f"_excl_{'_'.join(sorted(exclude_pivots))}"
        
        # Run analysis for each model
        all_reports = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"ANALYZING WITH {model_name.upper()}")
            self.logger.info('='*60)
            
            report = self._analyze_with_model(df.copy(), model, per_pivot=per_pivot,
                                              bootstrap_n=bootstrap_n,
                                              output_dir=output_dir,
                                              results_file_stem=results_file.stem + pivot_filter_suffix,
                                              model_name=model_name)
            all_reports[model_name] = report
        
        # Combine reports with pivot filter metadata
        final_report = self._combine_reports(all_reports, results_file, 
                                             include_pivots=include_pivots,
                                             exclude_pivots=exclude_pivots,
                                             analyzed_pivots=pivots)
        
        # Save report (with pivot filter suffix in filename)
        report_file = output_dir / f"{results_file.stem}{pivot_filter_suffix}_semantic_analysis.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Analysis complete!")
        self.logger.info(f"Report saved to: {report_file}")
        self.logger.info(f"Per-sample CSV files saved for each model (for scatter plots, etc.)")
        
        # Print extreme cases to console
        self._print_extreme_cases(final_report)
        
        return final_report
    
    def _load_results(self, results_file: Path) -> pd.DataFrame:
        """Load results from JSONL file."""
        results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return pd.DataFrame(results)
    
    def _analyze_with_model(self, df: pd.DataFrame, model: EmbeddingModel,
                            per_pivot: bool = True, bootstrap_n: int = 1000,
                            output_dir: Path = None, results_file_stem: str = None,
                            model_name: str = 'unknown') -> dict:
        """Run analysis using a specific embedding model."""
        # Compute semantic metrics
        computer = SemanticMetricsComputer(
            model, self.logger,
            comet_model=self.comet_model,
            comet_qe_model=self.comet_qe_model,
            comet_batch_size=self.comet_batch_size
        )
        df = computer.compute_for_dataframe(df)
        
        # Save per-sample data to CSV for post-hoc analysis (scatter plots, etc.)
        if output_dir and results_file_stem:
            samples_file = output_dir / f"{results_file_stem}_{model_name}_samples.csv"
            df.to_csv(samples_file, index=False)
            self.logger.info(f"Per-sample data saved to: {samples_file}")
        
        # === Combined Analysis (all pivots) ===
        self.logger.info("\n--- Combined Analysis (all pivots) ---")
        
        # Metrics summary
        metrics_summary = compute_metrics_summary(df)
        
        # Legacy semantic summary format (for backward compatibility)
        semantic_summary = {
            'quality_semantic': {
                'mean': float(df['quality_semantic'].mean()),
                'std': float(df['quality_semantic'].std()),
                'min': float(df['quality_semantic'].min()),
                'max': float(df['quality_semantic'].max()),
            },
            'quality_magnitude_ratio': {
                'mean': float(df['quality_magnitude_ratio'].mean()),
                'std': float(df['quality_magnitude_ratio'].std()),
            }
        }
        
        for col in ['rt_source_semantic', 'rt_hop2_semantic', 'rt_output_semantic',
                    'rt_target_ref_semantic', 'rt_min_semantic', 'rt_geometric_semantic',
                    'rt_min_semantic_st', 'rt_geometric_semantic_st', 'direct_source_output',
                    'combined_rt_direct_geometric', 'combined_rt_direct_mean', 'combined_rt_direct_min']:
            if col in df.columns:
                semantic_summary[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                }
        
        # Hypothesis A analysis
        self.logger.info("\nRunning Hypothesis A analysis (failure detection)...")
        hyp_a = HypothesisAAnalyzer(df, self.logger)
        hypothesis_a_results = hyp_a.full_analysis(bootstrap_n=bootstrap_n)
        
        # Correlation matrix
        self.logger.info("Computing correlation matrix...")
        correlation_matrix = compute_full_correlation_matrix(df)
        
        # Investigation analysis
        self.logger.info("\nRunning investigation analysis...")
        investigator = InvestigationAnalyzer(df, self.logger)
        investigation_results = investigator.full_investigation(embedding_model=model)
        
        # Key findings
        key_findings = self._extract_key_findings(hypothesis_a_results, correlation_matrix)
        
        report = {
            'model': model.name,
            'embedding_dimension': model.dimension,
            'semantic_metrics_summary': semantic_summary,
            'metrics_summary': metrics_summary,
            'hypothesis_a': hypothesis_a_results,
            'correlation_matrix': correlation_matrix,
            'investigation': investigation_results,
            'key_findings': key_findings,
        }
        
        # === Per-Pivot Analysis ===
        if per_pivot:
            self.logger.info("\n--- Per-Pivot Analysis ---")
            pivot_analyzer = PerPivotAnalyzer(df, embedding_model=model, logger=self.logger)
            report['per_pivot_analysis'] = pivot_analyzer.analyze_all_pivots(
                bootstrap_n=max(100, bootstrap_n // 2)  # Reduce for per-pivot
            )
        
        return report
    
    def _extract_key_findings(self, hyp_a: dict, correlations: dict) -> List[str]:
        """Extract key findings from analysis."""
        findings = []
        
        # Check ROC results across all quality metrics
        roc_results = hyp_a.get('roc_analysis', {})
        for quality_metric, rt_results in roc_results.items():
            for rt_metric, thresholds in rt_results.items():
                for threshold_key, data in thresholds.items():
                    if isinstance(data, dict) and 'auc' in data and data['auc'] is not None:
                        auc = data['auc']
                        if auc > 0.7:
                            findings.append(f"{rt_metric} shows good failure detection for {quality_metric} (AUC={auc:.2f} at {threshold_key})")
        
        # Check if semantic RT correlates better than surface RT with COMET
        surface_comet = correlations.get('rt_min_vs_quality_comet', {}).get('pearson_r', 0)
        semantic_comet = correlations.get('rt_min_semantic_vs_quality_comet', {}).get('pearson_r', 0)
        
        if semantic_comet and surface_comet:
            if abs(semantic_comet) > abs(surface_comet) + 0.1:
                findings.append(f"Semantic RT correlates better with COMET (r={semantic_comet:.2f}) than surface RT (r={surface_comet:.2f})")
        
        # Check if COMET-QE direct outperforms LaBSE direct
        labse_direct = correlations.get('direct_source_output_labse_vs_quality_comet', {}).get('pearson_r', 0)
        comet_direct = correlations.get('direct_source_output_comet_vs_quality_comet', {}).get('pearson_r', 0)
        
        if comet_direct and labse_direct:
            if abs(comet_direct) > abs(labse_direct) + 0.05:
                findings.append(f"COMET-QE direct (r={comet_direct:.2f}) outperforms LaBSE direct (r={labse_direct:.2f}) for quality prediction")
        
        # Check if combined metrics outperform individual components
        combined = correlations.get('combined_rt_direct_mean_vs_quality_comet', {}).get('pearson_r', 0)
        rt_only = correlations.get('rt_geometric_semantic_st_vs_quality_comet', {}).get('pearson_r', 0)
        
        if combined and rt_only and comet_direct:
            if abs(combined) > max(abs(rt_only), abs(comet_direct)) + 0.02:
                findings.append(f"Combined metric (r={combined:.2f}) outperforms RT alone (r={rt_only:.2f}) and COMET-QE alone (r={comet_direct:.2f})")
        
        # Limit findings
        if len(findings) > 10:
            findings = findings[:10]
            findings.append("... (additional findings truncated)")
        
        if not findings:
            findings.append("No strong patterns found in this dataset")
        
        return findings
    
    def _combine_reports(self, reports: Dict[str, dict], source_file: Path,
                         include_pivots: Optional[List[str]] = None,
                         exclude_pivots: Optional[List[str]] = None,
                         analyzed_pivots: Optional[List[str]] = None) -> dict:
        """Combine reports from multiple models."""
        metadata = {
            'source_file': str(source_file),
            'analysis_timestamp': datetime.now().isoformat(),
            'models_used': list(reports.keys()),
        }
        
        # Add pivot filter information
        if include_pivots:
            metadata['pivot_filter'] = {'type': 'include', 'pivots': include_pivots}
        elif exclude_pivots:
            metadata['pivot_filter'] = {'type': 'exclude', 'pivots': exclude_pivots}
        
        if analyzed_pivots:
            metadata['analyzed_pivots'] = analyzed_pivots
        
        return {
            'metadata': metadata,
            'results_by_model': reports,
        }
    
    def _print_extreme_cases(self, report: dict):
        """Print extreme cases to console for easy review."""
        self.logger.info("\n" + "="*60)
        self.logger.info("EXTREME CASES (for investigation)")
        self.logger.info("="*60)
        
        for model_name, model_report in report.get('results_by_model', {}).items():
            investigation = model_report.get('investigation', {})
            
            # Print high RT, low chrF cases
            extreme_chrf = investigation.get('extreme_cases_surface_rt_vs_chrf_high', {})
            high_rt_low_chrf = extreme_chrf.get('high_rt_low_quality', [])
            
            if high_rt_low_chrf:
                self.logger.info(f"\n[{model_name.upper()}] High Semantic RT but Low chrF (high stratum):")
                self.logger.info("-" * 50)
                for case in high_rt_low_chrf[:3]:
                    self.logger.info(f"  Sentence {case['sentence_id']} ({case['pivot_lang']})")
                    self.logger.info(f"    RT: {case.get('rt_geometric', 'N/A'):.3f}, chrF: {case.get('quality_chrf', 'N/A')}")
                    self.logger.info(f"    Output: {case['output_text'][:80]}...")
                    self.logger.info(f"    Ref:    {case['target_reference'][:80]}...")
                    self.logger.info("")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Semantic analysis for round-trip translation consistency"
    )
    parser.add_argument("results_file", type=Path,
                        help="Path to JSONL results file")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory (default: same as results file)")
    parser.add_argument("--model", "-m", choices=['labse', 'sonar', 'both', 'auto'],
                        default='auto',
                        help="Embedding model to use for RT metrics (default: auto)")
    
    # COMET model options
    parser.add_argument("--comet-path", type=str, default=None,
                        help="Path to COMET model for quality_comet")
    parser.add_argument("--comet-qe-path", type=str, default=None,
                        help="Path to COMET-QE model for direct_source_output_comet")
    parser.add_argument("--comet-batch-size", type=int, default=8,
                        help="Batch size for COMET processing (default: 8)")
    
    parser.add_argument("--no-per-pivot", action="store_true",
                        help="Disable per-pivot analysis (faster)")
    parser.add_argument("--bootstrap-n", type=int, default=1000,
                        help="Number of bootstrap samples for CI (default: 1000)")
    
    # Pivot filtering options
    parser.add_argument("--pivots", type=str, default=None,
                        help="Comma-separated list of pivot languages to INCLUDE (e.g., 'fra,deu,cmn')")
    parser.add_argument("--exclude-pivots", type=str, default=None,
                        help="Comma-separated list of pivot languages to EXCLUDE (e.g., 'hin,rus')")
    
    args = parser.parse_args()
    
    # Parse pivot filter arguments
    include_pivots = None
    exclude_pivots = None
    
    if args.pivots:
        include_pivots = [p.strip() for p in args.pivots.split(',')]
        print(f"Will include only pivots: {include_pivots}")
    
    if args.exclude_pivots:
        exclude_pivots = [p.strip() for p in args.exclude_pivots.split(',')]
        print(f"Will exclude pivots: {exclude_pivots}")
    
    # Run analysis
    analyzer = SemanticAnalyzer(
        model_name=args.model,
        comet_path=args.comet_path,
        comet_qe_path=args.comet_qe_path,
        comet_batch_size=args.comet_batch_size
    )
    analyzer.analyze(
        args.results_file, 
        output_dir=args.output_dir,
        per_pivot=not args.no_per_pivot,
        bootstrap_n=args.bootstrap_n,
        include_pivots=include_pivots,
        exclude_pivots=exclude_pivots
    )


if __name__ == "__main__":
    main()