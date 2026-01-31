#!/usr/bin/env python3
"""
Translate text using LLMs via the class CELS infrastructure.

This script translates source text and compares against reference translations.
Supports both direct translation and chained (multi-hop) translation methods.

Requirements:
    pip install openai pandas tqdm

Usage:
    # Translate a single sentence
    python translate_with_llm.py --text "Hello, how are you?" --target es

    # Translate from a dataset file
    python translate_with_llm.py --input data/Europarl_en_es.parquet --target es --max-samples 100

    # Use chained translation (experimental)
    python translate_with_llm.py --input data/test.parquet --target ja --chain es,fr

    # Use a different model
    python translate_with_llm.py --text "Hello" --target es --model qwen80
"""

import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Try to import dependencies
try:
    from openai import OpenAI
    import pandas as pd
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install openai pandas tqdm")
    sys.exit(1)


# =============================================================================
# Model Configuration (from lab1/model_servers.yaml)
# =============================================================================

MODELS = {
    "oss120": {
        "name": "OSS 120B",
        "base_url": "http://66.55.67.65:80/v1",
        "model": "oss120",
        "api_key": "CELS",
    },
    "llama70": {
        "name": "Llama 3.3 70B", 
        "base_url": "http://103.101.203.226:80/v1",
        "model": "llama70",
        "api_key": "CELS",
    },
    "qwen80": {
        "name": "Qwen 80B",
        "base_url": "http://103.90.163.143:80/v1",
        "model": "qwen80",
        "api_key": "CELS",
    },
    "oss20": {
        "name": "OSS 20B (unstable)",
        "base_url": "http://195.88.24.64:80/v1",
        "model": "oss20",
        "api_key": "CELS",
    },
    # Local Ollama models (if running)
    "ollama-llama": {
        "name": "Ollama Llama 3.2",
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2:latest",
        "api_key": "not-needed",
    },
}

LANGUAGE_NAMES = {
    # Major European Languages
    "en": "English",
    "es": "Spanish",
    "fr": "French", 
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "fi": "Finnish",
    "da": "Danish",
    "no": "Norwegian",
    "sv": "Swedish",
    "is": "Icelandic",
    "mt": "Maltese",
    "sq": "Albanian",
    "mk": "Macedonian",
    "bs": "Bosnian",
    "lb": "Luxembourgish",
    "be": "Belarusian",
    
    # Celtic Languages
    "cy": "Welsh",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "br": "Breton",
    
    # Other European
    "eu": "Basque",
    "ca": "Catalan",
    "gl": "Galician",
    
    # East Asian Languages
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "lo": "Lao",
    "km": "Khmer",
    "my": "Burmese",
    "mn": "Mongolian",
    
    # South Asian Languages
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ne": "Nepali",
    "si": "Sinhala",
    "or": "Odia",
    "as": "Assamese",
    
    # Southeast Asian Languages
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Tagalog",
    "jv": "Javanese",
    "su": "Sundanese",
    "ceb": "Cebuano",
    
    # Middle Eastern Languages
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
    "tr": "Turkish",
    "ku": "Kurdish",
    "ps": "Pashto",
    "az": "Azerbaijani",
    
    # African Languages
    "sw": "Swahili",  # CRITICAL: This was missing!
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "am": "Amharic",
    "om": "Oromo",
    "so": "Somali",
    "zu": "Zulu",
    "xh": "Xhosa",
    "af": "Afrikaans",
    "sn": "Shona",
    "ny": "Chichewa",
    "rw": "Kinyarwanda",
    "mg": "Malagasy",
    "ti": "Tigrinya",
    "wo": "Wolof",
    "ff": "Fulah",
    "tw": "Twi",
    "ln": "Lingala",
    "lg": "Luganda",
    
    # Central Asian Languages
    "kk": "Kazakh",
    "uz": "Uzbek",
    "ky": "Kyrgyz",
    "tg": "Tajik",
    "tt": "Tatar",
    "tk": "Turkmen",
    
    # Other Languages
    "ka": "Georgian",
    "hy": "Armenian",
    "ht": "Haitian Creole",
    "eo": "Esperanto",
    "la": "Latin",
}


# =============================================================================
# Translation Methods
# =============================================================================

@dataclass
class TranslationResult:
    """Result of a translation."""
    source_text: str
    source_lang: str
    target_lang: str
    translation: str
    method: str  # "direct" or "chain:es,fr,..."
    intermediate_translations: dict = field(default_factory=dict)  # lang -> text
    inference_time: float = 0.0
    model: str = ""
    

class Translator:
    """LLM-based translator using class infrastructure."""
    
    def __init__(self, model_key: str = "oss120", temperature: float = 0.3, max_tokens: int = 16384):
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
        
        self.model_config = MODELS[model_key]
        self.model_key = model_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = OpenAI(
            base_url=self.model_config["base_url"],
            api_key=self.model_config["api_key"],
        )
        
        print(f"Initialized translator with {self.model_config['name']}")
        print(f"  Endpoint: {self.model_config['base_url']}")
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make a single LLM call."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_config["model"],
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return response.choices[0].message.content.strip()
    
    def translate_direct(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> TranslationResult:
        """
        Direct translation from source to target language.
        
        This is the baseline method.
        """
        src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        
        system_prompt = (
            f"You are a professional translator. Translate the given {src_name} text "
            f"to {tgt_name}. Output ONLY the translation, nothing else. "
            "Do not include explanations, notes, or the original text."
        )
        
        prompt = f"Translate to {tgt_name}:\n\n{text}"
        
        start_time = time.time()
        translation = self._call_llm(prompt, system_prompt)
        elapsed = time.time() - start_time
        
        return TranslationResult(
            source_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            translation=translation,
            method="direct",
            inference_time=elapsed,
            model=self.model_key,
        )
    
    def translate_chain(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        intermediate_langs: list[str],
    ) -> TranslationResult:
        """
        Chained translation through intermediate languages.
        
        e.g., English -> Spanish -> French -> Japanese
        
        Hypothesis: More "thinking time" in translation space improves quality.
        """
        src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        
        # Build the full chain
        chain = [source_lang] + intermediate_langs + [target_lang]
        
        system_prompt = (
            "You are a professional translator. Translate the given text to the "
            "requested language. Output ONLY the translation, nothing else."
        )
        
        current_text = text
        current_lang = source_lang
        intermediates = {}
        total_time = 0.0
        
        # Translate through each hop
        for next_lang in chain[1:]:
            next_name = LANGUAGE_NAMES.get(next_lang, next_lang)
            curr_name = LANGUAGE_NAMES.get(current_lang, current_lang)
            
            prompt = f"Translate from {curr_name} to {next_name}:\n\n{current_text}"
            
            start_time = time.time()
            current_text = self._call_llm(prompt, system_prompt)
            total_time += time.time() - start_time
            
            # Store intermediate (but not final) translations
            if next_lang != target_lang:
                intermediates[next_lang] = current_text
        
        return TranslationResult(
            source_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            translation=current_text,
            method=f"chain:{','.join(intermediate_langs)}",
            intermediate_translations=intermediates,
            inference_time=total_time,
            model=self.model_key,
        )
    
    def translate_chain_single_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        intermediate_langs: list[str],
    ) -> TranslationResult:
        """
        Chained translation in a SINGLE prompt (show your work style).
        
        Asks the model to produce all intermediate translations in one call,
        similar to chain-of-thought prompting.
        """
        chain = [source_lang] + intermediate_langs + [target_lang]
        chain_desc = " -> ".join([LANGUAGE_NAMES.get(l, l) for l in chain])
        
        # Build structured prompt
        system_prompt = (
            "You are a professional translator. You will translate text through "
            "multiple languages step by step. Show each intermediate translation. "
            "Format your response as JSON."
        )
        
        steps_desc = []
        for i in range(len(chain) - 1):
            from_lang = LANGUAGE_NAMES.get(chain[i], chain[i])
            to_lang = LANGUAGE_NAMES.get(chain[i+1], chain[i+1])
            steps_desc.append(f"Step {i+1}: {from_lang} -> {to_lang}")
        
        prompt = f"""Translate the following text through this chain: {chain_desc}

Original text ({LANGUAGE_NAMES.get(source_lang, source_lang)}):
{text}

Provide all translations in this JSON format:
{{
    "steps": [
        {{"from": "lang1", "to": "lang2", "translation": "..."}},
        ...
    ],
    "final_translation": "..."
}}

{chr(10).join(steps_desc)}

Output only valid JSON, no other text."""

        start_time = time.time()
        response = self._call_llm(prompt, system_prompt)
        elapsed = time.time() - start_time
        
        # Parse JSON response
        intermediates = {}
        final_translation = response  # Fallback
        
        try:
            # Find JSON in response
            if "{" in response and "}" in response:
                json_start = response.index("{")
                json_end = response.rindex("}") + 1
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Extract intermediates
                if "steps" in data:
                    for step in data["steps"][:-1]:  # All but last
                        to_lang = step.get("to", "")
                        # Map language name back to code
                        lang_code = next(
                            (k for k, v in LANGUAGE_NAMES.items() if v.lower() == to_lang.lower()),
                            to_lang
                        )
                        intermediates[lang_code] = step.get("translation", "")
                
                final_translation = data.get("final_translation", response)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Could not parse JSON response: {e}")
            # Try to extract just the final translation
            if "final_translation" in response:
                try:
                    # Simple extraction
                    import re
                    match = re.search(r'"final_translation"\s*:\s*"([^"]+)"', response)
                    if match:
                        final_translation = match.group(1)
                except Exception:
                    pass
        
        return TranslationResult(
            source_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            translation=final_translation,
            method=f"chain_single:{','.join(intermediate_langs)}",
            intermediate_translations=intermediates,
            inference_time=elapsed,
            model=self.model_key,
        )


# =============================================================================
# Batch Processing
# =============================================================================

def translate_dataset(
    translator: Translator,
    df: pd.DataFrame,
    source_col: str,
    target_lang: str,
    source_lang: str = "en",
    method: str = "direct",
    chain_langs: Optional[list[str]] = None,
    max_samples: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Translate a dataset and return results with the LLM translations added.
    """
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples).copy()
    else:
        df = df.copy()
    
    results = []
    iterator = df[source_col].tolist()
    if show_progress:
        iterator = tqdm(iterator, desc=f"Translating ({method})")
    
    for text in iterator:
        try:
            if method == "direct":
                result = translator.translate_direct(text, source_lang, target_lang)
            elif method == "chain" and chain_langs:
                result = translator.translate_chain(text, source_lang, target_lang, chain_langs)
            elif method == "chain_single" and chain_langs:
                result = translator.translate_chain_single_prompt(text, source_lang, target_lang, chain_langs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results.append({
                "llm_translation": result.translation,
                "method": result.method,
                "inference_time": result.inference_time,
                "intermediates": json.dumps(result.intermediate_translations) if result.intermediate_translations else "",
            })
        except Exception as e:
            print(f"\nError translating: {e}")
            results.append({
                "llm_translation": f"ERROR: {e}",
                "method": method,
                "inference_time": 0.0,
                "intermediates": "",
            })
    
    # Add results to dataframe
    results_df = pd.DataFrame(results)
    for col in results_df.columns:
        df[col] = results_df[col].values
    
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Translate text using LLMs from class infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sentence translation
  python translate_with_llm.py --text "Hello, how are you?" --target es

  # Translate dataset with direct method
  python translate_with_llm.py --input data/Europarl_en_es.parquet --target es --max-samples 100

  # Chained translation (multi-hop)
  python translate_with_llm.py --text "Hello world" --target ja --chain es,fr
  
  # Chained translation in single prompt (CoT-style)
  python translate_with_llm.py --text "Hello world" --target ja --chain es,fr --method chain_single

Available models:
  oss120   - OSS 120B (default, recommended)
  llama70  - Llama 3.3 70B  
  qwen80   - Qwen 80B
  oss20    - OSS 20B (may be unstable)

Chain translation hypothesis:
  The --chain option tests whether intermediate translation steps improve quality,
  similar to how chain-of-thought improves reasoning. For example:
    --target ja --chain es,fr
  will translate: English -> Spanish -> French -> Japanese
        """
    )
    
    # Input options
    parser.add_argument("--text", type=str, help="Single text to translate")
    parser.add_argument("--input", type=str, help="Input file (parquet, csv, or jsonl)")
    parser.add_argument("--source-col", type=str, default="source", help="Source text column name")
    parser.add_argument("--source-lang", type=str, default="en", help="Source language code")
    
    # Translation options
    parser.add_argument("--target", type=str, required=True, help="Target language code")
    parser.add_argument("--method", type=str, choices=["direct", "chain", "chain_single"], 
                        default="direct", help="Translation method")
    parser.add_argument("--chain", type=str, help="Intermediate languages for chain translation (comma-separated)")
    
    # Model options
    parser.add_argument("--model", type=str, default="oss120", 
                        choices=list(MODELS.keys()), help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max tokens in response")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--max-samples", type=int, help="Max samples to process")
    
    args = parser.parse_args()
    
    # Parse chain languages
    chain_langs = None
    if args.chain:
        chain_langs = [l.strip() for l in args.chain.split(",")]
        if args.method == "direct":
            args.method = "chain"  # Auto-switch if chain provided
    
    # Initialize translator
    translator = Translator(
        model_key=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    # Single text mode
    if args.text:
        print(f"\nSource ({args.source_lang}): {args.text}")
        print(f"Target: {args.target}")
        print(f"Method: {args.method}")
        if chain_langs:
            print(f"Chain: {args.source_lang} -> {' -> '.join(chain_langs)} -> {args.target}")
        print("-" * 50)
        
        if args.method == "direct":
            result = translator.translate_direct(args.text, args.source_lang, args.target)
        elif args.method == "chain" and chain_langs:
            result = translator.translate_chain(args.text, args.source_lang, args.target, chain_langs)
        elif args.method == "chain_single" and chain_langs:
            result = translator.translate_chain_single_prompt(args.text, args.source_lang, args.target, chain_langs)
        else:
            print("Error: Chain method requires --chain argument")
            return
        
        print(f"\nTranslation: {result.translation}")
        if result.intermediate_translations:
            print("\nIntermediate translations:")
            for lang, trans in result.intermediate_translations.items():
                print(f"  {LANGUAGE_NAMES.get(lang, lang)}: {trans}")
        print(f"\nInference time: {result.inference_time:.2f}s")
        return
    
    # Dataset mode
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return
        
        # Load data
        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        elif input_path.suffix == ".csv":
            df = pd.read_csv(input_path)
        elif input_path.suffix in [".json", ".jsonl"]:
            df = pd.read_json(input_path, lines=input_path.suffix == ".jsonl")
        else:
            print(f"Error: Unknown file format: {input_path.suffix}")
            return
        
        print(f"\nLoaded {len(df)} samples from {input_path}")
        print(f"Columns: {list(df.columns)}")
        
        # Translate
        result_df = translate_dataset(
            translator=translator,
            df=df,
            source_col=args.source_col,
            target_lang=args.target,
            source_lang=args.source_lang,
            method=args.method,
            chain_langs=chain_langs,
            max_samples=args.max_samples,
        )
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_translated_{args.model}_{args.method}.parquet"
        
        if output_path.suffix == ".parquet":
            result_df.to_parquet(output_path, index=False)
        elif output_path.suffix == ".csv":
            result_df.to_csv(output_path, index=False)
        else:
            result_df.to_json(output_path, orient="records", lines=True)
        
        print(f"\nSaved results to {output_path}")
        
        # Print summary
        print(f"\n{'='*50}")
        print("TRANSLATION SUMMARY")
        print(f"{'='*50}")
        print(f"Model: {args.model}")
        print(f"Method: {args.method}")
        print(f"Samples: {len(result_df)}")
        print(f"Avg inference time: {result_df['inference_time'].mean():.2f}s")
        print(f"Total time: {result_df['inference_time'].sum():.1f}s")
        
        # Show a few examples
        print(f"\n{'='*50}")
        print("SAMPLE TRANSLATIONS")
        print(f"{'='*50}")
        for i, row in result_df.head(3).iterrows():
            print(f"\n[{i+1}] Source: {row[args.source_col][:80]}...")
            if "target" in row:
                print(f"    Reference: {row['target'][:80]}...")
            print(f"    LLM: {row['llm_translation'][:80]}...")
    
    else:
        print("Error: Must provide either --text or --input")
        parser.print_help()


if __name__ == "__main__":
    main()