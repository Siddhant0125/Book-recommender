#!/usr/bin/env python3
"""
Mood Classification for Books using Sentence-BERT and Zero-Shot Classification
==============================================================================

This module provides advanced mood classification for book descriptions using:
1. Sentence-BERT (SBERT) embeddings for semantic similarity
2. Zero-shot classification for direct mood prediction
3. Confidence-based mood assignment

Features:
- No need for labeled training data
- Works with any mood labels defined by the user
- Provides confidence scores for each mood prediction
- Efficient batch processing of multiple books
- Caching for repeated descriptions
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from functools import lru_cache

# Try importing transformers and sentence-transformers
logger_msg = "Transformers/Sentence-BERT not available. Install with: pip install transformers sentence-transformers"
try:
    from sentence_transformers import SentenceTransformer, util
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not TRANSFORMERS_AVAILABLE:
    logger.warning(logger_msg)


class MoodClassifier:
    """
    Classify book descriptions into mood categories using advanced NLP techniques.

    This class uses two approaches:
    1. Zero-Shot Classification: Direct mood prediction without fine-tuning
    2. Sentence-BERT Similarity: Semantic similarity between descriptions and mood templates

    Advantages:
    - Works with any mood category
    - No labeled training data required
    - Provides confidence scores
    - Efficient batch processing
    - Can handle custom mood definitions
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_zero_shot: bool = True):
        """
        Initialize the MoodClassifier

        Args:
            model_name: Sentence-BERT model to use
                       "all-MiniLM-L6-v2" (fast, 22M params)
                       "all-mpnet-base-v2" (more accurate, 109M params)
                       "paraphrase-MiniLM-L6-v2" (good for semantic search)
            use_zero_shot: Whether to use zero-shot classification in addition to SBERT
                          (requires 'facebook/bart-large-mnli' model)
        """
        self.model_name = model_name
        self.use_zero_shot = use_zero_shot
        self.sbert_model = None
        self.zero_shot_pipeline = None
        self.mood_templates = None

        # Initialize models if available
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading Sentence-BERT model: {model_name}")
                self.sbert_model = SentenceTransformer(model_name)
                logger.info("✓ Sentence-BERT model loaded successfully")

                if use_zero_shot:
                    logger.info("Loading zero-shot classification pipeline...")
                    self.zero_shot_pipeline = pipeline(
                        "zero-shot-classification",
                        model="facebook/bart-large-mnli"
                    )
                    logger.info("✓ Zero-shot classification pipeline loaded successfully")
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                logger.info("Mood classification will use fallback keyword-based approach")
        else:
            logger.warning("Transformers not available. Install: pip install transformers sentence-transformers")

    def set_mood_templates(self, mood_templates: Dict[str, List[str]]):
        """
        Set mood-specific templates for semantic similarity matching

        Args:
            mood_templates: Dictionary mapping moods to lists of template phrases
            Example: {
                "hopeful": ["uplifting story", "inspirational message", "positive ending"],
                "dark": ["scary atmosphere", "horror elements", "disturbing content"]
            }
        """
        self.mood_templates = mood_templates
        logger.info(f"Set mood templates for {len(mood_templates)} moods")

        # Pre-compute embeddings for templates
        if self.sbert_model and mood_templates:
            self._compute_template_embeddings()

    def _compute_template_embeddings(self):
        """Pre-compute and cache embeddings for mood templates"""
        if not self.mood_templates or not self.sbert_model:
            return

        logger.info("Computing template embeddings for faster inference...")
        self.mood_template_embeddings = {}

        for mood, templates in self.mood_templates.items():
            # Embed all templates for this mood
            embeddings = self.sbert_model.encode(templates, convert_to_tensor=True)
            # Store mean embedding for this mood
            self.mood_template_embeddings[mood] = embeddings.mean(dim=0)

        logger.info("✓ Template embeddings computed")

    def _split_into_chunks(self, text: str, max_words: int = 300) -> List[str]:
        """Split a (possibly long) description into word-based chunks."""
        words = str(text).split()
        if len(words) <= max_words:
            return [" ".join(words)]

        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        return chunks

    def _classify_single_chunk(self, description: str, mood_labels: List[str]) -> Dict[str, float]:
        """
        Classify a single (short) chunk of text into moods using the hybrid method.

        This is the per-chunk version; the main classify() will aggregate over chunks.
        """
        if not description or pd.isna(description):
            return {}

        description = str(description)
        scores_dict: Dict[str, float] = {}
        zero_shot_scores: Dict[str, float] = {}
        sbert_scores: Dict[str, float] = {}

        # ---- 1) Zero-shot classification (multi-label) ----
        if self.zero_shot_pipeline:
            try:
                result = self.zero_shot_pipeline(
                    description,
                    mood_labels,
                    multi_label=True,   # ✅ multi-label, not multi_class
                    truncation=True
                )
                for label, score in zip(result["labels"], result["scores"]):
                    zero_shot_scores[label] = float(score)
            except Exception as e:
                logger.debug(f"Zero-shot error: {e}")

        # ---- 2) SBERT semantic similarity ----
        if self.sbert_model and self.mood_template_embeddings:
            try:
                desc_embedding = self.sbert_model.encode(
                    description,
                    convert_to_tensor=True
                )
                for mood in mood_labels:
                    if mood in self.mood_template_embeddings:
                        mood_embedding = self.mood_template_embeddings[mood]
                        similarity = util.pytorch_cos_sim(
                            desc_embedding, mood_embedding
                        ).item()
                        # Map from [-1, 1] to [0, 1]
                        similarity = (similarity + 1.0) / 2.0
                        sbert_scores[mood] = float(similarity)
            except Exception as e:
                logger.debug(f"SBERT error: {e}")

        # ---- 3) Combine scores (weighted hybrid) ----
        all_moods = set(mood_labels)
        alpha = 0.7  # trust zero-shot slightly more than SBERT

        for mood in all_moods:
            z = zero_shot_scores.get(mood)
            s = sbert_scores.get(mood)

            if z is not None and s is not None:
                combined = alpha * z + (1.0 - alpha) * s
            elif z is not None:
                combined = z
            elif s is not None:
                combined = s
            else:
                continue

            scores_dict[mood] = float(combined)

        return scores_dict

    def classify(self, description: str, mood_labels: List[str],
                 temperature: float = 0.1) -> Dict[str, float]:
        """
        Classify book description mood using hybrid approach.

        For long descriptions, splits into chunks and aggregates scores.
        Returns a dict {mood: score} with scores in [0, 1] that do NOT
        necessarily sum to 1 (multi-label friendly).
        """
        if not description or pd.isna(description):
            return {}

        # 1) Split into word-based chunks (handles your ~1200-word blurbs)
        chunks = self._split_into_chunks(description, max_words=300)

        # 2) Run per-chunk classification and aggregate scores per mood
        aggregate: Dict[str, List[float]] = {}

        for chunk in chunks:
            chunk_scores = self._classify_single_chunk(chunk, mood_labels)
            for mood, score in chunk_scores.items():
                aggregate.setdefault(mood, []).append(score)

        if not aggregate:
            return {}

        # 3) Aggregation rule across chunks
        # Using max: if a mood is strong anywhere, treat it as strong overall
        agg_scores: Dict[str, float] = {
            mood: float(max(scores))    # or np.mean(scores) if you want average
            for mood, scores in aggregate.items()
        }

        # 4) Optional normalization to [0, 1] (NOT softmax, keeps multi-label meaning)
        scores_arr = np.array(list(agg_scores.values()), dtype=np.float32)
        min_s, max_s = scores_arr.min(), scores_arr.max()
        if max_s > min_s:
            norm_scores = (scores_arr - min_s) / (max_s - min_s)
        else:
            norm_scores = np.ones_like(scores_arr)

        return {
            mood: float(score)
            for mood, score in zip(agg_scores.keys(), norm_scores)
        }


    def classify_batch(self, descriptions: List[str], mood_labels: List[str],
                      threshold: float = 0.3) -> List[Dict[str, float]]:
        """
        Classify multiple book descriptions efficiently using hybrid approach

        Args:
            descriptions: List of book descriptions
            mood_labels: List of possible mood labels
            threshold: Minimum confidence score to include a mood (0.0-1.0)

        Returns:
            List of dictionaries, each mapping moods to confidence scores
        """
        results = []

        logger.info(f"Classifying {len(descriptions)} descriptions using hybrid method...")

        for i, description in enumerate(descriptions):
            if not description or pd.isna(description):
                results.append({})
                continue

            # Use hybrid classification
            scores = self.classify(str(description), mood_labels)

            # Apply threshold
            filtered_scores = {mood: score for mood, score in scores.items() if score >= threshold}
            results.append(filtered_scores)

            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(descriptions)} descriptions")

        return results

    def get_top_mood(self, description: str, mood_labels: List[str],
                    default_mood: str = "thoughtful") -> str:
        """
        Get the single most likely mood for a description using hybrid classification

        Args:
            description: Book description text
            mood_labels: List of possible mood labels
            default_mood: Mood to return if classification fails

        Returns:
            Most likely mood label
        """
        if not description or pd.isna(description):
            return default_mood

        # Get scores using hybrid method
        scores = self.classify(str(description), mood_labels)

        # Return mood with highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return default_mood

    def assign_moods_to_books(self, books_df: pd.DataFrame,
                             description_column: str = "description",
                             mood_labels: Optional[List[str]] = None,
                             threshold: float = 0.3) -> pd.DataFrame:
        """
        Assign mood labels to all books in a DataFrame using hybrid classification

        Args:
            books_df: DataFrame with book descriptions
            description_column: Name of column containing descriptions
            mood_labels: List of mood labels to classify into
            threshold: Minimum confidence to assign a mood

        Returns:
            DataFrame with new columns:
            - "predicted_mood": Primary mood
            - "mood_scores": Dictionary of all mood scores
            - "mood_confidence": Confidence in primary mood
        """
        if mood_labels is None:
            mood_labels = [
                "dark", "romantic", "hopeful", "thoughtful", "anxious",
                "curious", "emotional", "funny", "adventurous", "mysterious"
            ]

        logger.info(f"Assigning moods to {len(books_df)} books using hybrid method...")

        # Get descriptions
        descriptions = books_df[description_column].fillna("").tolist()

        # Classify in batch
        all_scores = self.classify_batch(descriptions, mood_labels, threshold=threshold)

        # Extract primary mood and confidence
        predicted_moods = []
        confidences = []

        for scores in all_scores:
            if scores:
                mood, confidence = max(scores.items(), key=lambda x: x[1])
                predicted_moods.append(mood)
                confidences.append(confidence)
            else:
                predicted_moods.append("thoughtful")
                confidences.append(0.0)

        # Add to DataFrame
        result_df = books_df.copy()
        result_df["predicted_mood"] = predicted_moods
        result_df["mood_confidence"] = confidences
        result_df["mood_scores"] = all_scores

        logger.info(f"✓ Assigned moods to {len(result_df)} books")

        return result_df


class MoodLabelAssigner:
    """
    Utility class for assigning mood labels to books and saving results

    This class handles:
    - Loading books with descriptions
    - Running mood classification
    - Saving results to enriched dataset
    - Providing statistics on mood distribution
    """

    def __init__(self, classifier: MoodClassifier):
        """
        Initialize the assigner

        Args:
            classifier: MoodClassifier instance
        """
        self.classifier = classifier

    def process_books_file(self, input_csv: str, output_csv: str,
                          description_column: str = "description",
                          mood_labels: Optional[List[str]] = None,
                          method: str = "hybrid") -> pd.DataFrame:
        """
        Process a CSV file of books and add mood labels

        Args:
            input_csv: Path to input CSV with book descriptions
            output_csv: Path to save enriched CSV with mood labels
            description_column: Column name containing descriptions
            mood_labels: List of mood labels to use
            method: Classification method

        Returns:
            DataFrame with added mood columns
        """
        logger.info(f"Loading books from {input_csv}...")
        books_df = pd.read_csv(input_csv)

        logger.info(f"Processing {len(books_df)} books...")
        enriched_df = self.classifier.assign_moods_to_books(
            books_df,
            description_column=description_column,
            mood_labels=mood_labels        )

        logger.info(f"Saving enriched data to {output_csv}...")
        enriched_df.to_csv(output_csv, index=False)

        # Print statistics
        self._print_mood_statistics(enriched_df)

        return enriched_df

    def _print_mood_statistics(self, df: pd.DataFrame):
        """Print statistics about mood distribution"""
        print("\n" + "="*60)
        print("MOOD DISTRIBUTION STATISTICS")
        print("="*60)

        mood_counts = df["predicted_mood"].value_counts()
        print(f"\nMood Distribution ({len(df)} total books):")
        for mood, count in mood_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {mood:15s}: {count:5d} books ({percentage:5.1f}%)")

        avg_confidence = df["mood_confidence"].mean()
        print(f"\nAverage Confidence: {avg_confidence:.3f}")
        print(f"Min Confidence:     {df['mood_confidence'].min():.3f}")
        print(f"Max Confidence:     {df['mood_confidence'].max():.3f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    logger.info("mood_classifier.py - Hybrid Mood Classification Module")

