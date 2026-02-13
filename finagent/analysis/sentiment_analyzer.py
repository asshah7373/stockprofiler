"""
Financial Sentiment Analysis Module

Provides NLP-based sentiment analysis for financial news and circulars.
Uses multiple approaches with fallbacks:
1. FinBERT (best for financial text) - requires transformers
2. VADER (rule-based) - requires nltk
3. Keyword matching (basic fallback)

The analyzer understands financial context like:
- "profit declined less than expected" -> POSITIVE
- "beat analyst estimates" -> POSITIVE
- "missed guidance" -> NEGATIVE
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    VERY_POSITIVE = "VERY_POSITIVE"
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    VERY_NEGATIVE = "VERY_NEGATIVE"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    label: SentimentLabel
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    method: str  # Which analyzer was used
    details: Dict = None

    def to_dict(self) -> Dict:
        return {
            'label': self.label.value,
            'score': round(self.score, 3),
            'confidence': round(self.confidence, 3),
            'method': self.method,
        }


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analyzer.

    Uses ProsusAI/finbert model which is specifically trained
    on financial text and understands financial context.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialized = False
        self._available = None

    def is_available(self) -> bool:
        """Check if transformers and model are available."""
        if self._available is not None:
            return self._available

        try:
            from transformers import pipeline
            self._available = True
        except ImportError:
            self._available = False
            logger.info("transformers not installed, FinBERT unavailable")

        return self._available

    def initialize(self) -> bool:
        """Initialize the FinBERT model."""
        if self._initialized:
            return True

        if not self.is_available():
            return False

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                pipeline
            )

            logger.info("Loading FinBERT model (this may take a moment)...")

            # Use ProsusAI/finbert - trained on financial sentiment
            model_name = "ProsusAI/finbert"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True,
                max_length=512
            )

            self._initialized = True
            logger.info("FinBERT model loaded successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize FinBERT: {e}")
            self._available = False
            return False

    def analyze(self, text: str) -> Optional[SentimentResult]:
        """Analyze sentiment using FinBERT."""
        if not self.initialize():
            return None

        try:
            # Truncate very long text
            text = text[:1000] if len(text) > 1000 else text

            result = self.pipeline(text)[0]
            label = result['label'].lower()
            score = result['score']

            # Map FinBERT labels to our labels
            # FinBERT outputs: positive, negative, neutral
            if label == 'positive':
                if score > 0.9:
                    sentiment_label = SentimentLabel.VERY_POSITIVE
                else:
                    sentiment_label = SentimentLabel.POSITIVE
                sentiment_score = score
            elif label == 'negative':
                if score > 0.9:
                    sentiment_label = SentimentLabel.VERY_NEGATIVE
                else:
                    sentiment_label = SentimentLabel.NEGATIVE
                sentiment_score = -score
            else:
                sentiment_label = SentimentLabel.NEUTRAL
                sentiment_score = 0.0

            return SentimentResult(
                label=sentiment_label,
                score=sentiment_score,
                confidence=score,
                method="FinBERT"
            )

        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return None


class VADERAnalyzer:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) analyzer.

    Rule-based sentiment analysis that works well for social media
    and news headlines. Enhanced with financial terms.
    """

    def __init__(self):
        self.analyzer = None
        self._available = None

    def is_available(self) -> bool:
        """Check if VADER is available."""
        if self._available is not None:
            return self._available

        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk

            # Try to use VADER, download if needed
            try:
                self.analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
                self.analyzer = SentimentIntensityAnalyzer()

            # Add financial terms to VADER lexicon
            self._add_financial_lexicon()

            self._available = True
        except ImportError:
            self._available = False
            logger.info("nltk not installed, VADER unavailable")
        except Exception as e:
            logger.warning(f"VADER initialization failed: {e}")
            self._available = False

        return self._available

    def _add_financial_lexicon(self):
        """Add financial terms to VADER's lexicon."""
        if not self.analyzer:
            return

        # Positive financial terms
        positive_terms = {
            'beat': 2.5,
            'beats': 2.5,
            'exceeded': 2.5,
            'surpassed': 2.5,
            'outperform': 2.0,
            'upgrade': 2.0,
            'upgraded': 2.0,
            'bullish': 2.0,
            'growth': 1.5,
            'profit': 1.5,
            'dividend': 1.5,
            'buyback': 1.5,
            'expansion': 1.5,
            'contract win': 2.0,
            'order win': 2.0,
            'record': 1.5,
            'highest': 1.5,
            'strong': 1.0,
            'robust': 1.0,
            'healthy': 1.0,
        }

        # Negative financial terms
        negative_terms = {
            'missed': -2.5,
            'miss': -2.5,
            'decline': -2.0,
            'declined': -2.0,
            'downgrade': -2.0,
            'downgraded': -2.0,
            'bearish': -2.0,
            'loss': -2.0,
            'losses': -2.0,
            'weak': -1.5,
            'below': -1.5,
            'disappointing': -2.0,
            'fraud': -3.0,
            'scam': -3.0,
            'penalty': -2.0,
            'fine': -1.5,
            'lawsuit': -2.0,
            'investigation': -2.0,
            'default': -2.5,
            'bankruptcy': -3.0,
        }

        # Update lexicon
        self.analyzer.lexicon.update(positive_terms)
        self.analyzer.lexicon.update(negative_terms)

    def analyze(self, text: str) -> Optional[SentimentResult]:
        """Analyze sentiment using VADER."""
        if not self.is_available():
            return None

        try:
            scores = self.analyzer.polarity_scores(text)
            compound = scores['compound']

            # Determine label based on compound score
            if compound >= 0.5:
                label = SentimentLabel.VERY_POSITIVE
            elif compound >= 0.1:
                label = SentimentLabel.POSITIVE
            elif compound <= -0.5:
                label = SentimentLabel.VERY_NEGATIVE
            elif compound <= -0.1:
                label = SentimentLabel.NEGATIVE
            else:
                label = SentimentLabel.NEUTRAL

            # Calculate confidence from the dominant sentiment
            pos = scores['pos']
            neg = scores['neg']
            neu = scores['neu']
            confidence = max(pos, neg, neu)

            return SentimentResult(
                label=label,
                score=compound,
                confidence=confidence,
                method="VADER",
                details={
                    'positive': pos,
                    'negative': neg,
                    'neutral': neu,
                    'compound': compound,
                }
            )

        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")
            return None


class KeywordAnalyzer:
    """
    Keyword-based sentiment analyzer (fallback).

    Uses pattern matching with financial keywords.
    Less accurate but requires no external dependencies.
    """

    # Context-aware patterns (understands financial nuances)
    POSITIVE_PATTERNS = [
        # Strong positives
        (r'beat\s+(?:estimates?|expectations?|guidance|forecast)', 3.0),
        (r'exceeded\s+(?:estimates?|expectations?|guidance)', 3.0),
        (r'surpassed\s+(?:estimates?|expectations?)', 3.0),
        (r'record\s+(?:profit|revenue|earnings|high)', 2.5),
        (r'(?:profit|revenue|earnings|income)\s+(?:up|rose|increased|jumped|surged)', 2.5),
        (r'(?:strong|robust|healthy)\s+(?:growth|results|performance)', 2.0),
        (r'(?:secured?|won|awarded?|bagged?)\s+(?:contract|order|deal)', 2.5),
        (r'(?:upgrade|upgraded)\s+(?:to|by|rating)', 2.0),
        (r'(?:declared?|announced?)\s+(?:dividend|bonus|buyback)', 2.0),
        (r'(?:expansion|capacity\s+addition|new\s+plant)', 1.5),
        (r'(?:partnership|alliance|tie-up|collaboration)', 1.5),
        (r'(?:approval|clearance)\s+(?:received|granted|obtained)', 2.0),

        # Moderate positives
        (r'\b(?:bullish|optimistic|positive)\b', 1.5),
        (r'\b(?:outperform|buy|accumulate)\b', 1.5),
        (r'\bgrowth\b', 1.0),
        (r'\bprofit\b', 0.5),
    ]

    NEGATIVE_PATTERNS = [
        # Strong negatives
        (r'missed\s+(?:estimates?|expectations?|guidance|forecast)', -3.0),
        (r'below\s+(?:estimates?|expectations?|guidance)', -2.5),
        (r'(?:profit|revenue|earnings|income)\s+(?:down|fell|declined|dropped|slumped)', -2.5),
        (r'(?:downgrade|downgraded)\s+(?:to|by|rating)', -2.0),
        (r'(?:fraud|scam|irregularities|manipulation)', -3.0),
        (r'(?:penalty|fine)\s+(?:of|worth|imposed)', -2.0),
        (r'(?:lawsuit|litigation|legal\s+action)', -2.0),
        (r'(?:investigation|probe|inquiry)\s+(?:into|by)', -2.0),
        (r'(?:default|bankruptcy|insolvency)', -3.0),
        (r'(?:resignation|stepped\s+down)\s+(?:of|from)\s+(?:ceo|cfo|md|director)', -2.0),

        # Moderate negatives
        (r'\b(?:bearish|pessimistic|negative)\b', -1.5),
        (r'\b(?:underperform|sell|reduce)\b', -1.5),
        (r'\b(?:weak|disappointing|poor)\b', -1.5),
        (r'\bloss(?:es)?\b', -1.0),
        (r'\bdecline\b', -0.5),
    ]

    # Context modifiers
    NEGATION_WORDS = ['not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', "n't", 'without']
    INTENSIFIERS = ['very', 'extremely', 'significantly', 'substantially', 'greatly', 'sharply']
    DIMINISHERS = ['slightly', 'marginally', 'somewhat', 'little', 'modest', 'minor']

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment using keyword patterns."""
        text_lower = text.lower()

        positive_score = 0.0
        negative_score = 0.0
        matches = []

        # Check positive patterns
        for pattern, weight in self.POSITIVE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                positive_score += weight
                matches.append(('positive', pattern, weight))

        # Check negative patterns
        for pattern, weight in self.NEGATIVE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                negative_score += abs(weight)
                matches.append(('negative', pattern, weight))

        # Check for negation (reverses sentiment)
        has_negation = any(word in text_lower.split() for word in self.NEGATION_WORDS)

        # Check for intensifiers/diminishers
        has_intensifier = any(word in text_lower for word in self.INTENSIFIERS)
        has_diminisher = any(word in text_lower for word in self.DIMINISHERS)

        # Calculate final score
        if has_negation:
            # Swap scores if negation detected
            positive_score, negative_score = negative_score * 0.7, positive_score * 0.7

        if has_intensifier:
            positive_score *= 1.3
            negative_score *= 1.3
        elif has_diminisher:
            positive_score *= 0.7
            negative_score *= 0.7

        # Normalize to -1 to 1 range
        total = positive_score + negative_score
        if total > 0:
            compound = (positive_score - negative_score) / (total + 1)
        else:
            compound = 0.0

        # Clamp to [-1, 1]
        compound = max(-1.0, min(1.0, compound))

        # Determine label
        if compound >= 0.4:
            label = SentimentLabel.VERY_POSITIVE
        elif compound >= 0.15:
            label = SentimentLabel.POSITIVE
        elif compound <= -0.4:
            label = SentimentLabel.VERY_NEGATIVE
        elif compound <= -0.15:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        # Confidence based on how many patterns matched
        confidence = min(len(matches) * 0.2, 0.8) if matches else 0.3

        return SentimentResult(
            label=label,
            score=compound,
            confidence=confidence,
            method="Keyword",
            details={
                'positive_score': positive_score,
                'negative_score': negative_score,
                'matches': len(matches),
                'has_negation': has_negation,
            }
        )


class FinancialSentimentAnalyzer:
    """
    Main sentiment analyzer with automatic fallback.

    Tries analyzers in order of accuracy:
    1. FinBERT (if transformers installed)
    2. VADER (if nltk installed)
    3. Keyword matching (always available)

    Usage:
        analyzer = FinancialSentimentAnalyzer()
        result = analyzer.analyze("Q3 profit beat estimates by 15%")
        print(result.label)  # POSITIVE
        print(result.score)  # 0.85
    """

    def __init__(self, prefer_finbert: bool = True, prefer_vader: bool = True):
        """
        Initialize the sentiment analyzer.

        Args:
            prefer_finbert: Try FinBERT first (requires transformers)
            prefer_vader: Try VADER if FinBERT unavailable (requires nltk)
        """
        self.finbert = FinBERTAnalyzer() if prefer_finbert else None
        self.vader = VADERAnalyzer() if prefer_vader else None
        self.keyword = KeywordAnalyzer()

        self._active_method = None

    def get_available_methods(self) -> List[str]:
        """Get list of available analysis methods."""
        methods = []
        if self.finbert and self.finbert.is_available():
            methods.append("FinBERT")
        if self.vader and self.vader.is_available():
            methods.append("VADER")
        methods.append("Keyword")
        return methods

    def analyze(self, text: str, method: str = None) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            method: Force specific method ("FinBERT", "VADER", "Keyword")

        Returns:
            SentimentResult with label, score, confidence
        """
        if not text or not text.strip():
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method="None"
            )

        # Clean text
        text = text.strip()

        # If specific method requested
        if method:
            if method.upper() == "FINBERT" and self.finbert:
                result = self.finbert.analyze(text)
                if result:
                    return result
            elif method.upper() == "VADER" and self.vader:
                result = self.vader.analyze(text)
                if result:
                    return result
            elif method.upper() == "KEYWORD":
                return self.keyword.analyze(text)

        # Try methods in order of preference
        if self.finbert and self.finbert.is_available():
            result = self.finbert.analyze(text)
            if result:
                self._active_method = "FinBERT"
                return result

        if self.vader and self.vader.is_available():
            result = self.vader.analyze(text)
            if result:
                self._active_method = "VADER"
                return result

        # Fallback to keyword analysis
        self._active_method = "Keyword"
        return self.keyword.analyze(text)

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]

    def analyze_with_context(self, headline: str, content: str = None) -> SentimentResult:
        """
        Analyze with headline weighted more heavily.

        Headlines often carry the most sentiment signal.
        """
        headline_result = self.analyze(headline)

        if not content:
            return headline_result

        content_result = self.analyze(content)

        # Weighted average: headline 60%, content 40%
        combined_score = (headline_result.score * 0.6) + (content_result.score * 0.4)
        combined_confidence = (headline_result.confidence * 0.6) + (content_result.confidence * 0.4)

        # Determine label from combined score
        if combined_score >= 0.4:
            label = SentimentLabel.VERY_POSITIVE
        elif combined_score >= 0.15:
            label = SentimentLabel.POSITIVE
        elif combined_score <= -0.4:
            label = SentimentLabel.VERY_NEGATIVE
        elif combined_score <= -0.15:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        return SentimentResult(
            label=label,
            score=combined_score,
            confidence=combined_confidence,
            method=f"{headline_result.method}+weighted",
            details={
                'headline_score': headline_result.score,
                'content_score': content_result.score,
            }
        )


# Convenience function
def analyze_sentiment(text: str) -> SentimentResult:
    """Quick sentiment analysis with default settings."""
    analyzer = FinancialSentimentAnalyzer()
    return analyzer.analyze(text)


# Example usage and testing
if __name__ == "__main__":
    # Test sentences
    test_cases = [
        "Q3 profit beat analyst estimates by 15%",
        "Revenue declined 10% YoY due to weak demand",
        "Company announces dividend of Rs 10 per share",
        "SEBI investigation into alleged accounting fraud",
        "Profit declined less than expected, shares rally",
        "Strong order book growth of 25% YoY",
        "Management guidance cut for FY25",
        "Board approves Rs 500 crore expansion plan",
    ]

    analyzer = FinancialSentimentAnalyzer()
    print(f"Available methods: {analyzer.get_available_methods()}")
    print()

    for text in test_cases:
        result = analyzer.analyze(text)
        print(f"Text: {text[:60]}...")
        print(f"  Sentiment: {result.label.value} (score: {result.score:.2f})")
        print(f"  Confidence: {result.confidence:.2f}, Method: {result.method}")
        print()
