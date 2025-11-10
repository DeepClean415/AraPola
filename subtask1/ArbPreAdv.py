from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer
from typing import List, Optional, Set
from ArbPreBasic import ArabicBasicPreprocessor

"""
Advanced Arabic Text Preprocessing Module

This module provides advanced preprocessing for Arabic text using CAMeL Tools,
with selective morphological segmentation strategies optimized for NLP tasks.
"""


class ArabicAdvancedPreprocessor:
    """
    Advanced Arabic text preprocessor using CAMeL Tools.
    
    Supports selective morphological segmentation, light stemming, and lemmatization
    with configurable segmentation strategies for optimal NLP performance.
    """
    
    def __init__(
        self,
        split_proclitics: Optional[Set[str]] = None,
        split_enclitics: Optional[Set[str]] = None,
        keep_definite_article: bool = True,
        keep_particles: bool = True,
        use_light_stemming: bool = False,
        use_lemmatization: bool = False,
        use_basic_preprocessing: bool = True
    ):
        """
        Initialize the advanced preprocessor.
        
        Args:
            split_proclitics: Set of proclitic types to split (e.g., {'CONJ', 'PREP'})
            split_enclitics: Set of enclitic types to split (default: {'PRON'} for pronouns)
            keep_definite_article: Keep definite article (Al+) attached (recommended: True)
            keep_particles: Keep particle proclitics (PART+) attached (recommended: True)
            use_light_stemming: Apply light stemming
            use_lemmatization: Apply lemmatization
            use_basic_preprocessing: Apply basic preprocessing (normalization, diacritics removal)
        """
        # Initialize basic preprocessor if needed
        self.basic_preprocessor = ArabicBasicPreprocessor() if use_basic_preprocessing else None
        
        # Initialize CAMeL Tools components
        self.db = MorphologyDB.builtin_db()
        self.analyzer = Analyzer(self.db)
        self.disambiguator = MLEDisambiguator(self.analyzer)
        
        # Default: split pronominal enclitics only (performance-driven strategy)
        self.split_enclitics = split_enclitics or {'PRON'}
        self.split_proclitics = split_proclitics or set()
        
        self.keep_definite_article = keep_definite_article
        self.keep_particles = keep_particles
        self.use_light_stemming = use_light_stemming
        self.use_lemmatization = use_lemmatization
        
        # Initialize tokenizer with custom scheme
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Setup morphological tokenizer with custom segmentation scheme."""
        scheme = self._create_segmentation_scheme()
        self.tokenizer = MorphologicalTokenizer(
            self.disambiguator,
            scheme=scheme,
            split=True  # Return list of segments instead of underscore-delimited string
        )
    
    def _create_segmentation_scheme(self) -> str:
        """
        Create a custom segmentation scheme based on configuration.
        
        Returns:
            Segmentation scheme string for MorphologicalTokenizer
        """
        if not self.split_proclitics and not self.split_enclitics:
            return 'd1seg'  # Minimal segmentation
        
        return 'd2seg'  # Medium segmentation for selective approach
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess Arabic text with advanced methods.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Preprocessed text
        """
        if not text or not text.strip():
            return ""
        
        # Apply basic preprocessing if configured
        if self.basic_preprocessor:
            text = self.basic_preprocessor.preprocess(text)
        
        # Tokenize and process
        tokens = self._process_tokens(text)
        
        return " ".join(tokens)
    
    def _process_tokens(self, text: str) -> List[str]:
        """
        Process tokens with morphological analysis.
        
        Args:
            text: Input text
            
        Returns:
            List of processed tokens
        """
        # Split text into words
        words = text.split()
        all_tokens = []
        
        for word in words:
            # Tokenize returns a flat list of segments when split=True
            segmented_tokens = self.tokenizer.tokenize([word])
            
            # Process each segment
            for token in segmented_tokens:
                if self.use_lemmatization:
                    processed_token = self._get_lemma(token)
                elif self.use_light_stemming:
                    processed_token = self._apply_light_stem(token)
                else:
                    processed_token = token
                
                all_tokens.append(processed_token)
        
        return all_tokens
    
    def _get_lemma(self, token: str) -> str:
        """
        Get lemmatized form of token.
        
        Args:
            token: Input token
            
        Returns:
            Lemmatized token
        """
        analyses = self.analyzer.analyze(token)
        if analyses:
            return analyses[0].get('lex', token)
        return token
    
    def _apply_light_stem(self, token: str) -> str:
        """
        Apply light stemming to token.
        
        Args:
            token: Input token
            
        Returns:
            Light-stemmed token
        """
        analyses = self.analyzer.analyze(token)
        if analyses:
            return analyses[0].get('stem', token)
        return token
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def get_morphological_features(self, text: str) -> List[dict]:
        """
        Extract morphological features from text.
        
        Args:
            text: Input text
            
        Returns:
            List of morphological feature dictionaries per token
        """
        # disambiguate expects a list of words (tokens), not a list of lists
        words = text.split()
        disambiguated = self.disambiguator.disambiguate(words)
        # Extract the analysis dict from each DisambiguatedWord
        return [token.analyses[0].analysis for token in disambiguated]


# Example usage
if __name__ == "__main__":
    preprocessor = ArabicAdvancedPreprocessor(
        split_enclitics={'PRON'},
        keep_definite_article=True,
        keep_particles=True,
        use_lemmatization=False,
        use_basic_preprocessing=True
    )
    
    sample_text = "كتابهم جميل والمعلمون يدرسون في المدرسة"
    processed = preprocessor.preprocess(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
