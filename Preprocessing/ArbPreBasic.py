import re

class ArabicPreprocessorBasic:
    """
    A class for preprocessing Arabic text.
    Includes character standardization and diacritic/tatweel removal.
    """
    
    # Arabic diacritics
    DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')
    
    # Tatweel character
    TATWEEL = '\u0640'
    
    # Character mappings for standardization
    ALEF_VARIANTS = {
        '\u0622': '\u0627',  # آ -> ا
        '\u0623': '\u0627',  # أ -> ا
        '\u0625': '\u0627',  # إ -> ا
        '\u0671': '\u0627',  # ٱ -> ا
    }
    
    HAMZA_VARIANTS = {
        '\u0624': '\u0621',  # ؤ -> ء
        '\u0626': '\u0621',  # ئ -> ء
    }
    
    YAA_VARIANTS = {
        '\u0649': '\u064A',  # ى -> ي
        '\u06CC': '\u064A',  # ی -> ي
    }
    
    def __init__(self):
        """Initialize the Arabic preprocessor."""
        self.all_char_mappings = {
            **self.ALEF_VARIANTS,
            **self.HAMZA_VARIANTS,
            **self.YAA_VARIANTS
        }
    
    def standardize_characters(self, text):
        """
        Standardize Arabic characters by unifying alef, hamza, and yaa variants.
        
        Args:
            text (str): Input Arabic text
            
        Returns:
            str: Text with standardized characters
        """
        for variant, standard in self.all_char_mappings.items():
            text = text.replace(variant, standard)
        return text


# Backwards-compatible alias: some modules/scripts import
# `ArabicBasicPreprocessor` from `ArbPreBasic`. Provide a thin alias
# so both names work without changing callers.
class ArabicBasicPreprocessor(ArabicPreprocessorBasic):
    """
    Compatibility alias for older import paths.
    Behaves identically to `ArabicPreprocessorBasic`.
    """
    pass
    
    def remove_diacritics(self, text):
        """
        Remove Arabic diacritics from text.
        
        Args:
            text (str): Input Arabic text
            
        Returns:
            str: Text without diacritics
        """
        return self.DIACRITICS.sub('', text)
    
    def remove_tatweel(self, text):
        """
        Remove tatweel character from text.
        
        Args:
            text (str): Input Arabic text
            
        Returns:
            str: Text without tatweel
        """
        return text.replace(self.TATWEEL, '')
    
    def remove_diacritics_and_tatweel(self, text):
        """
        Remove both diacritics and tatweel from text.
        
        Args:
            text (str): Input Arabic text
            
        Returns:
            str: Text without diacritics and tatweel
        """
        text = self.remove_diacritics(text)
        text = self.remove_tatweel(text)
        return text
    
    def preprocess(self, text, standardize=True, remove_diacritics=True, 
                   remove_tatweel=True):
        """
        Apply full preprocessing pipeline to Arabic text.
        
        Args:
            text (str): Input Arabic text
            standardize (bool): Whether to standardize characters
            remove_diacritics (bool): Whether to remove diacritics
            remove_tatweel (bool): Whether to remove tatweel
            
        Returns:
            str: Preprocessed text
        """
        if standardize:
            text = self.standardize_characters(text)
        if remove_diacritics:
            text = self.remove_diacritics(text)
        if remove_tatweel:
            text = self.remove_tatweel(text)
        return text