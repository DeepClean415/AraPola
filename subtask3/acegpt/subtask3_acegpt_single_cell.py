# ============================================================================
# MANIFESTATION CLASSIFIER - SINGLE CELL FOR KAGGLE (SUBTASK 3)
# Copy this entire cell into Kaggle and run it after loading your data
# ============================================================================

import pandas as pd
import numpy as np
import torch
import json
import re
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    model_name = "aubmindlab/aragpt2-medium"
    max_length = 1024
    temperature = 0.3  # Lowered for more consistent outputs
    top_p = 0.9
    max_new_tokens = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    eval_samples = 30

config = Config()
np.random.seed(config.seed)
torch.manual_seed(config.seed)

print(f"✓ Device: {config.device}")

# ============================================================================
# MANIFESTATION CONTEXT MAPPER
# ============================================================================

class ManifestationContextMapper:
    """Maps manifestation types to Arabic names and contextual descriptions."""
    
    def __init__(self):
        self.manifestation_contexts = {
            'stereotype': {
                'ar_name': 'القوالب النمطية',
                'context': 'القوالب النمطية هي تعميمات مبسطة عن مجموعة من الناس',
                'examples': ['كل X كسالى', 'دائماً Y يفعلون Z', 'المعروف عن هؤلاء']
            },
            'vilification': {
                'ar_name': 'التشويه والإهانة',
                'context': 'التشويه يشمل لغة مسيئة أو تشهيرية تهاجم شخص أو مجموعة',
                'examples': ['شتائم', 'إهانات', 'تحقير']
            },
            'dehumanization': {
                'ar_name': 'التجريد من الإنسانية',
                'context': 'التجريد من الإنسانية ينكر الصفات الإنسانية ويشبه الناس بالحيوانات',
                'examples': ['كلاب', 'حيوانات', 'وحوش', 'قطيع']
            },
            'extreme_language': {
                'ar_name': 'اللغة المتطرفة',
                'context': 'اللغة المتطرفة تشمل كلمات عدوانية أو تحريضية قوية',
                'examples': ['يجب القضاء على', 'لا يستحقون الحياة', 'يستحقون الموت']
            },
            'lack_of_empathy': {
                'ar_name': 'عدم التعاطف',
                'context': 'عدم التعاطف يتجاهل أو يستهزئ بمعاناة الآخرين',
                'examples': ['يستحقون ما حدث', 'لا أهتم', 'مشكلتهم']
            },
            'invalidation': {
                'ar_name': 'الإلغاء والإنكار',
                'context': 'الإلغاء ينكر شرعية وجهات نظر أو تجارب الآخرين',
                'examples': ['هذا غير صحيح', 'لا يحق لهم', 'رأيهم باطل']
            }
        }
    
    def get_ar_name(self, manifestation: str) -> str:
        """Get Arabic name for manifestation."""
        return self.manifestation_contexts.get(manifestation, {}).get('ar_name', manifestation)
    
    def get_context(self, manifestation: str) -> str:
        """Get context description for manifestation."""
        return self.manifestation_contexts.get(manifestation, {}).get('context', '')

# ============================================================================
# FEW-SHOT EXAMPLE BANK
# ============================================================================

class FewShotExampleBank:
    """Builds and manages few-shot examples for each manifestation type."""
    
    def __init__(self, df: pd.DataFrame, labels: List[str]):
        self.df = df
        self.labels = labels
        self.example_bank = self._build_example_bank()
    
    def _build_example_bank(self) -> Dict:
        """Build example bank with positive and negative samples."""
        bank = {label: {'positive': [], 'negative': []} for label in self.labels}
        
        for _, row in self.df.iterrows():
            text = row['text']
            if pd.isna(text) or len(text) > 200:
                continue
            
            for label in self.labels:
                label_val = row[label]
                if pd.isna(label_val):
                    continue
                
                # Collect positive examples
                if label_val == 1 and len(bank[label]['positive']) < 100:
                    bank[label]['positive'].append({'text': text, 'label': 1})
                
                # Collect negative examples
                elif label_val == 0 and len(bank[label]['negative']) < 100:
                    other_labels_sum = sum([row[l] for l in self.labels if pd.notna(row[l]) and l != label])
                    if other_labels_sum == 0:
                        bank[label]['negative'].append({'text': text, 'label': 0})
        
        return bank
    
    def get_few_shot_examples(self, manifestation: str, n: int = 2) -> List[Dict]:
        """Get balanced few-shot examples for a manifestation."""
        positive_examples = self.example_bank[manifestation]['positive']
        negative_examples = self.example_bank[manifestation]['negative']
        
        # Sample positive examples
        pos_sample = np.random.choice(
            len(positive_examples), 
            size=min(n, len(positive_examples)), 
            replace=False
        ) if len(positive_examples) > 0 else []
        
        # Sample negative examples
        neg_sample = np.random.choice(
            len(negative_examples), 
            size=min(n, len(negative_examples)), 
            replace=False
        ) if len(negative_examples) > 0 else []
        
        # Combine examples
        examples = []
        for idx in pos_sample:
            examples.append(positive_examples[idx])
        for idx in neg_sample:
            examples.append(negative_examples[idx])
        
        np.random.shuffle(examples)
        return examples

# ============================================================================
# IMPROVED PROMPTER
# ============================================================================

class ImprovedPrompter:
    """Creates culturally-aware prompts for manifestation detection."""
    
    def __init__(self, manifestation_mapper: ManifestationContextMapper):
        self.manifestation_mapper = manifestation_mapper
    
    def get_manifestation_hints(self, manifestation: str) -> str:
        """Get specific guidance for each manifestation to reduce false positives."""
        hints = {
            'stereotype': 'ملاحظة: يجب أن يحتوي على تعميم واضح عن مجموعة من الناس.',
            'vilification': 'ملاحظة: يجب أن يحتوي على شتائم أو إهانات مباشرة.',
            'dehumanization': 'ملاحظة: يجب أن يشبه الناس بالحيوانات أو ينكر إنسانيتهم.',
            'extreme_language': 'ملاحظة: يجب أن يحتوي على لغة عدوانية أو تحريضية قوية.',
            'lack_of_empathy': 'ملاحظة: يجب أن يتجاهل أو يستهزئ بمعاناة الآخرين.',
            'invalidation': 'ملاحظة: يجب أن ينكر شرعية وجهات نظر أو تجارب الآخرين.'
        }
        return hints.get(manifestation, '')
    
    def create_completion_prompt(self, text: str, manifestation: str, few_shot_examples: str = "") -> str:
        """Create completion-style prompt for the model."""
        ar_name = self.manifestation_mapper.get_ar_name(manifestation)
        hint = self.get_manifestation_hints(manifestation)
        
        prompt = f"""{few_shot_examples}{hint}

النص: "{text}"
السؤال: هل يحتوي على {ar_name}؟
الإجابة:"""
        
        return prompt
    
    def format_few_shot_with_reasoning(self, examples: List[Dict], manifestation: str) -> str:
        """Format few-shot examples with reasoning to teach the model."""
        if not examples:
            return ""
        
        ar_name = self.manifestation_mapper.get_ar_name(manifestation)
        prompt = ""
        
        for example in examples:
            text = example['text'][:80] + "..." if len(example['text']) > 80 else example['text']
            label = example['label']
            
            if label == 1:
                answer = f"نعم، يحتوي على {ar_name}"
            else:
                answer = f"لا، لا يحتوي على {ar_name}"
            
            prompt += f"""النص: "{text}"
السؤال: هل يحتوي على {ar_name}؟
الإجابة: {answer}

"""
        
        return prompt
    
    def parse_response_flexible(self, response: str) -> Tuple[int, str]:
        """Parse model response with balanced precision/recall."""
        response_lower = response.lower().strip()
        reasoning = response
        
        # Check first 100 characters
        first_part = response_lower[:100]
        
        # Explicit patterns (highest confidence)
        explicit_yes = ['نعم، يحتوي', 'نعم يحتوي', 'يحتوي بالفعل']
        explicit_no = ['لا، لا يحتوي', 'لا يحتوي', 'لا يوجد']
        
        # Check explicit patterns first
        for yes_phrase in explicit_yes:
            if yes_phrase in first_part[:50]:
                return 1, reasoning
        
        for no_phrase in explicit_no:
            if no_phrase in first_part[:50]:
                return 0, reasoning
        
        # Indicators with different weights
        strong_yes = ['نعم', 'يحتوي', 'يوجد']
        strong_no = ['لا،', 'لا.', 'لا يحتوي', 'لا يوجد']
        weak_yes = ['موجود', 'واضح']
        weak_no = ['غير', 'ليس']
        
        # Score calculation (focus on first 50 chars)
        first_50 = first_part[:50]
        yes_score = 0
        no_score = 0
        
        # Strong indicators in first 50 chars (high weight)
        yes_score += sum(4 for word in strong_yes if word in first_50)
        no_score += sum(4 for word in strong_no if word in first_50)
        
        # Weak indicators in first 50 chars (medium weight)
        yes_score += sum(2 for word in weak_yes if word in first_50)
        no_score += sum(2 for word in weak_no if word in first_50)
        
        # Strong indicators in rest of first 100 (lower weight)
        rest_50 = first_part[50:]
        yes_score += sum(1 for word in strong_yes if word in rest_50)
        no_score += sum(1 for word in strong_no if word in rest_50)
        
        # Decision with threshold
        if yes_score > no_score * 1.2:
            return 1, reasoning
        elif no_score > yes_score:
            return 0, reasoning
        else:
            return 0, reasoning  # Default to negative in ties

# ============================================================================
# MANIFESTATION CLASSIFIER
# ============================================================================

class ManifestationClassifier:
    """Multi-label classifier for polarization manifestations."""
    
    def __init__(self, model, tokenizer, manifestation_mapper, few_shot_bank, improved_prompter, labels):
        self.model = model
        self.tokenizer = tokenizer
        self.manifestation_mapper = manifestation_mapper
        self.few_shot_bank = few_shot_bank
        self.improved_prompter = improved_prompter
        self.labels = labels
    
    def classify_single_manifestation(self, text: str, manifestation: str, num_few_shot: int = 2) -> Dict:
        """Classify a single manifestation type for the given text."""
        try:
            # Get few-shot examples
            examples = self.few_shot_bank.get_few_shot_examples(manifestation, n=num_few_shot)
            few_shot_text = self.improved_prompter.format_few_shot_with_reasoning(examples, manifestation)
            
            # Create completion prompt
            prompt = self.improved_prompter.create_completion_prompt(text, manifestation, few_shot_text)
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt", 
                                   max_length=config.max_length, truncation=True).to(config.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    top_p=config.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], 
                                            skip_special_tokens=True)
            
            # Parse response
            prediction, reasoning = self.improved_prompter.parse_response_flexible(response)
            
            return {
                'manifestation': manifestation,
                'prediction': prediction,
                'reasoning': response
            }
            
        except Exception as e:
            return {'manifestation': manifestation, 'prediction': 0, 'reasoning': f"Error: {str(e)}"}
    
    def classify_text(self, text: str, num_few_shot: int = 2) -> Dict:
        """Classify all manifestation types for the given text."""
        results = {'text': text, 'predictions': {}, 'manifestation_details': {}}
        
        for manifestation in self.labels:
            manifestation_result = self.classify_single_manifestation(text, manifestation, num_few_shot)
            results['predictions'][manifestation] = manifestation_result['prediction']
            results['manifestation_details'][manifestation] = manifestation_result
        
        return results
    
    def batch_classify(self, texts: List[str], num_few_shot: int = 2, show_progress: bool = True) -> List[Dict]:
        """Classify multiple texts."""
        results = []
        iterator = tqdm(texts, desc="Classifying") if show_progress else texts
        
        for text in iterator:
            try:
                result = self.classify_text(text, num_few_shot)
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'text': text, 
                    'predictions': {label: 0 for label in self.labels}, 
                    'manifestation_details': {}
                })
        
        return results

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_manifestation_classification(df, labels, eval_samples=30):
    """
    Main function to run manifestation classification.
    
    Args:
        df: DataFrame with 'text' column and label columns
        labels: List of label names (manifestation types)
        eval_samples: Number of samples to evaluate
    
    Returns:
        Dictionary with results and metrics
    """
    
    print("="*70)
    print("MANIFESTATION CLASSIFIER (SUBTASK 3)")
    print("="*70)
    
    # Initialize components
    print("\n1. Initializing components...")
    manifestation_mapper = ManifestationContextMapper()
    few_shot_bank = FewShotExampleBank(df, labels)
    improved_prompter = ImprovedPrompter(manifestation_mapper)
    print("✓ Components initialized")
    
    # Load model
    print("\n2. Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        device_map="auto" if config.device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    if config.device == "cpu":
        model = model.to(config.device)
    
    model.eval()
    print("✓ Model loaded")
    
    # Initialize classifier
    print("\n3. Initializing classifier...")
    classifier = ManifestationClassifier(
        model=model,
        tokenizer=tokenizer,
        manifestation_mapper=manifestation_mapper,
        few_shot_bank=few_shot_bank,
        improved_prompter=improved_prompter,
        labels=labels
    )
    print("✓ Classifier ready")
    
    # Prepare evaluation data
    print("\n4. Preparing evaluation data...")
    df_labeled = df[(df[labels].notna().all(axis=1)) & (df[labels].sum(axis=1) > 0)].copy()
    eval_size = min(eval_samples, len(df_labeled))
    eval_df = df_labeled.sample(n=eval_size, random_state=config.seed)
    
    print(f"✓ Evaluation set: {len(eval_df)} samples")
    print("\nLabel distribution:")
    for label in labels:
        count = eval_df[label].sum()
        print(f"  {label}: {count} ({count/len(eval_df)*100:.1f}%)")
    
    # Run classification
    print(f"\n5. Running classification...")
    print(f"Processing {len(eval_df)} samples × {len(labels)} manifestations\n")
    
    eval_results = classifier.batch_classify(
        texts=eval_df['text'].tolist(),
        num_few_shot=3,
        show_progress=True
    )
    
    # Extract predictions
    y_true = []
    y_pred = []
    
    for idx, row in eval_df.iterrows():
        true_labels = [int(row[label]) for label in labels]
        y_true.append(true_labels)
    
    for result in eval_results:
        pred_labels = [result['predictions'][label] for label in labels]
        y_pred.append(pred_labels)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    hamming = hamming_loss(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    print("\nOverall Metrics:")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  F1 Micro: {f1_micro:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  F1 Samples: {f1_samples:.4f}")
    
    print("\nPer-Manifestation Metrics:")
    print(f"{'Manifestation':<22} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-"*70)
    
    per_class_metrics = {}
    for i, label in enumerate(labels):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        pos_support = int(y_true[:, i].sum())
        
        label_name = label.replace('_', ' ').title()
        print(f"{label_name:<22} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {pos_support}")
        
        per_class_metrics[label] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': pos_support
        }
    
    print("="*70)
    
    return {
        'classifier': classifier,
        'eval_df': eval_df,
        'eval_results': eval_results,
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': {
            'hamming_loss': float(hamming),
            'f1_micro': float(f1_micro),
            'f1_macro': float(f1_macro),
            'f1_samples': float(f1_samples),
            'per_class': per_class_metrics
        }
    }

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
# HOW TO USE IN KAGGLE:

# 1. Load your data
df = pd.read_csv('/kaggle/input/your-dataset/arb.csv')
labels = ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']

# Fill NaN values
for label in labels:
    if label in df.columns:
        df[label] = df[label].fillna(0).astype(int)

# 2. Run manifestation classification
results = run_manifestation_classification(df, labels, eval_samples=30)

# 3. Access results
classifier = results['classifier']
metrics = results['metrics']
print(f"Final F1 Macro: {metrics['f1_macro']:.4f}")

# 4. Test on new samples
test_text = "هؤلاء الناس كلهم كسالى ولا يستحقون أي احترام"
prediction = classifier.classify_text(test_text)
print(prediction['predictions'])
"""

print("\n" + "="*70)
print("✓ MANIFESTATION CLASSIFIER LOADED")
print("="*70)
print("\nTo use: results = run_manifestation_classification(df, labels, eval_samples=30)")
print("="*70)

# ============================================================================
# AUTO-EXECUTION
# ============================================================================

# Load your data (change the path!)
df = pd.read_csv('/kaggle/input/arbsubtask3/arb.csv')  # ← Update this path
labels = ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']

# Fill NaN values
for label in labels:
    if label in df.columns:
        df[label] = df[label].fillna(0).astype(int)

# Run classification
results = run_manifestation_classification(df, labels, eval_samples=30)

# Print F1 score
print(f"\n🎯 Final F1 Macro: {results['metrics']['f1_macro']:.4f}")

# Save results
results_df = results['eval_df'].copy()
y_pred = results['y_pred']
for i, label in enumerate(labels):
    results_df[f'{label}_pred'] = y_pred[:, i]

results_df.to_csv('/kaggle/working/predictions.csv', index=False)
print("✓ Predictions saved to /kaggle/working/predictions.csv")
