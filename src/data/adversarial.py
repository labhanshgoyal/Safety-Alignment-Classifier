import random
import string
from typing import Dict, List, Optional
import torch

_nlp = None
_marian_tokenizer = None
_marian_model = None

def get_spacy_nlp():
    """Lazily loads the spaCy model with graceful fallback."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_md")
        except Exception:
            _nlp = False
    return _nlp if _nlp is not False else None

def get_translation_models():
    """Lazily loads and caches the Helsinki MarianMT back-translation models."""
    global _marian_tokenizer, _marian_model
    if _marian_tokenizer is None or _marian_model is None:
        try:
            from transformers import MarianMTModel, MarianTokenizer
            model_name = "Helsinki-NLP/opus-mt-en-fr"
            _marian_tokenizer = MarianTokenizer.from_pretrained(model_name)
            _marian_model = MarianMTModel.from_pretrained(model_name)
        except Exception:
            _marian_tokenizer = False
            _marian_model = False
    if _marian_tokenizer is False or _marian_model is False:
        return None, None
    return _marian_tokenizer, _marian_model
KEYBOARD_MAP: Dict[str, str] = {
    'q': 'qw', 'w': 'weq', 'e': 'wer', 'r': 'ret', 't': 'tyr', 'y': 'ytu', 'u': 'uio', 'i': 'iop', 'o': 'op',
    'p': 'plo', 'a': 'as', 's': 'sad', 'd': 'dsf', 'f': 'fdg', 'g': 'gfh', 'h': 'hgj', 'j': 'jhk', 'k': 'kjl',
    'l': 'lk', 'z': 'zx', 'x': 'xzc', 'c': 'cvx', 'v': 'vbc', 'b': 'bnv', 'n': 'nbm', 'm': 'mn'
}
HOMOPHONE_MAP: Dict[str, str] = {
    'to': 'too', 'too': 'two', 'there': 'their', 'their': "they're", 'hear': 'here',
    'where': 'wear', 'know': 'no', 'new': 'knew', 'right': 'write', 'bare': 'bear'
}
def keyboard_typo(text: str, p: float = 0.1) -> str:
    """Simulates realistic QWERTY keyboard typos by swapping adjacent keys."""
    words = text.split()
    perturbed_words = []
    for word in words:
        if random.random() < p and len(word) > 1:
            i = random.randint(0, len(word) - 1)
            char_lower = word[i].lower()
            if char_lower in KEYBOARD_MAP:
                new_char = random.choice(KEYBOARD_MAP[char_lower])
                word = word[:i] + new_char + word[i + 1:]
        perturbed_words.append(word)
    return " ".join(perturbed_words)
def random_char_insertion(text: str, p: float = 0.1) -> str:
    """Inserts random ASCII characters inside words to evade tokenizers."""
    words = text.split()
    perturbed_words = []
    for word in words:
        if random.random() < p:
            i = random.randint(0, len(word))
            char = random.choice(string.ascii_lowercase)
            word = word[:i] + char + word[i:]
        perturbed_words.append(word)
    return " ".join(perturbed_words)
def synonym_replacement(text: str, p: float = 0.2, top_k: int = 5) -> str:
    """Replaces words with semantic synonyms using spaCy word vectors."""
    nlp = get_spacy_nlp()
    if nlp is None:
        return text
    words = text.split()
    perturbed_words = []
    for word in words:
        if random.random() < p:
            token = nlp(word)[0]
            if token.has_vector:
                similar_words = sorted(
                    [
                        (w.text, token.similarity(w))
                        for w in nlp.vocab
                        if w.has_vector and w.is_alpha and not w.is_stop and w.text != token.text and w.vector_norm > 0
                    ],
                    key=lambda item: item[1],
                    reverse=True,
                )
                if similar_words:
                    word = random.choice(similar_words[:top_k])[0]
        perturbed_words.append(word)
    return " ".join(perturbed_words)
def homophone_substitution(text: str, p: float = 0.2) -> str:
    """Substitutes sound-alike words (homophones) to test phonetic robustness."""
    words = text.split()
    perturbed_words = [HOMOPHONE_MAP.get(word.lower(), word) if random.random() < p else word for word in words]
    return " ".join(perturbed_words)
def word_order_shuffle(text: str, window_size: int = 2) -> str:
    """Locally perturbs word positions within a small sliding window."""
    words = text.split()
    n = len(words)
    if n < 2:
        return text
    shuffled_words = words.copy()
    for i in range(n - 1):
        if random.random() < 0.5:
            swap_idx = min(i + random.randint(1, window_size), n - 1)
            shuffled_words[i], shuffled_words[swap_idx] = shuffled_words[swap_idx], shuffled_words[i]
    return " ".join(shuffled_words)
def back_translation(text: str) -> str:
    """Paraphrases text via English -> French -> English round-trip translation."""
    tokenizer, model = get_translation_models()
    if tokenizer is None or model is None:
        return text
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception:
        return text
def apply_adversarial_perturbations(
    text: str,
    transform_prob: float = 0.4,
    apply_back_translation_flag: bool = False,
) -> str:
    """Applies all 6 perturbation vectors stochastically based on transform_prob.
    Args:
        text: Original clean string.
        transform_prob: Probability threshold for each perturbation step.
        apply_back_translation_flag: Whether to include back-translation (slower).
    Returns:
        Perturbed adversarial string.
    """
    if transform_prob <= 0:
        return text
    if random.random() < transform_prob:
        text = keyboard_typo(text, p=transform_prob)
    if random.random() < transform_prob:
        text = random_char_insertion(text, p=transform_prob)
    if random.random() < transform_prob:
        text = synonym_replacement(text, p=transform_prob)
    if random.random() < transform_prob:
        text = homophone_substitution(text, p=transform_prob)
    if random.random() < transform_prob:
        text = word_order_shuffle(text, window_size=2)
    if apply_back_translation_flag and random.random() < transform_prob:
        text = back_translation(text)
    return text