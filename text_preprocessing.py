
"""Text preprocessing for Verifier Lense."""

import re


def clean_text(text):
    """Apply the preprocessing sequence used during model training."""
    text = str(text)

    text = re.sub(r'https?://\s+|www\.\s+', '', text)

    text = re.sub(r'\breuters\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bgetty images\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfeatured image\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\btwitter\.com\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpic\.twitter\b', ' ', text, flags=re.IGNORECASE)

    text = re.sub(r'\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
