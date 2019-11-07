import pytest

from src.utils.nlp import (
    tokenize,
)

def test_tokenize():
    assert ['when', 'the', 'hous', 'light', 'all', 'go', 'dark', '.', 'shuffl', 'on', 'down', 'to', 'the', 'park'] == tokenize("When the house lights all go dark. Shuffle on down to the park")