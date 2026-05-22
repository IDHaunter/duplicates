import pytest

from dataset_processor import normalize_text

# =========================================================
# Basic normalization
# =========================================================

def test_empty_string():
    assert normalize_text("") == ""


def test_none_like_behavior():
    assert normalize_text(None) == ""


def test_lowercase_and_punctuation():
    result = normalize_text("Hello, World!")
    assert result == "hello world"


# =========================================================
# Unicode + transliteration
# =========================================================

def test_transliteration():
    result = normalize_text("Żółć")
    assert result == "zolc"


def test_unicode_normalization():
    result = normalize_text("Ａpple")
    assert result == "apple"


# =========================================================
# Symbol replacement
# =========================================================

def test_ampersand_replacement():
    result = normalize_text("Research & Development")
    assert result == "and development research"


# =========================================================
# Number removal
# =========================================================

def test_remove_number_tokens():
    result = normalize_text("Model 123 Version 456")
    assert result == "model version"


def test_keep_alphanumeric_tokens():
    result = normalize_text("A1 B2")
    assert result == "a1 b2"


# =========================================================
# Company suffix removal
# =========================================================

def test_company_suffixes_removed():
    result = normalize_text("Acme Sp. z o.o.")

    # assuming:
    # TOKEN_REPLACEMENTS["sp"] -> removed by suffix list
    # COMPANY_SUFFIXES contains:
    # "sp", "z", "oo", etc.

    assert result == "acme"


# =========================================================
# Duplicate removal
# =========================================================

def test_duplicate_tokens_removed():
    result = normalize_text("solution solution platform")
    assert result == "platform solution"


def test_duplicate_tokens_preserve_order_without_sort():
    result = normalize_text(
        "solution solution platform",
        sort_tokens=False
    )

    assert result == "solution platform"


# =========================================================
# Sorting behavior
# =========================================================

def test_sort_tokens_enabled_by_default():
    result = normalize_text("iphone apple")
    assert result == "apple iphone"


def test_sort_tokens_disabled():
    result = normalize_text(
        "iphone apple",
        sort_tokens=False
    )

    assert result == "iphone apple"


# =========================================================
# Short token removal
# =========================================================

def test_short_tokens_removed():
    result = normalize_text("a b cd ef")
    assert result == "cd ef"


# =========================================================
# Token replacements
# =========================================================

def test_token_replacements():

    result = normalize_text(
        text = "Acme Solutions Technologies", sort_tokens = True
    )

    assert result == "acme solution technology"


# =========================================================
# Complex real-world example
# =========================================================

def test_real_world_company_name():

    result = normalize_text(
        "Vistula Logistics Solutions Sp. z o.o."
    )

    assert result == "logistic solution vistula"


# =========================================================
# Multiple spaces cleanup
# =========================================================

def test_multiple_spaces():
    result = normalize_text("apple     iphone")
    assert result == "apple iphone"


# =========================================================
# Mixed punctuation
# =========================================================

def test_mixed_punctuation():
    result = normalize_text("Apple-iPhone_(Pro)")
    assert result == "apple iphone pro"