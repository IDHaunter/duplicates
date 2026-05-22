import re
import unicodedata
from unidecode import unidecode


# =========================================================
# Common company/legal suffixes
# =========================================================

COMPANY_SUFFIXES = {

    # English
    "llc",
    "ltd",
    "inc",
    "corp",
    "corporation",
    "company",
    "co",

    # German
    "gmbh",
    "ag",

    # Scandinavian
    "as",
    "ab",
    "oy",

    # Polish
    "sp",
    "z",
    "oo",
    "zoo",
    "sa",

    # French
    "sarl",
    "sas",

    # Dutch
    "bv",
    "nv",

    # Italian
    "srl",

    # Spanish / Portuguese
    "sl",
    "lda",

    # Russian / CIS transliterated
    "ooo",
    "zao",
    "oao",
    "ao",

    # Generic
    "holding",
    "group"
}


# =========================================================
# Stop words often useless for dedup
# =========================================================

GENERIC_BUSINESS_WORDS = {
    "international",
    "global",
    "services",
    "solutions",
    "systems"
}


# =========================================================
# Token-level replacements
# =========================================================

TOKEN_REPLACEMENTS = {

    # Common morphology simplification
    "technologies": "technology",
    "solutions": "solution",
    "systems": "system",
    "logistics": "logistic",

    # Normalize ampersands
    "&": "and",
}


# =========================================================
# Main normalization function
# =========================================================

def normalize_text(text: str, sort_tokens: bool = True) -> str:
    """
    Advanced normalization for entity deduplication.

    Args:
        text:
            Input text to normalize.

        sort_tokens:
            If True (default), tokens are alphabetically sorted.
            Useful for entity deduplication where token order
            should not matter.

            Example:
                "Apple iPhone" == "iPhone Apple"

            If False, original token order is preserved.

    Steps:
        1. Unicode normalization
        2. Transliteration
        3. Lowercase
        4. Remove punctuation
        5. Token cleanup
        6. Remove legal suffixes
        7. Remove duplicates
        8. Sort tokens (optional)

    Example:

        "Vistula Logistics Solutions Sp. z o.o."

    becomes:

        "logistic solution vistula"
    """

    if not text:
        return ""

    # =====================================================
    # Unicode normalization
    # =====================================================

    text = unicodedata.normalize("NFKC", text)

    # =====================================================
    # Transliteration to ASCII
    # =====================================================

    text = unidecode(text)

    # =====================================================
    # Lowercase
    # =====================================================

    text = text.lower()

    # =====================================================
    # Replace special symbols
    # =====================================================

    text = text.replace("&", " and ")

    # =====================================================
    # Remove punctuation
    # =====================================================

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # =====================================================
    # Remove numbers-only tokens
    #
    # Optional:
    # remove if numbers are not meaningful
    # =====================================================

    text = re.sub(r"\b\d+\b", " ", text)

    # =====================================================
    # Collapse spaces
    # =====================================================

    text = re.sub(r"\s+", " ", text).strip()

    # =====================================================
    # Split into tokens
    # =====================================================

    tokens = text.split()

    # =====================================================
    # Apply token replacements
    # =====================================================

    normalized_tokens = []

    for token in tokens:

        token = TOKEN_REPLACEMENTS.get(token, token)

        normalized_tokens.append(token)

    tokens = normalized_tokens

    # =====================================================
    # Remove company suffixes
    # =====================================================

    tokens = [
        t for t in tokens
        if t not in COMPANY_SUFFIXES
    ]

    # =====================================================
    # Remove very short tokens
    #
    # Prevent noise:
    # "x", "a", "1"
    # =====================================================

    tokens = [
        t for t in tokens
        if len(t) > 1
    ]

    # =====================================================
    # Remove duplicated tokens
    #
    # Keeps original order
    #
    # Example:
    # "solution solution"
    # =====================================================

    tokens = list(dict.fromkeys(tokens))

    # =====================================================
    # Sort tokens (optional)
    #
    # HUGE improvement for:
    #
    # "Apple iPhone"
    # "iPhone Apple"
    # =====================================================

    if sort_tokens:
        tokens.sort()

    # =====================================================
    # Final normalized string
    # =====================================================

    return " ".join(tokens)


def build_analysis_records(
    data: dict,
    key_fields: list[str],
    analysis_fields: list[str],
    pk_separator: str = "|",
    field_separator: str = ";"
) -> list[dict]:
    """Build analysis-ready records from structured input data.

    Creates a list of dictionaries containing:
    - `pk_source`: composite key built from key fields
    - `fields_source`: concatenated raw analysis field values
    - `fields_normalized`: concatenated normalized analysis field values

    Missing fields are treated as empty strings.

    Args:
        data (dict): Source dictionary containing the "items" list.
        key_fields (list[str]): List of fields used to build the composite key.
        analysis_fields (list[str]): List of fields used for analysis content.
        pk_separator (str, optional): Separator used for composite keys.
            Defaults to "|".
        field_separator (str, optional): Separator used for concatenated fields.
            Defaults to ";".

    Returns:
        list[dict]: List of processed analysis records.
    """

    result = []

    for item in data.get("items", []):

        # Build composite primary key
        pk_source = pk_separator.join(
            str(item.get(field, "") or "")
            for field in key_fields
        )

        # Collect raw analysis values
        analysis_values = [
            str(item.get(field, "") or "")
            for field in analysis_fields
        ]

        fields_source = field_separator.join(analysis_values)

        # Normalize analysis values
        normalized_values = [
            normalize_text(value)
            for value in analysis_values
        ]

        fields_normalized = field_separator.join(normalized_values)

        result.append({
            "pk_source": pk_source,
            "fields_source": fields_source,
            "fields_normalized": fields_normalized
        })

    return result