import re
import unicodedata
from unidecode import unidecode


def normalize_text(text: str) -> str:
    """Normalize text for consistent search and comparison.

    Performs unicode normalization, transliteration to ASCII,
    lowercasing, punctuation removal, and whitespace cleanup.

    Args:
        text (str): Source text to normalize.

    Returns:
        str: Normalized text string.
    """

    if text is None:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Transliteration
    text = unidecode(text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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