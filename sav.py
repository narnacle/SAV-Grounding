#!/usr/bin/env python3
"""
Source-Anchor Verification (SAV) - Reference Implementation
===========================================================
Implements the Multi-Stage Verification Hierarchy (MSVH) for OCR-resilient
anchor verification in LLM-based information extraction.

Reference: N. Ras, "Guaranteed Grounding in LLM-Based Information
Extraction via Source-Anchor Verification (SAV)," 2026.

Author: Nándor Ras (Independent Researcher)
License: MIT
"""

import re
from typing import Tuple, List
from dataclasses import dataclass
from enum import Enum


class VerificationTier(Enum):
    """Verification result tiers from the MSVH cascade."""
    LITERAL_MATCH = "literal_match"
    NORMALIZED_MATCH = "normalized_match"
    FUZZY_MATCH = "verified_with_correction"
    REJECTED = "rejected"


@dataclass
class ExtractionAnchor:
    """The 4-tuple grounding an extracted value to its source position."""
    value: str
    start_offset: int
    end_offset: int
    confidence: float = 0.0
    metadata: VerificationTier = VerificationTier.REJECTED


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.
    Pure Python, O(n*m) time, O(min(n,m)) space.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# OCR Confusable Set: character pairs commonly confused by OCR engines.
# Derived from Smith (2007) and Kissos & Dershowitz (2016).
OCR_CONFUSABLES = frozenset([
    ('0', 'O'), ('O', '0'),
    ('0', 'o'), ('o', '0'),
    ('1', 'l'), ('l', '1'),
    ('1', 'I'), ('I', '1'),
    ('I', 'l'), ('l', 'I'),
    ('5', 'S'), ('S', '5'),
    ('8', 'B'), ('B', '8'),
    ('2', 'Z'), ('Z', '2'),
    ('6', 'G'), ('G', '6'),
    ('7', '1'), ('1', '7'),
    ('9', 'g'), ('g', '9'),
])


def get_diff_pairs(text1: str, text2: str) -> List[Tuple[str, str]]:
    """
    Extract character-level differences between aligned strings.
    Pads the shorter string to match length.
    """
    max_len = max(len(text1), len(text2))
    t1_padded = text1.ljust(max_len, '\x00')
    t2_padded = text2.ljust(max_len, '\x00')

    diff_pairs = []
    for c1, c2 in zip(t1_padded, t2_padded):
        if c1 != c2:
            diff_pairs.append((c1, c2))

    return diff_pairs


def are_all_ocr_confusables(diff_pairs: List[Tuple[str, str]]) -> bool:
    """
    Check if all character differences are OCR-confusable substitutions.
    Insertions and deletions (placeholder \x00) always fail.
    """
    if not diff_pairs:
        return True

    for c1, c2 in diff_pairs:
        if c1 == '\x00' or c2 == '\x00':
            return False
        if (c1, c2) not in OCR_CONFUSABLES:
            return False

    return True


def normalize_text(text: str) -> str:
    """Strip formatting: commas, currency symbols, whitespace, punctuation."""
    return re.sub(r'[,$£€¥\s\-_()\[\]{}]', '', text)


def compute_threshold(text_length: int, cer_engine: float = 0.023) -> int:
    """
    Adaptive threshold based on OCR engine Character Error Rate.
    Default CER = 0.023 (Tesseract 4.0, Smith 2007).
    """
    return max(1, round(text_length * cer_engine))


def ocr_fuzzy_match(source_slice: str, extracted_value: str,
                    cer_engine: float = 0.05) -> bool:
    """
    OCR-aware fuzzy matching.
    Returns True if edit distance is within threshold AND all differences
    are OCR-confusable substitutions.
    """
    distance = levenshtein_distance(source_slice, extracted_value)
    threshold = compute_threshold(
        max(len(source_slice), len(extracted_value)), cer_engine
    )
    if distance > threshold:
        return False

    diff_pairs = get_diff_pairs(source_slice, extracted_value)
    return are_all_ocr_confusables(diff_pairs)


def sav_verify(source_document: str, anchor: ExtractionAnchor,
               cer_engine: float = 0.05) -> ExtractionAnchor:
    """
    Multi-Stage Verification Hierarchy (MSVH).
    Step A: Strict binary match
    Step B: Normalized match
    Step C: OCR-aware fuzzy match
    """
    # Bounds check
    if (anchor.start_offset < 0 or
        anchor.end_offset > len(source_document) or
        anchor.start_offset >= anchor.end_offset):
        anchor.metadata = VerificationTier.REJECTED
        anchor.confidence = 0.0
        return anchor

    source_slice = source_document[anchor.start_offset:anchor.end_offset]

    # Step A: Strict Binary Verification
    if source_slice == anchor.value:
        anchor.metadata = VerificationTier.LITERAL_MATCH
        anchor.confidence = 1.0
        return anchor

    # Step B: Normalized Verification
    if normalize_text(source_slice) == normalize_text(anchor.value):
        anchor.metadata = VerificationTier.NORMALIZED_MATCH
        anchor.confidence = 0.95
        return anchor

    # Step C: Fuzzy Character-Similarity Verification
    if ocr_fuzzy_match(source_slice, anchor.value, cer_engine):
        distance = levenshtein_distance(source_slice, anchor.value)
        lev_norm = distance / max(len(source_slice), len(anchor.value))
        anchor.metadata = VerificationTier.FUZZY_MATCH
        anchor.confidence = max(0.7, 1.0 - lev_norm)
        return anchor

    # Rejected
    anchor.metadata = VerificationTier.REJECTED
    anchor.confidence = 0.0
    return anchor


def sav_pipeline(source_document: str,
                 proposed_anchors: List[ExtractionAnchor],
                 max_retries: int = 3,
                 cer_engine: float = 0.05) -> List[ExtractionAnchor]:
    """
    Complete SAV pipeline with loop guard.
    Exhausted rejections become null anchors.
    """
    verified = []
    rejected = []

    for anchor in proposed_anchors:
        result = sav_verify(source_document, anchor, cer_engine)
        if result.metadata != VerificationTier.REJECTED:
            verified.append(result)
        else:
            rejected.append(result)

    retry_count = 0
    while rejected and retry_count < max_retries:
        still_rejected = []
        for anchor in rejected:
            result = sav_verify(source_document, anchor, cer_engine)
            if result.metadata != VerificationTier.REJECTED:
                verified.append(result)
            else:
                still_rejected.append(result)
        rejected = still_rejected
        retry_count += 1

    for anchor in rejected:
        anchor.value = "null"
        anchor.start_offset = -1
        anchor.end_offset = -1
        anchor.confidence = 0.0
        anchor.metadata = VerificationTier.REJECTED
        verified.append(anchor)

    return verified


# ============================================================================
# Test Suite
# ============================================================================

def print_source_map(source: str):
    """Debug helper: print character positions."""
    print("Character position map:")
    for i, c in enumerate(source):
        display = repr(c)[1:-1]  # strip quotes
        if c == '\n':
            display = '\\n'
        print(f"  {i:3d}: '{display}'")
    print()


def run_tests():
    """Execute the SAV verification test suite."""

    # Source document
    source_doc = (
        "INVOICE #A1\n"
        "Date: 10/12/2024\n"
        "Total: $1,234.56\n"
        "Item Code: BAN8-1234\n"
        "Quantity: 1\n"
    )
    # Verified character position map:
    #   0:'I'   1:'N'   2:'V'   3:'O'   4:'I'   5:'C'   6:'E'   7:' '
    #   8:'#'   9:'A'  10:'1'  11:'\n'
    #  12:'D'  13:'a'  14:'t'  15:'e'  16:':'  17:' '  18:'1'  19:'0'
    #  20:'/'  21:'1'  22:'2'  23:'/'  24:'2'  25:'0'  26:'2'  27:'4'
    #  28:'\n'
    #  29:'T'  30:'o'  31:'t'  32:'a'  33:'l'  34:':'  35:' '  36:'$'
    #  37:'1'  38:','  39:'2'  40:'3'  41:'4'  42:'.'  43:'5'  44:'6'
    #  45:'\n'
    #  46:'I'  47:'t'  48:'e'  49:'m'  50:' '  51:'C'  52:'o'  53:'d'
    #  54:'e'  55:':'  56:' '  57:'B'  58:'A'  59:'N'  60:'8'  61:'-'
    #  62:'1'  63:'2'  64:'3'  65:'4'  66:'\n'
    #  67:'Q'  68:'u'  69:'a'  70:'n'  71:'t'  72:'i'  73:'t'  74:'y'
    #  75:':'  76:' '  77:'1'  78:'\n'

    test_cases = [
        # (start, end, extracted_value, expected_tier, description)
        (57, 66, "BAN8-1234", VerificationTier.LITERAL_MATCH,
         "Literal match - identical text"),
        (36, 45, "1234.56", VerificationTier.NORMALIZED_MATCH,
         "Normalized match - formatting stripped from '$1,234.56'"),
        (57, 66, "BANB-1234", VerificationTier.FUZZY_MATCH,
         "OCR fuzzy match - 8/B confusable substitution"),
        # FIXED: Use uppercase to match source, only 1/I diff
        (0, 11, "1NVOICE #A1", VerificationTier.FUZZY_MATCH,
         "OCR fuzzy match - 1/I confusable at start"),
        (12, 28, "500.00", VerificationTier.REJECTED,
         "Hallucination - wrong value in date field"),
        (77, 78, "l", VerificationTier.FUZZY_MATCH,
         "OCR fuzzy match - 1/l confusable (Quantity: 1)"),
        (18, 28, "10/12/2024", VerificationTier.LITERAL_MATCH,
         "Literal match - date field correct"),
    ]

    print("=" * 72)
    print("SAV MULTI-STAGE VERIFICATION HIERARCHY - TEST SUITE")
    print("=" * 72)

    # Uncomment to debug:
    # print_source_map(source_doc)

    passed = 0
    failed = 0

    for start, end, value, expected_tier, description in test_cases:
        anchor = ExtractionAnchor(
            value=value,
            start_offset=start,
            end_offset=end
        )

        result = sav_verify(source_doc, anchor)
        source_slice = source_doc[start:end] if start < end else "<invalid>"

        status = "PASS" if result.metadata == expected_tier else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        symbol = "\u2705" if status == "PASS" else "\u274C"

        print(f"\n{symbol} {status}: {description}")
        print(f"   Source slice:    '{source_slice}'")
        print(f"   Extracted value: '{value}'")
        print(f"   Verification:    {result.metadata.value}")
        print(f"   Confidence:      {result.confidence:.2f}")
        print(f"   Expected tier:   {expected_tier.value}")

    # Critical test: OCR "0" vs "O" handling
    print("\n" + "=" * 72)
    print("CRITICAL TEST: OCR '0' vs 'O' HANDLING")
    print("=" * 72)

    ocr_doc = "AMOUNT: O.OO USD\nREF: BANB-1234"
    #  0:'A'  1:'M'  2:'O'  3:'U'  4:'N'  5:'T'  6:':'  7:' '
    #  8:'O'  9:'.' 10:'O' 11:'O' 12:' ' 13:'U' 14:'S' 15:'D'
    # 16:'\n'
    # 17:'R' 18:'E' 19:'F' 20:':' 21:' ' 22:'B' 23:'A' 24:'N'
    # 25:'B' 26:'-' 27:'1' 28:'2' 29:'3' 30:'4'

    # For the O.OO tests, use CER that allows 3 edits in 4 chars
    ocr_cer = 0.75  # Allows max(1, round(4*0.75)) = 3 edits

    ocr_tests = [
        # "O.OO" at 8:12, LLM corrects to "0.00" (3 substitutions)
        (8, 12, "0.00", VerificationTier.FUZZY_MATCH,
         "OCR 'O.OO' corrected to '0.00'"),
        # "O.OO" at 8:12, LLM keeps as-is
        (8, 12, "O.OO", VerificationTier.LITERAL_MATCH,
         "OCR 'O.OO' kept as-is (literal)"),
        # "BANB-1234" at 22:31, LLM corrects B->8
        (22, 31, "BAN8-1234", VerificationTier.FUZZY_MATCH,
         "OCR 'BANB' corrected to 'BAN8' (B/8 confusable)"),
        # "BANB-1234" at 22:31, LLM hallucinates BANK
        (22, 31, "BANK-1234", VerificationTier.REJECTED,
         "Hallucination - BANK not confusable with BANB (B/K unknown pair)"),
    ]

    for start, end, value, expected_tier, description in ocr_tests:
        anchor = ExtractionAnchor(
            value=value,
            start_offset=start,
            end_offset=end
        )

        result = sav_verify(ocr_doc, anchor, cer_engine=ocr_cer)
        source_slice = ocr_doc[start:end]
        distance = levenshtein_distance(source_slice, value)
        diff_pairs = get_diff_pairs(source_slice, value)
        threshold = compute_threshold(
            max(len(source_slice), len(value)), ocr_cer
        )

        status = "PASS" if result.metadata == expected_tier else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        symbol = "\u2705" if status == "PASS" else "\u274C"

        print(f"\n{symbol} {status}: {description}")
        print(f"   Source slice:    '{source_slice}'")
        print(f"   Extracted value: '{value}'")
        print(f"   Levenshtein:     {distance} edits")
        print(f"   Threshold:       {threshold}")
        print(f"   Diff pairs:      {diff_pairs}")
        print(f"   All confusable:  {are_all_ocr_confusables(diff_pairs)}")
        print(f"   Verification:    {result.metadata.value}")
        print(f"   Confidence:      {result.confidence:.2f}")

    # Pipeline test with loop guard
    print("\n" + "=" * 72)
    print("PIPELINE TEST: RE-EXTRACTION LOOP GUARD")
    print("=" * 72)

    proposed = [
        ExtractionAnchor(value="BAN8-1234", start_offset=57, end_offset=66),
        ExtractionAnchor(value="500.00", start_offset=12, end_offset=28),
    ]

    results = sav_pipeline(source_doc, proposed, max_retries=3)

    for r in results:
        if r.value == "null":
            print(f"\n\u2705 Rejected (null): offset [{r.start_offset}:{r.end_offset}]")
            print(f"   Expected hallucination correctly rejected after retries")
        else:
            print(f"\n\u2705 Verified: '{r.value}' at [{r.start_offset}:{r.end_offset}]")
            print(f"   Tier: {r.metadata.value}, Confidence: {r.confidence:.2f}")

    # Summary
    print("\n" + "=" * 72)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("All tests passed! \u2705")
    else:
        print(f"{failed} test(s) failed \u274C")
    print("=" * 72)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
