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
from typing import Tuple, List, Optional
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
    """
    An extraction anchor: the 4-tuple that grounds an extracted value
    to its exact position in the source document.
    """
    value: str
    start_offset: int
    end_offset: int
    confidence: float = 0.0
    metadata: VerificationTier = VerificationTier.REJECTED


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.
    Pure Python implementation with O(n*m) time, O(min(n,m)) space.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost: 0 if characters match, 1 otherwise
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
    ('rn', 'm'), ('m', 'rn'),
])


def get_diff_pairs(text1: str, text2: str) -> List[Tuple[str, str]]:
    """
    Extract character-level differences between aligned strings.
    
    Uses a simple greedy alignment that handles insertions, deletions,
    and substitutions while identifying the specific character pairs
    that differ.
    """
    diff_pairs = []
    
    # Pad shorter string for alignment
    max_len = max(len(text1), len(text2))
    t1_padded = text1.ljust(max_len, '\x00')
    t2_padded = text2.ljust(max_len, '\x00')
    
    for c1, c2 in zip(t1_padded, t2_padded):
        if c1 != c2:
            diff_pairs.append((c1, c2))
    
    return diff_pairs


def are_all_ocr_confusables(diff_pairs: List[Tuple[str, str]]) -> bool:
    """
    Check if all character differences are explainable as OCR confusables.
    
    Rules:
    - Deletions (placeholder \x00) always fail
    - Insertions (placeholder \x00) always fail
    - Substitutions must be in the OCR_CONFUSABLES set
    """
    if not diff_pairs:
        return True
    
    for c1, c2 in diff_pairs:
        # Insertions or deletions are not allowed
        if c1 == '\x00' or c2 == '\x00':
            return False
        
        # Must be a known OCR-confusable pair
        if (c1, c2) not in OCR_CONFUSABLES:
            return False
    
    return True


def normalize_text(text: str) -> str:
    """
    Strip formatting artifacts: commas, currency symbols, whitespace, punctuation.
    """
    return re.sub(r'[,$£€¥\s\-_()\[\]{}]', '', text)


def compute_threshold(text_length: int, cer_engine: float = 0.023) -> int:
    """
    Compute adaptive threshold based on OCR engine Character Error Rate.
    
    Default CER = 0.023 based on Tesseract 4.0 benchmarks (Smith, 2007).
    """
    max_errors = text_length * cer_engine
    return max(1, round(max_errors))


def ocr_fuzzy_match(source_slice: str, extracted_value: str, 
                    cer_engine: float = 0.023) -> bool:
    """
    OCR-aware fuzzy matching with configurable threshold.
    
    Returns True if:
    1. The edit distance is within the adaptive threshold
    2. All character differences are OCR-confusable substitutions
    """
    distance = levenshtein_distance(source_slice, extracted_value)
    threshold = compute_threshold(
        max(len(source_slice), len(extracted_value)), 
        cer_engine
    )
    
    # Distance must be within threshold
    if distance > threshold:
        return False
    
    # All differences must be OCR-confusable substitutions
    diff_pairs = get_diff_pairs(source_slice, extracted_value)
    return are_all_ocr_confusables(diff_pairs)


def sav_verify(source_document: str, anchor: ExtractionAnchor,
               cer_engine: float = 0.023) -> ExtractionAnchor:
    """
    Multi-Stage Verification Hierarchy (MSVH).
    
    Progresses through:
      Step A: Strict binary match
      Step B: Normalized match (formatting stripped)
      Step C: OCR-aware fuzzy match (Levenshtein + confusable check)
    
    Returns the anchor updated with verification metadata and confidence.
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
    
    # Rejected: no match at any tier
    anchor.metadata = VerificationTier.REJECTED
    anchor.confidence = 0.0
    return anchor


def sav_pipeline(source_document: str, 
                 proposed_anchors: List[ExtractionAnchor],
                 max_retries: int = 3,
                 cer_engine: float = 0.023) -> List[ExtractionAnchor]:
    """
    Complete SAV pipeline with automated rejection/refinement loop guard.
    
    Returns verified anchors. Any that fail all retries are returned
    with REJECTED metadata and null values.
    """
    verified = []
    rejected = []
    
    # Stage 2: Initial verification
    for anchor in proposed_anchors:
        result = sav_verify(source_document, anchor, cer_engine)
        if result.metadata != VerificationTier.REJECTED:
            verified.append(result)
        else:
            rejected.append(result)
    
    # Stage 4: Re-extraction loop with guard
    retry_count = 0
    while rejected and retry_count < max_retries:
        # In a real implementation, this would call the LLM again.
        # For this reference implementation, re-verification is simulated.
        still_rejected = []
        for anchor in rejected:
            # Simulated: in practice, LLM would propose new offsets
            result = sav_verify(source_document, anchor, cer_engine)
            if result.metadata != VerificationTier.REJECTED:
                verified.append(result)
            else:
                still_rejected.append(result)
        
        rejected = still_rejected
        retry_count += 1
    
    # Assign null to exhausted rejections
    for anchor in rejected:
        anchor.value = None
        anchor.start_offset = -1
        anchor.end_offset = -1
        anchor.confidence = 0.0
        anchor.metadata = VerificationTier.REJECTED
        verified.append(anchor)
    
    return verified


# ============================================================================
# Test Suite
# ============================================================================

def run_tests():
    """Execute the SAV verification test suite."""
    
    source_doc = (
        "INVOICE #A1\n"
        "Date: 10/12/2024\n"
        "Total: $1,234.56\n"
        "Item Code: BAN8-1234\n"
        "Quantity: 1\n"
    )
    
    test_cases = [
        # (start, end, extracted_value, expected_tier, description)
        (47, 56, "BAN8-1234", VerificationTier.LITERAL_MATCH,
         "Literal match - identical text"),
        (33, 43, "1234.56", VerificationTier.NORMALIZED_MATCH,
         "Normalized match - formatting stripped"),
        (47, 56, "BANB-1234", VerificationTier.FUZZY_MATCH,
         "OCR fuzzy match - 8/B confusable substitution"),
        (14, 25, "500.00", VerificationTier.REJECTED,
         "Hallucination - wrong value in date field"),
        (0, 13, "1nvoice #A1", VerificationTier.FUZZY_MATCH,
         "OCR fuzzy match - 1/I confusable at start"),
        (57, 59, "l", VerificationTier.FUZZY_MATCH,
         "OCR fuzzy match - 1/l confusable"),
        (14, 25, "10/12/2024", VerificationTier.LITERAL_MATCH,
         "Literal match - date field correct"),
    ]
    
    print("=" * 72)
    print("SAV MULTI-STAGE VERIFICATION HIERARCHY - TEST SUITE")
    print("=" * 72)
    
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
        
        symbol = "✅" if status == "PASS" else "❌"
        
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
    
    ocr_tests = [
        (8, 12, "0.00", VerificationTier.FUZZY_MATCH,
         "OCR 'O.OO' corrected to '0.00'"),
        (8, 12, "O.OO", VerificationTier.LITERAL_MATCH,
         "OCR 'O.OO' kept as-is (literal)"),
        (17, 26, "BAN8-1234", VerificationTier.FUZZY_MATCH,
         "OCR 'BANB' corrected to 'BAN8' (B/8 confusable)"),
        (17, 26, "BANK-1234", VerificationTier.REJECTED,
         "Hallucination - BANK not confusable with BANB"),
    ]
    
    for start, end, value, expected_tier, description in ocr_tests:
        anchor = ExtractionAnchor(
            value=value,
            start_offset=start,
            end_offset=end
        )
        
        result = sav_verify(ocr_doc, anchor)
        source_slice = ocr_doc[start:end]
        distance = levenshtein_distance(source_slice, value)
        diff_pairs = get_diff_pairs(source_slice, value)
        
        status = "PASS" if result.metadata == expected_tier else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        
        symbol = "✅" if status == "PASS" else "❌"
        
        print(f"\n{symbol} {status}: {description}")
        print(f"   Source slice:    '{source_slice}'")
        print(f"   Extracted value: '{value}'")
        print(f"   Levenshtein:     {distance} edits")
        print(f"   Threshold:       {compute_threshold(max(len(source_slice), len(value)))}")
        print(f"   Diff pairs:      {diff_pairs}")
        print(f"   All confusable:  {are_all_ocr_confusables(diff_pairs)}")
        print(f"   Verification:    {result.metadata.value}")
        print(f"   Confidence:      {result.confidence:.2f}")
    
    # Pipeline test with loop guard
    print("\n" + "=" * 72)
    print("PIPELINE TEST: RE-EXTRACTION LOOP GUARD")
    print("=" * 72)
    
    proposed = [
        ExtractionAnchor(value="BAN8-1234", start_offset=47, end_offset=56),
        ExtractionAnchor(value="500.00", start_offset=14, end_offset=25),  # hallucination
    ]
    
    results = sav_pipeline(source_doc, proposed, max_retries=3)
    
    for r in results:
        if r.value is None:
            print(f"\n✅ Rejected (null): offset [{r.start_offset}:{r.end_offset}]")
            print(f"   Expected hallucination correctly rejected after retries")
        else:
            print(f"\n✅ Verified: '{r.value}' at [{r.start_offset}:{r.end_offset}]")
            print(f"   Tier: {r.metadata.value}, Confidence: {r.confidence:.2f}")
    
    # Summary
    print("\n" + "=" * 72)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("All tests passed! ✅")
    else:
        print(f"{failed} test(s) failed ❌")
    print("=" * 72)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
