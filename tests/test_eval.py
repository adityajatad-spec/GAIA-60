"""Tests for the GAIA evaluation harness, focusing on _quasi_exact_match."""

from __future__ import annotations

from eval.run_eval import (
    _compound_match,
    _extract_numbers,
    _find_bool_terms,
    _is_compound,
    _is_number_subsequence,
    _normalize,
    _quasi_exact_match,
)


# ── Compound-match helper tests ───────────────────────────────────────


class TestExtractNumbers:
    def test_simple_integers(self):
        assert _extract_numbers("1896, 1989, 1903") == [1896.0, 1989.0, 1903.0]

    def test_floats(self):
        assert _extract_numbers("ratio~1.88, final~3.76") == [1.88, 3.76]

    def test_mixed(self):
        assert _extract_numbers("sum=5788, average=1929") == [5788.0, 1929.0]

    def test_no_numbers(self):
        assert _extract_numbers("hello world") == []

    def test_single_number(self):
        assert _extract_numbers("6765") == [6765.0]


class TestFindBoolTerms:
    def test_yes(self):
        assert _find_bool_terms("yes") == {"yes"}

    def test_no(self):
        assert _find_bool_terms("no") == {"no"}

    def test_true(self):
        assert _find_bool_terms("true") == {"true"}

    def test_false(self):
        assert _find_bool_terms("false") == {"false"}

    def test_even(self):
        assert _find_bool_terms("even") == {"even"}

    def test_odd(self):
        assert _find_bool_terms("odd") == {"odd"}

    def test_compound_with_bool(self):
        assert _find_bool_terms("ratio~1.88, final~3.76, yes") == {"yes"}

    def test_compound_with_even(self):
        assert _find_bool_terms("1896, 1989, 1903, sum=5788, average=1929, even") == {"even"}

    def test_multiple_terms(self):
        assert _find_bool_terms("true and even") == {"true", "even"}

    def test_no_terms(self):
        assert _find_bool_terms("just 42 numbers") == set()


class TestIsCompound:
    def test_two_numbers(self):
        assert _is_compound("ratio~1.88, final~3.76, yes") is True

    def test_five_numbers(self):
        assert _is_compound("1896, 1989, 1903, sum=5788, average=1929, even") is True

    def test_single_number_not_compound(self):
        assert _is_compound("6765") is False

    def test_yes_alone_not_compound(self):
        assert _is_compound("Yes") is False

    def test_single_number_with_bool_is_compound(self):
        assert _is_compound("21, yes") is True

    def test_empty(self):
        assert _is_compound("") is False

    def test_text_no_numbers(self):
        assert _is_compound("hello world") is False


class TestIsNumberSubsequence:
    def test_exact_match(self):
        assert _is_number_subsequence([1.88, 3.76], [1.88, 3.76]) is True

    def test_within_tolerance(self):
        assert _is_number_subsequence([1.88001, 3.76001], [1.88, 3.76]) is True

    def test_subsequence_with_extra(self):
        assert _is_number_subsequence([1.88, 2.0, 3.76], [1.88, 3.76]) is True

    def test_different_value(self):
        assert _is_number_subsequence([1.88, 3.76], [1.88, 2.76]) is False

    def test_empty_exp_is_match(self):
        assert _is_number_subsequence([1.88], []) is True

    def test_empty_pred_not_match(self):
        assert _is_number_subsequence([], [1.88]) is False

    def test_tiny_values(self):
        assert _is_number_subsequence([1e-10], [1e-10]) is True
        assert _is_number_subsequence([1e-10], [1e-6]) is False

    def test_large_values(self):
        assert _is_number_subsequence([5788, 1929], [5788, 1929]) is True

    def test_extra_number_in_pred(self):
        """Predicted has narrative number '3' that is not in GT."""
        assert _is_number_subsequence([1.88, 3.76, 3.0], [1.88, 3.76]) is True


class TestCompoundMatch:
    def test_example_1_compound(self):
        """Example 1: multi-year sum problem."""
        pred = "1896, 1989, 1903, 5788, 1929, even"
        gt = "1896, 1989, 1903, sum=5788, average=1929, even"
        assert _compound_match(pred, gt) is True

    def test_example_2_compound(self):
        """Example 2: ratio problem."""
        pred = "Ratio: 1.88, Multiplied Result: 3.76, Is it greater than 3? Yes."
        gt = "ratio~1.88, final~3.76, yes"
        assert _compound_match(pred, gt) is True

    def test_wrong_number(self):
        """Genuinely wrong answer should fail."""
        pred = "Ratio: 1.88, Multiplied Result: 2.76, Is it greater than 3? No."
        gt = "ratio~1.88, final~3.76, yes"
        assert _compound_match(pred, gt) is False

    def test_wrong_bool_term(self):
        """Even/odd mismatch should fail."""
        pred = "1896, 1989, 1903, 5788, 1929, odd"
        gt = "1896, 1989, 1903, sum=5788, average=1929, even"
        assert _compound_match(pred, gt) is False

    def test_missing_number(self):
        """Missing a number should fail."""
        pred = "1896, 1989, 1903, 5788"
        gt = "1896, 1989, 1903, sum=5788, average=1929, even"
        assert _compound_match(pred, gt) is False

    def test_missing_bool_term(self):
        """Predicted missing a boolean should fail."""
        pred = "1896, 1989, 1903, 5788, 1929"
        gt = "1896, 1989, 1903, sum=5788, average=1929, even"
        assert _compound_match(pred, gt) is False

    def test_extra_number_still_passes(self):
        """Extra number in predicted is OK as long as GT numbers are present in order."""
        pred = "1.88, 3.76, 42, yes"
        gt = "ratio~1.88, final~3.76, yes"
        assert _compound_match(pred, gt) is True

    def test_extra_number_missing_bool_fails(self):
        """Extra number doesn't compensate for missing boolean term."""
        pred = "1.88, 3.76, 42"
        gt = "ratio~1.88, final~3.76, yes"
        assert _compound_match(pred, gt) is False


# ── Top-level _quasi_exact_match tests ────────────────────────────────


class TestQuasiExactMatch:
    def test_example_1_top_level(self):
        """Example 1 via _quasi_exact_match — should now pass."""
        pred = "1896, 1989, 1903, 5788, 1929, even"
        gt = "1896, 1989, 1903, sum=5788, average=1929, even"
        assert _quasi_exact_match(pred, gt) is True

    def test_example_2_top_level(self):
        """Example 2 via _quasi_exact_match — should now pass."""
        pred = "Ratio: 1.88, Multiplied Result: 3.76, Is it greater than 3? Yes."
        gt = "ratio~1.88, final~3.76, yes"
        assert _quasi_exact_match(pred, gt) is True

    def test_genuinely_wrong_still_fails(self):
        """Wrong numbers in compound should still fail at top level."""
        pred = "Ratio: 1.88, Multiplied Result: 2.76, Is it greater than 3? No."
        gt = "ratio~1.88, final~3.76, yes"
        assert _quasi_exact_match(pred, gt) is False

    def test_single_number_still_works(self):
        """Single-value answers unchanged."""
        assert _quasi_exact_match("6765", "6765") is True

    def test_single_number_with_tolerance(self):
        assert _quasi_exact_match("6765.0001", "6765") is True

    def test_wrong_single_number(self):
        assert _quasi_exact_match("6766", "6765") is False

    def test_yes_no_still_works(self):
        assert _quasi_exact_match("Yes", "yes") is True
        assert _quasi_exact_match("No", "no") is True
        assert _quasi_exact_match("yes", "no") is False

    def test_true_false_still_works(self):
        assert _quasi_exact_match("TRUE", "true") is True
        assert _quasi_exact_match("false", "true") is False

    def test_normalized_string_match(self):
        assert _quasi_exact_match("The answer is 42", "42") is True

    def test_pipe_alternatives(self):
        assert _quasi_exact_match("Paris", "Paris | London") is True

    def test_empty_predicted(self):
        assert _quasi_exact_match("", "Paris") is False

    def test_empty_expected(self):
        assert _quasi_exact_match("Paris", "") is False

    def test_both_empty(self):
        assert _quasi_exact_match("", "") is True
