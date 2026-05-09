"""Tests for ExpressionEngine.validate() — pure string validation, no Qlib needed."""
import pytest

from app.services.expression_engine import (
    ALLOWED_OPERATORS,
    ALLOWED_VARIABLES,
    ExpressionEngine,
)


class TestValidateHappyPath:
    """Valid expressions should pass validation."""

    @pytest.mark.parametrize(
        "expr",
        [
            "Corr($close, $volume, 20)",
            "Mean($close, 5)",
            "Std($close, 20) / Mean($close, 20)",
            "Rank(Mean($close, 5))",
            "If(Greater($close, Ref($close, 1)), $volume, 0)",
            "$close / Ref($close, 1) - 1",
            "$close",
            "EMA($close, 12) - EMA($close, 26)",
        ],
    )
    def test_valid_expressions(self, expr):
        is_valid, error, ops = ExpressionEngine.validate(expr)
        assert is_valid is True
        assert error == ""

    def test_operators_extracted(self):
        _, _, ops = ExpressionEngine.validate("Corr($close, Mean($volume, 5), 20)")
        assert set(ops) == {"Corr", "Mean"}

    def test_all_allowed_operators_pass(self):
        for op in ALLOWED_OPERATORS:
            expr = f"{op}($close, 5)"
            is_valid, error, _ = ExpressionEngine.validate(expr)
            assert is_valid, f"Operator {op} should be allowed but got: {error}"

    def test_all_allowed_variables_pass(self):
        for var in ALLOWED_VARIABLES:
            is_valid, error, _ = ExpressionEngine.validate(f"Mean({var}, 5)")
            assert is_valid, f"Variable {var} should be allowed but got: {error}"

    def test_arithmetic_operators_allowed(self):
        is_valid, _, _ = ExpressionEngine.validate("$close * 2 + $volume / 100 - 1")
        assert is_valid

    def test_numeric_literals_allowed(self):
        is_valid, _, _ = ExpressionEngine.validate("Mean($close, 5) + 0.5")
        assert is_valid


class TestValidateRejection:
    """Invalid expressions should be rejected with clear error messages."""

    def test_empty_expression(self):
        is_valid, error, _ = ExpressionEngine.validate("")
        assert not is_valid
        assert "Empty" in error

    def test_whitespace_only(self):
        is_valid, error, _ = ExpressionEngine.validate("   ")
        assert not is_valid
        assert "Empty" in error

    def test_too_long(self):
        expr = "Mean($close, 5)" * 100
        is_valid, error, _ = ExpressionEngine.validate(expr, max_length=500)
        assert not is_valid
        assert "too long" in error.lower()

    def test_custom_max_length(self):
        is_valid, _, _ = ExpressionEngine.validate("Mean($close, 5)", max_length=10)
        assert not is_valid

    def test_unknown_operator(self):
        is_valid, error, ops = ExpressionEngine.validate("FakeOp($close, 5)")
        assert not is_valid
        assert "Unknown operators" in error
        assert "FakeOp" in error

    def test_multiple_unknown_operators(self):
        is_valid, error, _ = ExpressionEngine.validate("BadA(BadB($close, 1), 2)")
        assert not is_valid
        assert "BadA" in error or "BadB" in error

    def test_unknown_variable(self):
        is_valid, error, _ = ExpressionEngine.validate("Mean($badvar, 5)")
        assert not is_valid
        assert "Unknown variables" in error
        assert "$badvar" in error

    def test_unbalanced_parentheses(self):
        is_valid, error, _ = ExpressionEngine.validate("Mean($close, 5")
        assert not is_valid
        assert "parentheses" in error.lower()

    def test_extra_closing_paren(self):
        is_valid, error, _ = ExpressionEngine.validate("Mean($close, 5))")
        assert not is_valid
        assert "parentheses" in error.lower()


class TestDangerousPatterns:
    """Security: injection attempts must be rejected."""

    @pytest.mark.parametrize(
        "dangerous_expr",
        [
            "__import__('os').system('rm -rf /')",
            "import os",
            "exec('print(1)')",
            "eval('1+1')",
            "open('/etc/passwd')",
            "os.system('ls')",
            "sys.exit()",
            "subprocess.run(['ls'])",
        ],
    )
    def test_dangerous_patterns_rejected(self, dangerous_expr):
        is_valid, error, _ = ExpressionEngine.validate(dangerous_expr)
        assert not is_valid
        assert "disallowed" in error.lower()

    def test_dunder_in_expression(self):
        is_valid, _, _ = ExpressionEngine.validate("$close.__class__")
        assert not is_valid

    def test_case_insensitive_dangerous_patterns(self):
        is_valid, _, _ = ExpressionEngine.validate("IMPORT os")
        assert not is_valid

    def test_exec_with_spaces(self):
        is_valid, _, _ = ExpressionEngine.validate("exec  ('code')")
        assert not is_valid


class TestEdgeCases:

    def test_max_length_boundary_valid(self):
        expr = "Mean($close, 5)"
        is_valid, _, _ = ExpressionEngine.validate(expr, max_length=len(expr))
        assert is_valid

    def test_max_length_boundary_invalid(self):
        expr = "Mean($close, 5)"
        is_valid, _, _ = ExpressionEngine.validate(expr, max_length=len(expr) - 1)
        assert not is_valid

    def test_period_to_days_mapping(self):
        assert ExpressionEngine.PERIOD_TO_DAYS["1mo"] == 30
        assert ExpressionEngine.PERIOD_TO_DAYS["1y"] == 365
        assert ExpressionEngine.PERIOD_TO_DAYS["5y"] == 1825

    def test_nested_expression(self):
        expr = "Rank(Mean(Std($close, 20), 5))"
        is_valid, _, ops = ExpressionEngine.validate(expr)
        assert is_valid
        assert set(ops) == {"Rank", "Mean", "Std"}
