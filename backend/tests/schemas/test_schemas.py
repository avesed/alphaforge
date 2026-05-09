"""Tests for Pydantic schema validation."""
import pytest
from pydantic import ValidationError

from app.schemas.base import ApiResponse, CamelModel, to_camel
from app.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.schemas.qlib import (
    BacktestStatus,
    ExpressionEvaluateRequest,
    ExpressionBatchRequest,
    MarketCode,
)


class TestToCamel:

    @pytest.mark.parametrize(
        "snake, camel",
        [
            ("access_token", "accessToken"),
            ("is_active", "isActive"),
            ("refresh_token_expire_days", "refreshTokenExpireDays"),
            ("id", "id"),
            ("email", "email"),
        ],
    )
    def test_conversion(self, snake, camel):
        assert to_camel(snake) == camel


class TestCamelModel:

    def test_serialize_by_alias(self):
        class Sample(CamelModel):
            first_name: str
            last_name: str

        obj = Sample(first_name="A", last_name="B")
        data = obj.model_dump(by_alias=True)
        assert "firstName" in data
        assert "lastName" in data

    def test_populate_by_name(self):
        class Sample(CamelModel):
            some_field: str

        obj = Sample(some_field="x")
        assert obj.some_field == "x"

    def test_populate_by_alias(self):
        class Sample(CamelModel):
            some_field: str

        obj = Sample(someField="y")
        assert obj.some_field == "y"


class TestApiResponse:

    def test_success_response(self):
        resp = ApiResponse[str](data="hello")
        assert resp.success is True
        assert resp.data == "hello"
        assert resp.error is None

    def test_error_response(self):
        resp = ApiResponse[str](success=False, error="bad request")
        assert resp.success is False
        assert resp.data is None
        assert resp.error == "bad request"

    def test_serialization_camel(self):
        resp = ApiResponse[str](data="x")
        d = resp.model_dump(by_alias=True)
        assert "success" in d
        assert "data" in d


class TestLoginRequest:

    def test_valid_login(self):
        req = LoginRequest(email="user@example.com", password="pass123")
        assert req.email == "user@example.com"

    def test_invalid_email(self):
        with pytest.raises(ValidationError):
            LoginRequest(email="not-an-email", password="pass")

    def test_empty_password_rejected(self):
        with pytest.raises(ValidationError):
            LoginRequest(email="user@example.com", password="")


class TestRegisterRequest:

    def test_valid_registration(self):
        req = RegisterRequest(email="new@user.com", password="secure123")
        assert req.password == "secure123"

    def test_short_password_rejected(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="new@user.com", password="12345")

    def test_min_password_length(self):
        req = RegisterRequest(email="new@user.com", password="123456")
        assert len(req.password) == 6


class TestChangePasswordRequest:

    def test_valid(self):
        req = ChangePasswordRequest(current_password="old", new_password="newpass")
        assert req.new_password == "newpass"

    def test_new_password_too_short(self):
        with pytest.raises(ValidationError):
            ChangePasswordRequest(current_password="old", new_password="12345")


class TestTokenResponse:

    def test_default_token_type(self):
        resp = TokenResponse(access_token="abc")
        assert resp.token_type == "bearer"

    def test_camel_serialization(self):
        resp = TokenResponse(access_token="abc", refresh_token="def")
        d = resp.model_dump(by_alias=True)
        assert "accessToken" in d
        assert "refreshToken" in d


class TestUserResponse:

    def test_from_dict(self):
        resp = UserResponse(id=1, email="a@b.com", role="admin", locale="zh", is_active=True)
        d = resp.model_dump(by_alias=True)
        assert d["isActive"] is True
        assert d["email"] == "a@b.com"


class TestMarketCode:

    @pytest.mark.parametrize("code", ["us", "hk", "sh", "sz", "cn", "metal"])
    def test_valid_codes(self, code):
        assert MarketCode(code).value == code

    def test_invalid_code(self):
        with pytest.raises(ValueError):
            MarketCode("jp")


class TestBacktestStatus:

    def test_all_statuses(self):
        statuses = {"pending", "running", "completed", "failed", "cancelled"}
        assert {s.value for s in BacktestStatus} == statuses


class TestExpressionEvaluateRequest:

    def test_valid_request(self):
        req = ExpressionEvaluateRequest(
            symbol="AAPL",
            expression="Mean($close, 5)",
            market=MarketCode.US,
        )
        assert req.period == "3mo"

    def test_symbol_pattern_valid(self):
        ExpressionEvaluateRequest(symbol="600000.SS", expression="$close")
        ExpressionEvaluateRequest(symbol="GC=F", expression="$close")

    def test_symbol_too_long(self):
        with pytest.raises(ValidationError):
            ExpressionEvaluateRequest(
                symbol="A" * 21, expression="$close"
            )

    def test_symbol_invalid_chars(self):
        with pytest.raises(ValidationError):
            ExpressionEvaluateRequest(
                symbol="AAPL;DROP TABLE", expression="$close"
            )


class TestExpressionBatchRequest:

    def test_empty_symbols_rejected(self):
        with pytest.raises(ValidationError):
            ExpressionBatchRequest(symbols=[], expression="$close")

    def test_min_one_symbol(self):
        req = ExpressionBatchRequest(symbols=["AAPL"], expression="$close")
        assert len(req.symbols) == 1
