"""Unit tests for the formatting / parsing helpers."""

from __future__ import annotations

import time

import pytest

from src.utils import (
    age_hours,
    clamp,
    detect_address_kind,
    dig,
    fmt_age,
    fmt_pct,
    fmt_usd,
    is_burn_address,
    log_scale,
    parse_addresses,
    safe_bool,
    safe_float,
    scale,
    short_address,
    TTLCache,
)


class TestAddressParsing:
    def test_detects_evm_and_solana(self):
        assert detect_address_kind("0x532f27101965dd16442E59d40670FaF5eBB142E4") == "evm"
        assert detect_address_kind("DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263") == "solana"
        assert detect_address_kind("not-an-address") is None
        assert detect_address_kind("") is None

    def test_parse_addresses_handles_mixed_separators_and_urls(self):
        raw = (
            "0x532f27101965dd16442E59d40670FaF5eBB142E4, "
            "https://dexscreener.com/base/0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed\n"
            "garbage 0xdeadbeef"
        )
        valid, invalid = parse_addresses(raw)
        assert len(valid) == 2
        assert "garbage" in invalid and "0xdeadbeef" in invalid

    def test_parse_addresses_deduplicates_case_insensitively(self):
        addr = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
        valid, _ = parse_addresses(f"{addr} {addr.lower()}")
        assert len(valid) == 1

    def test_burn_address_detection(self):
        assert is_burn_address("0x000000000000000000000000000000000000dEaD")
        assert is_burn_address("0xabc", tag="Burn Address")
        assert not is_burn_address("0xaaa1111111111111111111111111111111111111")

    def test_short_address(self):
        assert short_address("0x532f27101965dd16442E59d40670FaF5eBB142E4") == "0x532f...42E4"
        assert short_address("0xabc") == "0xabc"


class TestCoercion:
    @pytest.mark.parametrize(
        "value,expected",
        [("1,234.56", 1234.56), ("$42", 42.0), ("5%", 5.0), (None, 0.0), ("", 0.0), ("abc", 0.0), (7, 7.0)],
    )
    def test_safe_float(self, value, expected):
        assert safe_float(value) == expected

    def test_safe_bool_tristate(self):
        assert safe_bool("1") is True
        assert safe_bool("0") is False
        assert safe_bool(None) is None
        assert safe_bool("") is None

    def test_dig_never_raises(self):
        payload = {"a": {"b": [{"c": 1}]}}
        assert dig(payload, "a", "b", 0, "c") == 1
        assert dig(payload, "a", "x", "y", default="fallback") == "fallback"
        assert dig(None, "a", default=3) == 3


class TestScales:
    def test_clamp_and_scale(self):
        assert clamp(150) == 100
        assert clamp(-5) == 0
        assert scale(5, 0, 10) == 50

    def test_log_scale_is_monotonic_and_bounded(self):
        assert log_scale(1_000, 10_000, 1_000_000) == 0
        assert log_scale(10_000_000, 10_000, 1_000_000) == 100
        assert 40 < log_scale(100_000, 10_000, 1_000_000) < 60
        assert log_scale(0, 10, 100) == 0


class TestFormatting:
    def test_fmt_usd(self):
        assert fmt_usd(1_234_567) == "$1.23M"
        assert fmt_usd(999) == "$999.00"
        assert fmt_usd(0.000001234).startswith("$0.0000012")
        assert fmt_usd(None) == "n/a"

    def test_fmt_pct(self):
        assert fmt_pct(12.345, 1, signed=True) == "+12.3%"
        assert fmt_pct(None) == "n/a"

    def test_fmt_age(self):
        now_ms = int(time.time() * 1000)
        assert fmt_age(now_ms - 3_600_000).endswith("h")
        assert fmt_age(now_ms - 10 * 86_400_000).endswith("d")
        assert fmt_age(None) == "unknown"
        assert age_hours(None) is None


class TestTTLCache:
    def test_stores_and_expires(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("k", "v")
        assert cache.get("k") == "v"
        cache.set("expired", "v", ttl=-1)
        assert cache.get("expired") is None

    def test_get_or_set_calls_producer_once(self):
        cache = TTLCache(ttl_seconds=60)
        calls = []

        def producer():
            calls.append(1)
            return "value"

        assert cache.get_or_set("k", producer) == "value"
        assert cache.get_or_set("k", producer) == "value"
        assert len(calls) == 1
