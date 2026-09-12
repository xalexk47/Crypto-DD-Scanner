"""Tests for wallet balance reading, with every RPC and explorer stubbed."""

from __future__ import annotations

import pytest

from src import balances, config
from src.balances import (BlockscoutProvider, EvmRpcProvider, SolanaRpcProvider,
                          decode_uint, units_from_raw)
from src.models import Wallet

WALLET = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
SOL_WALLET = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
BREW = "0x1234567890abcdef1234567890abcdef12345678"


def hex_word(value: int) -> str:
    return "0x" + format(value, "064x")


class TestDecoding:
    def test_decode_uint_reads_a_hex_word(self):
        assert decode_uint(hex_word(10**18)) == 10**18

    @pytest.mark.parametrize("value", ["0x", "", None, "nonsense", 42])
    def test_empty_answer_is_not_a_zero_balance(self, value):
        # "0x" means no contract answered. Reading that as zero would turn a
        # wrong-chain query into a confident "you hold none".
        assert decode_uint(value) is None

    def test_units_respect_decimals(self):
        assert units_from_raw(1_500_000, 6) == pytest.approx(1.5)
        assert units_from_raw(10**18, 18) == pytest.approx(1.0)
        assert units_from_raw(1234, 0) == pytest.approx(1234.0)

    def test_missing_raw_value_is_zero_units(self):
        assert units_from_raw(None, 18) == 0.0


class TestBlockscoutProvider:
    def test_parses_token_list(self):
        payload = {
            "status": "1",
            "result": [
                {"contractAddress": BREW, "symbol": "BREW", "name": "Brew",
                 "decimals": "18", "balance": str(25 * 10**18), "type": "ERC-20"},
                {"contractAddress": "0xnft", "symbol": "APE", "name": "Ape",
                 "decimals": "0", "balance": "3", "type": "ERC-721"},
            ],
        }
        provider = BlockscoutProvider("bsc", base_url="https://x.test",
                                      session=lambda url, params: payload)
        result = provider.fetch(WALLET)

        assert result.ok
        assert len(result.balances) == 1           # the NFT is not a position
        assert result.balances[0].symbol == "BREW"
        assert result.balances[0].quantity == pytest.approx(25.0)
        assert result.balances[0].decimals == 18

    def test_zero_balances_are_dropped(self):
        payload = {"status": "1", "result": [
            {"contractAddress": BREW, "symbol": "BREW", "decimals": "18",
             "balance": "0", "type": "ERC-20"},
        ]}
        provider = BlockscoutProvider("bsc", base_url="https://x.test",
                                      session=lambda url, params: payload)
        assert provider.fetch(WALLET).balances == []

    def test_unconfigured_chain_says_why(self, monkeypatch):
        monkeypatch.setattr(config, "BLOCKSCOUT_BASE_URLS", {})
        provider = BlockscoutProvider("bsc")
        assert "No Blockscout instance" in provider.unavailable_reason()
        result = provider.fetch(WALLET)
        assert not result.ok and result.warnings

    def test_error_payload_is_a_failure_not_an_empty_wallet(self):
        provider = BlockscoutProvider(
            "bsc", base_url="https://x.test",
            session=lambda url, params: {"status": "0", "message": "rate limited"},
        )
        result = provider.fetch(WALLET)
        assert not result.ok
        assert "rate limited" in result.warnings[0]

    def test_no_tokens_found_is_an_empty_wallet_not_a_failure(self):
        provider = BlockscoutProvider(
            "bsc", base_url="https://x.test",
            session=lambda url, params: {"status": "0", "message": "No tokens found"},
        )
        result = provider.fetch(WALLET)
        assert result.ok and result.balances == []


class TestEvmRpcProvider:
    def _provider(self, responses, chain="bsc", api_key="key"):
        """Stub session routing Etherscan GETs and RPC POSTs to canned data."""
        def session(url, payload):
            if isinstance(payload, dict) and payload.get("jsonrpc"):
                return responses["rpc"](payload)
            return responses["etherscan"]
        return EvmRpcProvider(chain, rpc_url="https://rpc.test", api_key=api_key, session=session)

    def test_discovery_reads_decimals_from_transfer_rows(self):
        provider = self._provider({
            "etherscan": {"status": "1", "result": [
                {"contractAddress": BREW, "tokenSymbol": "BREW",
                 "tokenName": "Brew", "tokenDecimal": "9"},
                {"contractAddress": BREW.upper(), "tokenSymbol": "BREW",
                 "tokenName": "Brew", "tokenDecimal": "9"},
            ]},
            "rpc": lambda payload: {},
        })
        tokens = provider.discover_tokens(WALLET)
        assert len(tokens) == 1                    # case-folded to one entry
        assert tokens[BREW.lower()]["decimals"] == 9

    def test_fetch_combines_discovery_and_balance_calls(self):
        def rpc(payload):
            if payload["method"] == "eth_getBalance":
                return {"result": hex_word(2 * 10**18)}
            data = payload["params"][0]["data"]
            if data.startswith(balances.SELECTOR_BALANCE_OF):
                return {"result": hex_word(500 * 10**18)}
            return {"result": hex_word(18)}

        provider = self._provider({
            "etherscan": {"status": "1", "result": [
                {"contractAddress": BREW, "tokenSymbol": "BREW",
                 "tokenName": "Brew", "tokenDecimal": "18"},
            ]},
            "rpc": rpc,
        })
        result = provider.fetch(WALLET)

        held = {b.symbol: b for b in result.balances}
        assert held["BREW"].quantity == pytest.approx(500.0)
        # The native coin is dry powder for a rotation, so it is a position too.
        assert held["BNB"].quantity == pytest.approx(2.0)
        assert held["BNB"].is_native
        assert held["BNB"].address == config.WRAPPED_NATIVE["bsc"]

    def test_zero_and_unanswered_balances_are_skipped(self):
        def rpc(payload):
            if payload["method"] == "eth_getBalance":
                return {"result": "0x0"}
            return {"result": "0x"}          # contract did not answer

        provider = self._provider({
            "etherscan": {"status": "1", "result": [
                {"contractAddress": BREW, "tokenSymbol": "BREW", "tokenDecimal": "18"},
            ]},
            "rpc": rpc,
        })
        assert provider.fetch(WALLET).balances == []

    def test_manual_token_list_skips_discovery_entirely(self):
        def session(url, payload):
            if isinstance(payload, dict) and payload.get("jsonrpc"):
                if payload["method"] == "eth_getBalance":
                    return {"result": "0x0"}
                return {"result": hex_word(7 * 10**6)}
            raise AssertionError("discovery should not run when tokens are given")

        provider = EvmRpcProvider("robinhood", rpc_url="https://rpc.test",
                                  api_key="", session=session)
        result = provider.fetch(WALLET, tokens=[{"address": BREW, "symbol": "HOOD", "decimals": 6}])
        assert result.balances[0].quantity == pytest.approx(7.0)

    def test_missing_rpc_url_reports_the_missing_setting(self, monkeypatch):
        monkeypatch.setitem(config.EVM_RPC_URLS, "robinhood", "")
        provider = EvmRpcProvider("robinhood")
        assert "ROBINHOOD_RPC_URL" in provider.unavailable_reason()
        assert not provider.fetch(WALLET).ok

    def test_robinhood_cannot_use_etherscan_discovery(self):
        # Robinhood Chain has no etherscan_chain_id, so discovery must say so
        # rather than querying the wrong chain id.
        provider = EvmRpcProvider("robinhood", rpc_url="https://rpc.test", api_key="key")
        assert "not an Etherscan V2 chain" in provider.discovery_reason()

    def test_missing_etherscan_key_is_a_discovery_problem_only(self):
        provider = EvmRpcProvider("bsc", rpc_url="https://rpc.test", api_key="")
        assert provider.unavailable_reason() == ""
        assert "ETHERSCAN_API_KEY" in provider.discovery_reason()

    def test_partial_rpc_failure_is_reported_not_hidden(self):
        calls = {"n": 0}

        def rpc(payload):
            if payload["method"] == "eth_getBalance":
                return {"result": "0x0"}
            calls["n"] += 1
            if calls["n"] == 1:
                return {"error": {"message": "execution reverted"}}
            return {"result": hex_word(10**18)}

        provider = self._provider({
            "etherscan": {"status": "1", "result": [
                {"contractAddress": BREW, "tokenSymbol": "A", "tokenDecimal": "18"},
                {"contractAddress": "0xother", "tokenSymbol": "B", "tokenDecimal": "18"},
            ]},
            "rpc": rpc,
        })
        result = provider.fetch(WALLET)
        assert len(result.balances) == 1
        assert any("failed" in w for w in result.warnings)

    def test_token_fan_out_is_capped(self, monkeypatch):
        monkeypatch.setattr(config, "PORTFOLIO_MAX_TOKENS_PER_WALLET", 2)
        rows = [
            {"contractAddress": f"0x{i:040x}", "tokenSymbol": f"T{i}", "tokenDecimal": "18"}
            for i in range(10)
        ]
        provider = self._provider({
            "etherscan": {"status": "1", "result": rows},
            "rpc": lambda payload: {"result": hex_word(10**18)},
        })
        result = provider.fetch(WALLET)
        # 2 capped tokens plus the native row.
        assert len([b for b in result.balances if not b.is_native]) == 2
        assert any("Raise" in w for w in result.warnings)


class TestSolanaRpcProvider:
    def _accounts(self, mint, amount, decimals=6):
        return {"value": [{"account": {"data": {"parsed": {"info": {
            "mint": mint,
            "tokenAmount": {"uiAmount": amount, "amount": str(int(amount * 10**decimals)),
                            "decimals": decimals},
        }}}}}]}

    def test_parses_token_accounts_and_native_sol(self):
        def session(url, payload):
            if payload["method"] == "getTokenAccountsByOwner":
                program = payload["params"][1]["programId"]
                if program == balances.SPL_TOKEN_PROGRAM:
                    return {"result": self._accounts("MintOne", 1200.5)}
                return {"result": {"value": []}}
            return {"result": {"value": 3 * 10**9}}

        result = SolanaRpcProvider(rpc_url="https://sol.test", session=session).fetch(SOL_WALLET)
        held = {b.address: b for b in result.balances}
        assert held["MintOne"].quantity == pytest.approx(1200.5)
        assert held[config.WRAPPED_NATIVE["solana"]].quantity == pytest.approx(3.0)

    def test_token_2022_positions_are_included(self):
        def session(url, payload):
            if payload["method"] == "getTokenAccountsByOwner":
                program = payload["params"][1]["programId"]
                if program == balances.SPL_TOKEN_2022_PROGRAM:
                    return {"result": self._accounts("Mint2022", 42)}
                return {"result": {"value": []}}
            return {"result": {"value": 0}}

        result = SolanaRpcProvider(rpc_url="https://sol.test", session=session).fetch(SOL_WALLET)
        assert [b.address for b in result.balances] == ["Mint2022"]

    def test_one_mint_across_several_accounts_is_one_position(self):
        def session(url, payload):
            if payload["method"] == "getTokenAccountsByOwner":
                if payload["params"][1]["programId"] == balances.SPL_TOKEN_PROGRAM:
                    accounts = self._accounts("MintOne", 100)
                    accounts["value"].extend(self._accounts("MintOne", 50)["value"])
                    return {"result": accounts}
                return {"result": {"value": []}}
            return {"result": {"value": 0}}

        result = SolanaRpcProvider(rpc_url="https://sol.test", session=session).fetch(SOL_WALLET)
        assert len(result.balances) == 1
        assert result.balances[0].quantity == pytest.approx(150.0)

    def test_unreachable_rpc_is_a_named_failure(self):
        def session(url, payload):
            raise RuntimeError("connection refused")

        result = SolanaRpcProvider(rpc_url="https://sol.test", session=session).fetch(SOL_WALLET)
        assert not result.ok
        assert "unreachable" in result.warnings[0]


class TestOrchestration:
    def test_blockscout_result_wins_before_rpc_is_tried(self, monkeypatch):
        monkeypatch.setitem(config.BLOCKSCOUT_BASE_URLS, "bsc", "https://scout.test")
        monkeypatch.setattr(
            BlockscoutProvider, "fetch",
            lambda self, wallet: balances.BalanceResult(
                balances=[balances.TokenBalance(address=BREW, chain="bsc", quantity=5.0)],
                source="blockscout",
            ),
        )
        monkeypatch.setattr(EvmRpcProvider, "fetch",
                            lambda *a, **k: pytest.fail("RPC should not be reached"))
        monkeypatch.setattr(EvmRpcProvider, "native_balance", lambda self, wallet: None)

        result = balances.fetch_balances_for_wallet(Wallet(address=WALLET, chain="bsc"))
        assert result.source == "blockscout"
        assert len(result.balances) == 1

    def test_rpc_is_the_fallback_when_blockscout_fails(self, monkeypatch):
        monkeypatch.setitem(config.BLOCKSCOUT_BASE_URLS, "bsc", "https://scout.test")
        monkeypatch.setattr(
            BlockscoutProvider, "fetch",
            lambda self, wallet: balances.BalanceResult(
                ok=False, warnings=["Blockscout (bsc) unreachable"], source="blockscout"),
        )
        monkeypatch.setattr(
            EvmRpcProvider, "fetch",
            lambda self, wallet, tokens=None: balances.BalanceResult(
                balances=[balances.TokenBalance(address=BREW, chain="bsc", quantity=9.0)],
                source="rpc"),
        )
        result = balances.fetch_balances_for_wallet(Wallet(address=WALLET, chain="bsc"))
        assert result.source == "rpc"
        assert result.balances[0].quantity == 9.0
        # The failed attempt is still reported, not swallowed.
        assert any("Blockscout" in w for w in result.warnings)

    def test_one_failing_chain_does_not_lose_the_others(self, monkeypatch):
        def fetch(wallet, manual_tokens=None):
            if wallet.chain == "bsc":
                raise RuntimeError("rpc exploded")
            return balances.BalanceResult(
                balances=[balances.TokenBalance(address="MintOne", chain=wallet.chain, quantity=1.0)])

        monkeypatch.setattr(balances, "fetch_balances_for_wallet", fetch)
        result = balances.fetch_all_balances([
            Wallet(address=WALLET, chain="bsc"),
            Wallet(address=SOL_WALLET, chain="solana"),
        ])
        assert len(result.balances) == 1
        assert any("rpc exploded" in w for w in result.warnings)
