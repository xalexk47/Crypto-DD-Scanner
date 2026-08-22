"""Tests for the DexScreener token-profiles feed and snapshot enrichment."""

from __future__ import annotations

import pytest

from src import data_fetchers
from src.data_fetchers import FetchError, snapshot_from_pair
from src.models import SocialLink, TokenProfile
from tests.fixtures import (
    TOKEN_ADDRESS,
    dexscreener_pair,
    token_profile_item,
    token_profiles_response,
)


@pytest.fixture(autouse=True)
def clear_caches():
    data_fetchers.clear_caches()
    yield
    data_fetchers.clear_caches()


@pytest.fixture
def stub_profiles(monkeypatch):
    monkeypatch.setattr(
        data_fetchers, "_get_json",
        lambda url, params=None, retries=None: token_profiles_response(),
    )


class TestNormalization:
    def test_maps_every_profile_field(self):
        profile = data_fetchers.normalize_token_profile(token_profile_item())

        assert profile.address == TOKEN_ADDRESS
        assert profile.chain == "base"
        assert profile.description.startswith("Brett is Pepe's best friend")
        assert profile.icon_url.endswith("icon.png")
        assert profile.header_url.endswith("header.png")
        assert profile.has_content

    def test_handles_both_typed_and_labelled_links(self):
        profile = data_fetchers.normalize_token_profile(token_profile_item())
        kinds = {link.kind for link in profile.links}
        assert kinds == {"twitter", "telegram", "website"}

    def test_entry_without_an_address_is_dropped(self):
        assert data_fetchers.normalize_token_profile({"chainId": "base"}) is None

    def test_links_without_urls_are_skipped(self):
        profile = data_fetchers.normalize_token_profile(
            token_profile_item(links=[{"type": "twitter"}, {"type": "telegram", "url": "https://t.me/x"}])
        )
        assert len(profile.links) == 1

    def test_empty_profile_reports_no_content(self):
        assert TokenProfile(address="0x1", chain="base").has_content is False


class TestFeed:
    def test_filters_to_the_requested_chain(self, stub_profiles):
        base = data_fetchers.fetch_latest_profiles("base")
        assert len(base) == 2
        assert all(p.chain == "base" for p in base)

        assert len(data_fetchers.fetch_latest_profiles("solana")) == 1
        assert len(data_fetchers.fetch_latest_profiles()) == 3   # no filter

    def test_lookup_by_address_is_case_insensitive(self, stub_profiles):
        found = data_fetchers.fetch_token_profile(TOKEN_ADDRESS.lower(), "base")
        assert found is not None and found.address == TOKEN_ADDRESS

    def test_missing_token_returns_none_not_an_error(self, stub_profiles):
        assert data_fetchers.fetch_token_profile("0x" + "9" * 40, "base") is None

    def test_feed_outage_degrades_to_empty(self, monkeypatch):
        def boom(*args, **kwargs):
            raise FetchError("HTTP 503")

        monkeypatch.setattr(data_fetchers, "_get_json", boom)
        assert data_fetchers.fetch_latest_profiles("base") == []
        assert data_fetchers.fetch_token_profile(TOKEN_ADDRESS, "base") is None

    def test_feed_is_cached_across_lookups(self, monkeypatch):
        calls = []

        def counting(url, params=None, retries=None):
            calls.append(url)
            return token_profiles_response()

        monkeypatch.setattr(data_fetchers, "_get_json", counting)
        data_fetchers.fetch_token_profile(TOKEN_ADDRESS, "base")
        data_fetchers.fetch_token_profile("0xB000000000000000000000000000000000000002", "base")
        assert len(calls) == 1


class TestEnrichment:
    def _snapshot(self, **overrides):
        pair = dexscreener_pair(**overrides)
        return snapshot_from_pair(pair, [pair], address=TOKEN_ADDRESS)

    def test_fills_missing_description_and_socials(self):
        snapshot = self._snapshot(info={})
        assert snapshot.description == "" and snapshot.socials == []

        profile = data_fetchers.normalize_token_profile(token_profile_item())
        data_fetchers.enrich_snapshot_with_profile(snapshot, profile)

        assert snapshot.description.startswith("Brett is Pepe's")
        assert {link.kind for link in snapshot.socials} == {"twitter", "telegram", "website"}
        assert snapshot.image_url.endswith("icon.png")

    def test_does_not_overwrite_existing_pair_data(self):
        snapshot = self._snapshot()
        snapshot.description = "Original description"
        original_image = snapshot.image_url

        data_fetchers.enrich_snapshot_with_profile(
            snapshot, data_fetchers.normalize_token_profile(token_profile_item())
        )
        assert snapshot.description == "Original description"
        assert snapshot.image_url == original_image

    def test_does_not_duplicate_links_already_present(self):
        snapshot = self._snapshot(info={})
        snapshot.socials = [SocialLink(kind="twitter", url="https://x.com/basedbrett")]

        data_fetchers.enrich_snapshot_with_profile(
            snapshot, data_fetchers.normalize_token_profile(token_profile_item())
        )
        twitter_links = [link for link in snapshot.socials if link.kind == "twitter"]
        assert len(twitter_links) == 1

    def test_none_or_empty_profile_is_a_no_op(self):
        snapshot = self._snapshot()
        # to_dict() recomputes a live age, so compare only what enrichment writes.
        fields = ("description", "image_url")
        before = ({f: getattr(snapshot, f) for f in fields}, list(snapshot.socials))

        data_fetchers.enrich_snapshot_with_profile(snapshot, None)
        data_fetchers.enrich_snapshot_with_profile(snapshot, TokenProfile(address="0x1", chain="base"))

        assert ({f: getattr(snapshot, f) for f in fields}, list(snapshot.socials)) == before
