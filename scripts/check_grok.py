#!/usr/bin/env python3
"""Diagnose the Grok / xAI connection from your own machine.

The app's X-mindshare feature depends on three things this repo cannot verify
for you: that your key works, which model ids your account can call, and
whether the server-side ``x_search`` tool is accepted. This script checks all
three and prints exactly what to change if something is off.

    python scripts/check_grok.py

Costs a few tenths of a cent: one tiny completion, plus one X search if that
step gets that far.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.mindshare import MINDSHARE_SYSTEM_PROMPT, GrokMindshareClient  # noqa: E402
from src.models import TokenSnapshot  # noqa: E402

OK, BAD, INFO = "  ✅", "  ❌", "  •"


def explain(exc: Exception) -> list:
    """Turn a vendor exception into advice that fits the actual failure.

    Getting this wrong is worse than saying nothing: a billing error reported
    as "your key is wrong" sends people to regenerate a perfectly good key.
    """
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    lines = []

    if status == 401 or "invalid_api_key" in text or "incorrect api key" in text:
        lines += [
            f"{INFO} The key itself was rejected — wrong, revoked, or truncated.",
            f"{INFO} Generate a fresh one at https://console.x.ai/ and update .env.",
        ]
    elif status == 403 or "credit" in text or "permission-denied" in text or "license" in text:
        lines += [
            f"{INFO} Your key is VALID — this is a billing/permission problem, not auth.",
            f"{INFO} The xAI team this key belongs to has no credits or licenses yet.",
            f"{INFO} Open the console URL in the error above and add credits to THAT team.",
            f"{INFO} Watch out: sign-up credits often sit on a different team than a",
            f"{INFO}   newly created one. If you have more than one team, either add",
            f"{INFO}   credits here or make a new key under the team that has them.",
            f"{INFO} Everything else in this app works without Grok — the ensemble and",
            f"{INFO}   X mindshare are optional layers on top.",
        ]
    elif status == 429 or "rate limit" in text:
        lines += [f"{INFO} Rate limited. Wait a moment and re-run this script."]
    elif status == 404 or "does not exist" in text or "not found" in text:
        lines += [
            f"{INFO} The model id was not recognised by your account.",
            f"{INFO} Use one of the ids listed in step 3 and set it in .env.",
        ]
    else:
        lines += [f"{INFO} Unexpected failure — paste this output back to Claude."]
    return lines


def main() -> int:
    print("\nMemeDD — Grok / xAI connection check")
    print("=" * 52)

    # 1. key -------------------------------------------------------------
    print("\n1. API key")
    if not config.XAI_API_KEY:
        print(f"{BAD} XAI_API_KEY is not set.")
        print(f"{INFO} Add it to .env:  XAI_API_KEY=xai-...")
        print(f"{INFO} Then restart — .env is only read at startup.")
        return 1
    key = config.XAI_API_KEY
    print(f"{OK} Found: {key[:8]}...{key[-4:]} (from .env)")

    # 2. SDK -------------------------------------------------------------
    print("\n2. openai SDK (used for xAI's OpenAI-compatible endpoint)")
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        print(f"{BAD} Not installed.  Fix:  pip install openai")
        return 1
    print(f"{OK} openai {getattr(openai, '__version__', '?')}")

    client = OpenAI(api_key=key, base_url=config.XAI_BASE_URL, timeout=60)
    print(f"{INFO} Endpoint: {config.XAI_BASE_URL}")

    # 3. which models can this key actually call? ------------------------
    print("\n3. Models available to your key")
    available = []
    try:
        available = sorted(m.id for m in client.models.list().data)
        for model_id in available:
            print(f"{INFO} {model_id}")
    except Exception as exc:
        print(f"{BAD} Could not list models: {type(exc).__name__}: {exc}")
        for line in explain(exc):
            print(line)
        return 1

    for label, configured in (("ensemble", config.XAI_MODEL), ("X search", config.X_SEARCH_MODEL)):
        if not available:
            break
        if configured in available:
            print(f"{OK} {label} model '{configured}' is available.")
        else:
            print(f"{BAD} {label} model '{configured}' is NOT in your account's list.")
            env = "MEMEDD_XAI_MODEL" if label == "ensemble" else "MEMEDD_X_SEARCH_MODEL"
            print(f"{INFO} Set {env}=<one of the ids above> in .env")

    # 4. plain completion -------------------------------------------------
    print(f"\n4. Basic completion ({config.XAI_MODEL})")
    try:
        response = client.chat.completions.create(
            model=config.XAI_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=10,
        )
        print(f"{OK} Replied: {response.choices[0].message.content!r}")
    except Exception as exc:
        print(f"{BAD} {type(exc).__name__}: {exc}")
        for line in explain(exc):
            print(line)
        return 1

    # 5. the server-side X search tool ------------------------------------
    print(f"\n5. X search tool ('{config.X_SEARCH_TOOL_TYPE}' on {config.X_SEARCH_MODEL})")
    grok = GrokMindshareClient()
    tool = grok._x_search_tool(config.X_SEARCH_WINDOW_HOURS)
    try:
        response = client.chat.completions.create(
            model=config.X_SEARCH_MODEL,
            messages=[
                {"role": "system", "content": MINDSHARE_SYSTEM_PROMPT},
                {"role": "user", "content": "Search X for posts about $BRETT on Base in the last 24 hours."},
            ],
            tools=[tool],
            max_tokens=800,
        )
        text = (response.choices[0].message.content or "").strip()
        print(f"{OK} Tool accepted. First 200 chars of the answer:")
        print(f"     {text[:200]}")
    except Exception as exc:
        print(f"{BAD} {type(exc).__name__}: {exc}")
        for line in explain(exc):
            print(line)
        print(f"{INFO} The app falls back to a non-live answer, clearly labelled.")
        print(f"{INFO} If the tool name or model is wrong, set MEMEDD_X_SEARCH_TOOL /")
        print(f"{INFO} MEMEDD_X_SEARCH_MODEL in .env — no code change needed.")
        print(f"{INFO} Paste this error back to Claude and it can correct the shape.")
        return 2

    # 6. the real thing ---------------------------------------------------
    print("\n6. Full mindshare fetch through the app's own code path")
    snapshot = TokenSnapshot(
        address="0x532f27101965dd16442E59d40670FaF5eBB142E4",
        chain="base", name="Based Brett", symbol="BRETT",
    )
    report = grok.fetch(snapshot)
    if report.available:
        print(f"{OK} {report.headline}")
        print(f"{INFO} attention {report.mindshare_score}/100 · trend {report.trend} "
              f"· organic {report.is_organic} · {report.latency_ms}ms")
        if report.summary:
            print(f"{INFO} {report.summary[:180]}")
        for post in report.sample_posts[:3]:
            print(f"{INFO} @{post.handle}: {post.text[:90]}")
        if not report.is_live:
            print(f"{BAD} NOT live — this is model knowledge. See step 5.")
    else:
        print(f"{BAD} {report.error}")
        return 2

    print("\n" + "=" * 52)
    print("All good. Turn on 'Query X via Grok' in the sidebar.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
