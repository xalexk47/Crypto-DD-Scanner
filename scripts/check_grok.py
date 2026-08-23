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
        print(f"{INFO} An auth error here means the key is wrong or revoked.")
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
