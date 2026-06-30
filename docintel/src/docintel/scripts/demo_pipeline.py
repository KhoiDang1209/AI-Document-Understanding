"""End-to-end HTTP demo: extract a contract, then ask and run the agent over it.

Drives a running DocIntel API over HTTP (no in-process pipeline imports), so it
exercises the same surface a real client would. Start the stack first, then::

    docintel-demo                      # synthesizes a sample contract PDF
    docintel-demo --pdf path/to.pdf    # use your own contract

Without an LLM configured, ``/ask`` and ``/agent`` degrade to citations-only;
this script prints whatever the API returns.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import httpx

from docintel.config import get_settings

_PDF_TYPE = "application/pdf"


def build_sample_pdf() -> bytes:
    """Synthesize a tiny multi-clause contract PDF in memory (no committed binary)."""
    import fitz  # pymupdf

    text = (
        "MASTER SERVICES AGREEMENT\n\n"
        'This Agreement is entered into by and between Acme Corporation ("Provider") '
        'and Globex Inc. ("Customer").\n\n'
        "Governing Law. This Agreement shall be governed by the laws of the State of "
        "New York.\n\n"
        "Term. This Agreement expires on 2030-12-31 and renews automatically for "
        "successive one-year terms unless terminated.\n"
    )
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def _print_stage(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_demo(base_url: str, timeout_s: float, pdf: bytes) -> int:
    """Run extract -> ask -> agent against a live API; return a process exit code."""
    base = base_url.rstrip("/")
    try:
        _print_stage("C1 - Extract clauses (POST /contracts/extract)")
        extract = httpx.post(
            f"{base}/contracts/extract",
            files={"file": ("sample_contract.pdf", pdf, _PDF_TYPE)},
            timeout=timeout_s,
        )
        extract.raise_for_status()
        doc: dict[str, Any] = extract.json()
        contract_id = doc["id"]
        print(f"contract_id={contract_id} source={doc['source']} clauses={len(doc['clauses'])}")
        print(f"derived fields: {sorted(doc['derived'])}")

        _print_stage("C2/C3 - Ask a grounded question (POST /ask)")
        ask = httpx.post(
            f"{base}/ask",
            json={"question": "What is the governing law?", "contract_id": contract_id},
            timeout=timeout_s,
        )
        ask.raise_for_status()
        ask_body = ask.json()
        print(f"answer: {ask_body['answer']}")
        print(
            f"generation_skipped={ask_body['generation_skipped']} "
            f"citations={len(ask_body['citations'])}"
        )

        _print_stage("C4 - Run the agent (POST /agent)")
        agent = httpx.post(
            f"{base}/agent",
            json={
                "task": "Summarize the parties and governing law.",
                "contract_id": contract_id,
            },
            timeout=timeout_s,
        )
        agent.raise_for_status()
        agent_body = agent.json()
        print(f"status={agent_body['status']} steps={agent_body['steps']}")
        print(f"answer: {agent_body['answer']}")
        print(f"citations={len(agent_body['citations'])} trace_id={agent_body['trace_id']}")
    except httpx.HTTPError as exc:
        print(
            f"\nERROR: could not complete the demo against {base_url}. "
            f"Is the API running? ({exc})",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    """Parse args, obtain a PDF (synthesized by default), and run the demo."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="DocIntel C1-C4 end-to-end HTTP demo.")
    parser.add_argument("--base-url", default=settings.ui_api_base_url)
    parser.add_argument(
        "--pdf", default=None, help="Path to a contract PDF (default: synthesized)."
    )
    parser.add_argument("--timeout", type=float, default=settings.ui_request_timeout_s)
    args = parser.parse_args()

    if args.pdf:
        with open(args.pdf, "rb") as handle:
            pdf = handle.read()
    else:
        pdf = build_sample_pdf()
    return run_demo(args.base_url, args.timeout, pdf)


if __name__ == "__main__":
    raise SystemExit(main())
