"""Streamlit entrypoint: upload a receipt, call ``/extract``, render the result.

Run with ``streamlit run src/docintel/ui/app.py``. The API base URL and request
timeout come from :class:`docintel.config.Settings` (``DOCINTEL_`` env prefix).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from docintel.config import get_settings
from docintel.ui.client import ExtractError, extract_receipt, format_money, line_item_rows

_ACCEPTED_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}


def _render_validation(document: dict[str, Any]) -> None:
    """Render the validation badge plus any errors and warnings."""
    validation = document.get("validation", {})
    if validation.get("ok"):
        st.success("Valid")
    else:
        st.error("Validation failed")
    for issue in validation.get("errors", []):
        st.error(f"{issue.get('rule', 'error')}: {issue.get('message', '')}")
    for issue in validation.get("warnings", []):
        st.warning(f"{issue.get('rule', 'warning')}: {issue.get('message', '')}")


def _render_totals(document: dict[str, Any]) -> None:
    """Render subtotal / tax / service / total as metrics."""
    currency = document.get("currency", "")
    cols = st.columns(4)
    for col, label, key in (
        (cols[0], "Subtotal", "subtotal"),
        (cols[1], "Tax", "tax"),
        (cols[2], "Service", "service"),
        (cols[3], "Total", "total"),
    ):
        col.metric(label, format_money(document.get(key), currency))


def _render_result(document: dict[str, Any]) -> None:
    """Render the full result panel for an extracted document."""
    _render_validation(document)
    _render_totals(document)
    rows = line_item_rows(document)
    if rows:
        st.subheader("Line items")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No line items detected.")
    with st.expander("Raw JSON"):
        st.json(document)


def main() -> None:
    """Build the Streamlit page."""
    settings = get_settings()
    st.set_page_config(page_title="DocIntel — Receipt Extraction", layout="wide")
    st.title("Receipt Extraction")
    st.caption("Upload a receipt image to extract its structured fields.")

    uploaded = st.file_uploader("Receipt image", type=list(_ACCEPTED_TYPES))
    if uploaded is None:
        st.info("Choose a PNG or JPEG receipt to begin.")
        return

    suffix = uploaded.name.rsplit(".", 1)[-1].lower()
    content_type = _ACCEPTED_TYPES.get(suffix, uploaded.type or "application/octet-stream")
    data = uploaded.getvalue()

    image_col, result_col = st.columns(2)
    image_col.image(data, caption=uploaded.name, use_container_width=True)

    with result_col:
        with st.spinner("Extracting…"):
            try:
                document = extract_receipt(
                    settings.ui_api_base_url,
                    settings.ui_request_timeout_s,
                    uploaded.name,
                    content_type,
                    data,
                )
            except ExtractError as exc:
                st.error(str(exc))
                return
        _render_result(document)


if __name__ == "__main__":
    main()
