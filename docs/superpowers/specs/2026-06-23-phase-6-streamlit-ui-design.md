# Phase 6 — Streamlit Receipt Upload UI

**Date:** 2026-06-23
**Status:** Approved

## Goal

Give users a simple web interface to upload a receipt image and see the
extracted, validated result rendered nicely — instead of calling `POST /extract`
directly and reading raw JSON.

## Approach

A Streamlit app that is a **pure consumer** of the existing FastAPI `/extract`
endpoint. No changes to the extraction backend. It runs as a new `ui` service in
`docker-compose` (port 8501) and POSTs uploads to `api:8000` over HTTP.

```
browser :8501  ──>  Streamlit (ui)  ──HTTP POST /extract──>  FastAPI (api:8000)
```

## File layout

- `docintel/src/docintel/ui/__init__.py`
- `docintel/src/docintel/ui/client.py` — pure, testable helpers:
  - `extract_receipt(base_url, timeout_s, filename, content_type, data) -> dict`
    — httpx POST to `/extract`; returns parsed JSON, raises a typed error
    (`ExtractError`) carrying a user-facing message on non-2xx / transport failure.
  - `format_money(value: float | None, currency: str) -> str` — pure formatting.
  - `line_item_rows(document: dict) -> list[dict]` — flatten line items to table rows.
- `docintel/src/docintel/ui/app.py` — Streamlit entrypoint; thin `st.*` rendering
  that calls the helpers above.
- `docintel/Dockerfile.ui` — lightweight image installing only `.[ui]`
  (no torch / doctr / onnx), runs `streamlit run`.

Rationale: keeping HTTP + formatting in `client.py` as pure functions makes them
unit-testable; only the thin `st.*` layer stays untested.

## Configuration

Add to `Settings` (`docintel/src/docintel/config.py`), env-prefixed `DOCINTEL_`:

- `ui_api_base_url: str = "http://localhost:8000"` — compose overrides to
  `http://api:8000`.
- `ui_request_timeout_s: float = 120.0` — CPU OCR+KIE is slow.

No hardcoded constants in the UI code; values come from `get_settings()`.

## Dependencies

New optional extra in `pyproject.toml`:

```toml
ui = ["streamlit>=1.37", "httpx>=0.27"]
```

httpx is chosen over requests because it is already used in dev and provides
`MockTransport` for clean tests.

## Result view

Layout: title + `st.file_uploader` (accepts png/jpeg). After a successful
extract, two columns:

- **Left:** uploaded image preview (`st.image`).
- **Right:**
  - **Validation badge** — `st.success("Valid")` when `validation.ok`, else
    `st.error` listing each error; `st.warning` for any warnings.
  - **Totals summary** — `st.metric` for subtotal / tax / service / total,
    formatted with `format_money` + currency.
  - **Line items table** — `st.dataframe` built from `line_item_rows`
    (name / qty / unit price / price).
  - `st.expander` with the raw JSON for debugging.

## Error handling

- No file selected → no request sent; show an inline prompt.
- Non-image content type → rejected client-side before sending.
- API returns 4xx/5xx → `st.error` with the API's `detail`.
- Connection refused / timeout → `st.error` ("API unreachable or timed out").

These map to `ExtractError` raised by `extract_receipt`, rendered by `app.py`.

## Testing

`docintel/tests/test_ui_client.py`:

- `format_money` — value formatting incl. `None` and thousands separators.
- `line_item_rows` — rows built from a sample document, including missing fields.
- `extract_receipt` against an httpx `MockTransport`: success (200 → dict),
  415 and 500 (→ `ExtractError` with detail), and transport/timeout error.

The `app.py` rendering layer is thin and verified manually via `streamlit run`.

## docker-compose

Add a `ui` service:

```yaml
ui:
  build:
    context: .
    dockerfile: Dockerfile.ui
  container_name: docintel-ui
  ports:
    - "8501:8501"
  environment:
    DOCINTEL_UI_API_BASE_URL: http://api:8000
  depends_on:
    - api
```

## Out of scope

- Bounding-box overlay on the image (would require `/extract` to return word
  predictions; deferred).
- Authentication, history browsing, or editing extracted fields.
