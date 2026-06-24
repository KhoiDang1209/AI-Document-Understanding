# Launch the DocIntel Streamlit UI locally.
# The UI is not containerised; it talks to the API (run via `docker compose up`)
# over HTTP. The API base URL comes from Settings (DOCINTEL_UI_API_BASE_URL,
# default http://localhost:8000); override the env var before running to change it.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found on PATH. Install it from https://docs.astral.sh/uv/ (or: pip install uv), then re-run."
    exit 1
}

# Additive install of the UI extra (streamlit, httpx); safe to re-run.
uv pip install -q ".[ui]"
uv run --no-sync streamlit run src/docintel/ui/app.py
