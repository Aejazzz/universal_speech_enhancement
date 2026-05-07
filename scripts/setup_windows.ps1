$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Push-Location frontend
npm install
Pop-Location

Write-Host "Setup complete. Start backend with: uvicorn backend.app.main:app --reload"
