# Script thay thế Makefile cho Windows PowerShell
# Dùng khi máy chưa cài make
# Cách dùng: .\make.ps1 <lệnh>
# Ví dụ:     .\make.ps1 install
#            .\make.ps1 dev

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

switch ($Command) {
    "install" {
        Write-Host "Cai dat thu vien voi uv..." -ForegroundColor Cyan
        uv sync --all-extras
    }
    "dev" {
        Write-Host "Khoi dong FastAPI Backend tai http://localhost:8000 ..." -ForegroundColor Cyan
        uv run uvicorn vshield.api.app:app --reload --port 8000
    }
    "frontend" {
        Write-Host "Khoi dong React Frontend..." -ForegroundColor Cyan
        Set-Location frontend
        npm run dev
    }
    "docker" {
        Write-Host "Khoi dong toan bo he thong voi Docker Compose..." -ForegroundColor Cyan
        docker compose up --build
    }
    "train" {
        Write-Host "Bat dau training mo hinh..." -ForegroundColor Cyan
        uv run python src/vshield/training/train.py
    }
    "collect" {
        Write-Host "Mo webcam thu thap data..." -ForegroundColor Cyan
        uv run python scripts/collect_data.py
    }
    "test" {
        Write-Host "Chay unit tests..." -ForegroundColor Cyan
        uv run pytest tests/ -v
    }
    "format" {
        Write-Host "Format code..." -ForegroundColor Cyan
        uv run ruff format src/
    }
    "clean" {
        Write-Host "Don dep cache..." -ForegroundColor Cyan
        Get-ChildItem -Recurse -Directory -Name "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force ".pytest_cache" -ErrorAction SilentlyContinue
        Write-Host "Done!" -ForegroundColor Green
    }
    "help" {
        Write-Host ""
        Write-Host "V-Shield — Danh sach lenh:" -ForegroundColor Yellow
        Write-Host "  .\make.ps1 install    - Cai dat thu vien voi uv"
        Write-Host "  .\make.ps1 dev        - Chay Backend FastAPI"
        Write-Host "  .\make.ps1 frontend   - Chay Frontend React"
        Write-Host "  .\make.ps1 docker     - Chay toan bo he thong bang Docker"
        Write-Host "  .\make.ps1 train      - Train lai mo hinh CNN"
        Write-Host "  .\make.ps1 collect    - Mo webcam thu thap data"
        Write-Host "  .\make.ps1 test       - Chay unit tests"
        Write-Host "  .\make.ps1 format     - Format code"
        Write-Host "  .\make.ps1 clean      - Xoa cache"
        Write-Host ""
    }
    default {
        Write-Host "Lenh '$Command' khong ton tai. Dung '.\make.ps1 help' de xem danh sach lenh." -ForegroundColor Red
    }
}
