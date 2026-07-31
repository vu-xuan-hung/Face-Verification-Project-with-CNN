.PHONY: install dev train collect collect-real collect-fake test format help

install:
	uv sync --all-extras

dev:
	uv run uvicorn vshield.api.app:app --reload --port 8000

frontend:
	cd frontend && npm run dev

train:
	uv run python src/vshield/training/train.py

collect:
	uv run python -m vshield.data.collector

collect-real:
	uv run python -m vshield.data.collector --class-id 1 --output data/DataCollect/real

collect-fake:
	uv run python -m vshield.data.collector --class-id 0 --output data/DataCollect/fake

test:
	uv run pytest tests/

docker:
	docker compose up --build

format:
	uv run ruff format src/

clean:
	rm -rf __pycache__ .pytest_cache

help:
	@echo "Danh sách các lệnh:"
	@echo "  make install  - Cài đặt thư viện với uv"
	@echo "  make dev      - Chạy Backend (FastAPI)"
	@echo "  make frontend - Chạy Frontend (React)"
	@echo "  make docker   - Chạy toàn bộ hệ thống bằng Docker"
	@echo "  make train    - Huấn luyện mô hình Anti-Spoofing"
	@echo "  make collect  - Thu thập dữ liệu khuôn mặt"
	@echo "  make test     - Chạy unit tests"
	@echo "  make format   - Định dạng code"
	@echo "  make clean    - Xóa các file rác"
