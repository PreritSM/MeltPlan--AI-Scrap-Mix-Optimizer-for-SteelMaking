.PHONY: setup data profile fit backtest test api web
setup:
	cd backend && uv sync --group dev && cd ../frontend && npm install
profile:
	cd backend && uv run python -m scrapchef.data.profile
data:
	cd backend && uv run python -m scrapchef.data.ingest
fit: data
	cd backend && uv run python -m scrapchef.intelligence.estimator.validate && uv run python -m scrapchef.intelligence.energy
backtest:
	cd backend && uv run python -m scrapchef.intelligence.backtest
test:
	cd backend && uv run pytest -q && cd ../frontend && npx tsc --noEmit
api:
	cd backend && uv run uvicorn scrapchef.api.main:app --port 8000
web:
	cd frontend && npm run dev
