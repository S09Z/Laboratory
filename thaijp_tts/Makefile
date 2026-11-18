.PHONY: sync sync-train test lint format notebooks zip
sync:            ## dev env (.venv) for inference / tests
	uv sync --extra g2p
sync-train:      ## training deps (run on a CUDA machine)
	uv sync --extra g2p --group train
test:
	uv run --extra g2p pytest -m "not network" -q
lint:
	uv run --extra g2p ruff check .
format:
	uv run --extra g2p ruff format .
notebooks:
	uv run --extra g2p jupytext --to ipynb notebooks_src/ThaiJP_TTS_AllInOne.py -o notebooks/ThaiJP_TTS_AllInOne.ipynb
zip: notebooks
	cd .. && rm -f thaijp_tts.zip && zip -qr thaijp_tts.zip thaijp_tts -x 'thaijp_tts/.git/*' 'thaijp_tts/.venv/*' 'thaijp_tts/.ruff_cache/*' 'thaijp_tts/.superpowers/*' '*/__pycache__/*' '*/.pytest_cache/*'
