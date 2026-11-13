.PHONY: test notebooks zip
test:
	python3 -m pytest -m "not network" -q
notebooks:
	python3 -m jupytext --to ipynb notebooks_src/ThaiJP_TTS_AllInOne.py -o notebooks/ThaiJP_TTS_AllInOne.ipynb
zip: notebooks
	cd .. && rm -f thaijp_tts.zip && zip -qr thaijp_tts.zip thaijp_tts -x 'thaijp_tts/.git/*' 'thaijp_tts/.superpowers/*' '*/__pycache__/*' '*/.pytest_cache/*'
