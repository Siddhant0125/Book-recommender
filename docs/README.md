# Mood-based Book Recommender

This repository implements a two-tower (user–item) recommender focused on moods extracted from text. It includes:
- A mood tagging pipeline for books
- Preprocessing to build vocabularies and fixed-size features
- A PyTorch two-tower model and training loop
- A Streamlit app to test recommendations from a mood sentence

## Quick start (Windows cmd.exe)

Create a virtual environment and install dependencies (install PyTorch separately to suit your platform):
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Optional: install NLP packages for better mood detection (already listed in requirements; ensure installed on your system):
```
pip install transformers sentence-transformers
```

Run the Streamlit app:
```
streamlit run test_sentence.py
```

Run mood tagging (books_enriched → books_with_moods):
```
python .\mood_classifier_tagger\mood_tagger.py --csv dataset\books_enriched.csv --out dataset\books_with_moods.csv --top-k 3
```

Train the two-tower model (optional):
```
python .\models\train_two_tower.py --data-dir dataset --preprocessed-dir preprocessed
```

## Documentation
- Process flow (end-to-end): `docs/process_flow.md`
- Model code: `models/two_tower_model.py`
- Training script: `models/train_two_tower.py`
- Mood tagging: `mood_classifier_tagger/`

## Artifacts
- Preprocessed features: `preprocessed/`
- Datasets: `dataset/`
- Trained checkpoint (example): `models/two_tower.pt`

If you change vocabularies or caps during preprocessing, ensure they align with the checkpoint and the Streamlit app settings.
