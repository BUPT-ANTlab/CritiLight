# CritiLight

This repository is organized around two public workflows:

- `assessor/`: build the assessor dataset and train the assessor model.
- `critilight/`: run the CritiLight online decision pipeline.

## Repository Structure

- `map/`: shared SUMO road networks and route files.
- `assessor/`: dataset construction, baseline policies, training, and exported assessor artifacts.
- `critilight/`: CritiLight runtime pipeline, LLM interface, and runtime traffic simulation code.
- `outputs/`: runtime logs, SUMO outputs, and training logs.

## Environment

1. Install Python dependencies from `requirements.txt`.
2. Install SUMO and set the `SUMO_HOME` environment variable.
3. If you run the local LLM path, place the model files under `critilight/model/` or set `CRITILIGHT_LLM_MODEL_PATH`.
4. The local backend uses standard `transformers` loading and does not enable extra deployment acceleration or 4-bit runtime compression.

## Public Entrypoints

- Build assessor dataset: `python assessor/build_dataset.py`
- Train assessor: `python assessor/train.py`
- Run CritiLight: `python critilight/run_critilight.py`