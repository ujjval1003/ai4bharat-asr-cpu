
---

# AI4Bharat Indic ASR (CPU Setup)

![Python](https://img.shields.io/badge/python-3.10-blue)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange)
![NeMo](https://img.shields.io/badge/NVIDIA-NeMo-green)
![Docker](https://img.shields.io/badge/docker-supported-blue)
![Docker](https://img.shields.io/docker/pulls/ujjvalpatel1003/ai4bharat-asr-cpu)
![Docker Image](https://img.shields.io/docker/image-size/ujjvalpatel1003/ai4bharat-asr-cpu/latest)
![License](https://img.shields.io/badge/license-MIT-blue)

A **CPU-only setup** for running **AI4Bharat Indic ASR models** using **NVIDIA NeMo**.

This repository provides a **stable, patched environment** to run AI4Bharat models locally **without requiring a GPU**, supporting **real-time streaming transcription and multilingual speech recognition**.

---

# Tested Hardware

This project was tested on the following system.

| Component | Value                      |
| --------- | -------------------------- |
| CPU       | Intel / AMD Multi-core CPU |
| GPU       | Not required               |
| PyTorch   | CPU build                  |
| Python    | 3.10                       |
| Framework | NVIDIA NeMo                |

The setup script installs **CPU-only PyTorch automatically**.

---

# Supported Models

| Model                                     | Type         | Languages           |
| ----------------------------------------- | ------------ | ------------------- |
| `indicconformer_stt_gu_hybrid_rnnt_large` | Gujarati     | Gujarati            |
| `indic-conformer-600m-multilingual`       | Multilingual | 20+ Indic languages |

The streaming ASR script supports **22 Indian languages**.

---

# Features

* CPU-only inference
* Real-time streaming ASR
* Terminal live transcription
* Gujarati speech recognition
* Multilingual ASR
* 22 Indic languages supported
* Continuous transcription mode
* VAD-based utterance segmentation
* Transcript export
* Docker support
* HuggingFace model caching

---

# Project Structure

```
AI4Bharat-CPU/
│
├── setup.sh
├── requirements.txt
├── Dockerfile
│
├── ai4bharat-gu.py
├── ai4bharat-mul.py
├── live.py
│
└── README.md
```

| File             | Description                    |
| ---------------- | ------------------------------ |
| setup.sh         | Complete CPU environment setup |
| requirements.txt | Dependency versions            |
| Dockerfile       | CPU Docker container           |
| ai4bharat-gu.py  | Gujarati ASR example           |
| ai4bharat-mul.py | Multilingual ASR example       |
| live.py          | Terminal real-time ASR         |

---

# PyTorch CPU Installation

The setup script installs **CPU-only PyTorch**.

Example command:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

This avoids installing CUDA dependencies and keeps the environment lightweight.

---

# Installation (Local CPU Setup)

## 1 Clone repository

```
git clone https://github.com/ujjval1003/ai4bharat-asr-cpu.git
cd ai4bharat-asr-cpu
```

---

## 2 Run setup script

```
chmod +x setup.sh
./setup.sh
```

The setup script automatically:

* creates a virtual environment
* installs CPU-only PyTorch
* clones AI4Bharat NeMo
* applies compatibility patches
* installs dependencies
* verifies NeMo installation

---

## 3 Activate environment

```
source nemo/bin/activate
```

---

## 4 Login to HuggingFace

The models are gated.

```
huggingface-cli login
```

---

# Run Gujarati ASR

```
python ai4bharat-gu.py
```

Example:

```python
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large",
    map_location="cpu"
)
```

The model will run entirely on CPU.

---

# Run Multilingual ASR

```
python ai4bharat-mul.py
```

The script:

* loads audio
* converts to mono
* resamples to **16 kHz**
* runs **CTC** and **RNNT** decoding

Example:

```
transcription_ctc = model(wav, "gu", "ctc")
```

---

# Real-Time Streaming ASR (Terminal)

```
python live.py
```

Features:

* microphone streaming
* language selection
* real-time transcription
* optional transcript saving

Example:

```
python live.py --utterance --save
```

---

# Docker Support

You can run the entire system using Docker.

## Build Docker Image

```
docker build -t ai4bharat-asr-cpu .
```

---

## Run Container

```
docker run -it -p 7860:7860 ai4bharat-asr-cpu
```

Explanation:

| Flag                                               | Purpose                     |
| -------------------------------------------------- | --------------------------- |
| `-p 7860:7860`                                     | Exposes streaming interface |
| `-i`                                               | Keeps input open            |
| `-t`                                               | Gives a terminal interface  |

---

# Docker Hub Image

You can pull the prebuilt image (~2GB).

```
docker pull ujjvalpatel1003/ai4bharat-asr-cpu
```

Run it:

```
docker run -it -p 7860:7860 ujjvalpatel1003/ai4bharat-asr-cpu
```

---

# Model Download

Models download automatically on first run.

Approximate size:

```
~1.8 GB
```

Cached in:

```
~/.cache/huggingface
```

If using Docker without volume mounting, the model will be stored in:

```
/root/.cache/huggingface
```

Mounting the cache directory prevents repeated downloads.

---

# Audio Requirements

Input audio must be:

```
Format: WAV
Channels: Mono
Sample rate: 16000 Hz
```

Scripts automatically resample audio if needed.

---

# Dependencies

Main dependencies include:

* PyTorch (CPU)
* NVIDIA NeMo
* Transformers
* HuggingFace Hub
* ONNXRuntime
* TorchCodec
* SoundDevice

Pinned versions are listed in:

```
requirements.txt
```

to avoid compatibility issues.

---

# Related Repository

GPU version:

[https://github.com/ujjval1003/ai4bharat-asr-gpu](https://github.com/ujjval1003/ai4bharat-asr-gpu)

---

# License

MIT License
