# AI4Bharat Indic ASR (CPU Setup)

![Python](https://img.shields.io/badge/python-3.10-blue)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange)
![NeMo](https://img.shields.io/badge/NVIDIA-NeMo-green)
![License](https://img.shields.io/badge/license-MIT-blue)

A **fully working CPU-only setup** for running **AI4Bharat Indic ASR models** using NVIDIA NeMo.

This repository provides a **stable environment, patched dependencies, and example scripts** for running speech recognition models locally without a GPU.

Supports:

* Gujarati ASR
* Multilingual Indic ASR
* Real-time streaming transcription
* CTC and RNNT decoding

---

# Supported Models

| Model                                     | Type         | Languages           |
| ----------------------------------------- | ------------ | ------------------- |
| `indicconformer_stt_gu_hybrid_rnnt_large` | Gujarati     | Gujarati            |
| `indic-conformer-600m-multilingual`       | Multilingual | 20+ Indic languages |

These models are hosted on HuggingFace and require authentication.

---

# Features

* CPU-only inference
* Fully automated setup
* Patched NeMo compatibility
* Gujarati speech recognition
* Multilingual ASR
* Real-time streaming transcription
* 22 Indic languages supported
* Continuous and VAD-based transcription modes

---

# Demo

### Example output

```
Loading AI4Bharat Gujarati model on CPU...

CTC Decoder  : આજે હવામાન ખુબ સારું છે
RNN-T Decoder: આજે હવામાન ખુબ સારું છે
```

---

# Project Structure

```
AI4Bharat-CPU/
│
├── setup.sh
├── requirements.txt
│
├── ai4bharat-gu.py
├── ai4bharat-mul.py
├── live.py
│
└── README.md
```

| File               | Description                          |
| ------------------ | ------------------------------------ |
| `setup.sh`         | Complete automated environment setup |
| `requirements.txt` | Fixed dependency versions            |
| `ai4bharat-gu.py`  | Gujarati ASR example                 |
| `ai4bharat-mul.py` | Multilingual ASR example             |
| `live.py`          | Real-time streaming ASR              |

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI4Bharat-CPU.git
cd AI4Bharat-CPU
```

---

# 2. Run setup

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will automatically:

* create a virtual environment
* install CPU-only PyTorch
* clone and patch AI4Bharat NeMo
* install dependencies
* fix HuggingFace compatibility issues

The script performs the full environment configuration automatically. 

---

# 3. Activate environment

```bash
source nemo/bin/activate
```

---

# 4. Login to HuggingFace

The models are **gated**, so you must login first.

```bash
huggingface-cli login
```

---

# Running Examples

## Gujarati ASR

Run:

```bash
python ai4bharat-gu.py
```

The script loads the Gujarati Conformer model and runs both decoders:

* CTC
* RNNT

Example model loading:

```python
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large",
    map_location="cpu"
)
```

The script then transcribes a WAV file using both decoding methods. 

---

# Multilingual ASR

Run:

```bash
python ai4bharat-mul.py
```

This loads the **Indic Conformer 600M multilingual model** and performs speech recognition.

The script:

* loads an audio file
* converts it to mono
* resamples to **16kHz**
* performs CTC and RNNT decoding

Example usage in the script: 

```python
transcription_ctc = model(wav, "gu", "ctc")
```

---

# Real-Time Streaming ASR

Run:

```bash
python live.py
```

Features include:

* microphone streaming
* language selection
* CPU/GPU auto detection
* real-time transcription
* transcript saving
* continuous or utterance modes

Example:

```bash
python live.py --utterance --save
```

This enables:

* VAD-based utterance segmentation
* automatic transcript saving

The streaming script supports **22 Indic languages** via AI4Bharat models. 

---

# Audio Requirements

Input audio should be:

```
Format: WAV
Channels: Mono
Sample Rate: 16000 Hz
```

If the sample rate differs, the script automatically resamples it.

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

All versions are pinned in:

```
requirements.txt
```

to avoid dependency conflicts. 

---

# Model Download

The first run downloads the model automatically.

Approximate size:

```
~1.8 GB
```

Cached in:

```
~/.cache/huggingface
```

---

# Troubleshooting

### Cannot access model

Run:

```bash
huggingface-cli login
```

Also make sure you **accepted the model license** on HuggingFace.

---

### Python version issues

Use Python **3.10**.

```bash
python3.10 --version
```

---

### Microphone errors

Install PortAudio:

```
sudo apt install portaudio19-dev
```

---

# License

Model weights belong to **AI4Bharat** and follow their respective licenses.

See:

* AI4Bharat repositories
* HuggingFace model pages

---

# Acknowledgements

* AI4Bharat
* NVIDIA NeMo
* HuggingFace
* PyTorch

---

# Related Repository

GPU version:

```
https://github.com/ujjval1003/ai4bharat-asr-gpu
```