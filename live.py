"""
IndicConformer Real-Time Streaming ASR v7.2 — Definitive
========================================================
Google Voice Typing style (default) + Utterance Mode
Background inference thread • Auto GPU/CPU • Clean display • Zero spam

Changes over v7.1:
  1. time.sleep(0.01) added to print loop → even lower CPU in idle
  2. Save filename uses language name: transcript_gujarati_20260306_101200.txt
  3. Named constants for partial intervals (GPU vs CPU) instead of inline on_cpu flag

Install:
    pip install nemo_toolkit[asr] sounddevice soundfile numpy torch

Usage:
    python indic_streaming_asr.py                        # Continuous Live (Google style)
    python indic_streaming_asr.py --utterance            # Utterance / VAD mode
    python indic_streaming_asr.py --save                 # Save transcript to file
    python indic_streaming_asr.py --utterance --save     # Both
"""

import torch
import numpy as np
import sounddevice as sd
import queue
import threading
import time
import sys
import os
import tempfile
import soundfile as sf
import nemo.collections.asr as nemo_asr
import argparse
import logging
from datetime import datetime

# ── Suppress ALL NeMo / tqdm / PyTorch Lightning spam ──────────────────────
logging.getLogger("nemo").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

# ── Safe CPU tuning (harmless on GPU too) ───────────────────────────────────
torch.set_flush_denormal(True)          # avoids slow denormal FP ops on CPU
torch.set_num_threads(os.cpu_count())   # use all physical cores for CPU ops

# ====================== 22 LANGUAGES ======================
LANGUAGES = {
    "1":  ("Assamese",  "as",  "ai4bharat/indicconformer_stt_as_hybrid_rnnt_large"),
    "2":  ("Bengali",   "bn",  "ai4bharat/indicconformer_stt_bn_hybrid_rnnt_large"),
    "3":  ("Bodo",      "brx", "ai4bharat/indicconformer_stt_brx_hybrid_rnnt_large"),
    "4":  ("Dogri",     "doi", "ai4bharat/indicconformer_stt_doi_hybrid_rnnt_large"),
    "5":  ("Gujarati",  "gu",  "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large"),
    "6":  ("Hindi",     "hi",  "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large"),
    "7":  ("Kannada",   "kn",  "ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large"),
    "8":  ("Konkani",   "kok", "ai4bharat/indicconformer_stt_kok_hybrid_rnnt_large"),
    "9":  ("Kashmiri",  "ks",  "ai4bharat/indicconformer_stt_ks_hybrid_rnnt_large"),
    "10": ("Maithili",  "mai", "ai4bharat/indicconformer_stt_mai_hybrid_rnnt_large"),
    "11": ("Malayalam", "ml",  "ai4bharat/indicconformer_stt_ml_hybrid_rnnt_large"),
    "12": ("Manipuri",  "mni", "ai4bharat/indicconformer_stt_mni_hybrid_rnnt_large"),
    "13": ("Marathi",   "mr",  "ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large"),
    "14": ("Nepali",    "ne",  "ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large"),
    "15": ("Odia",      "or",  "ai4bharat/indicconformer_stt_or_hybrid_rnnt_large"),
    "16": ("Punjabi",   "pa",  "ai4bharat/indicconformer_stt_pa_hybrid_rnnt_large"),
    "17": ("Sanskrit",  "sa",  "ai4bharat/indicconformer_stt_sa_hybrid_rnnt_large"),
    "18": ("Santali",   "sat", "ai4bharat/indicconformer_stt_sat_hybrid_rnnt_large"),
    "19": ("Sindhi",    "sd",  "ai4bharat/indicconformer_stt_sd_hybrid_rnnt_large"),
    "20": ("Tamil",     "ta",  "ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large"),
    "21": ("Telugu",    "te",  "ai4bharat/indicconformer_stt_te_hybrid_rnnt_large"),
    "22": ("Urdu",      "ur",  "ai4bharat/indicconformer_stt_ur_hybrid_rnnt_large"),
}

# ====================== CONFIG ======================
SAMPLE_RATE              = 16000
CHANNELS                 = 1
CHUNK_MS                 = 100
CHUNK_SAMPLES            = int(SAMPLE_RATE * CHUNK_MS / 1000)

SILENCE_RMS_THRESH       = 0.010   # overridden at runtime by mic calibration
MAX_BUFFER_SEC           = 12
LONG_SILENCE_RESET_SEC   = 4.0
PARTIAL_INTERVAL_SEC     = 1.25    # GPU: snappy updates
CPU_PARTIAL_INTERVAL_SEC = 2.0     # CPU: prevents decode queue backup


# ====================== HELPERS ======================

def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def audio_to_wav_file(audio: np.ndarray) -> str:
    """Write float32 PCM to a temp WAV. NeMo prefers float32 — no clipping."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, audio, SAMPLE_RATE, subtype="FLOAT")
    return tmp.name


def transcribe(model, lang_id: str, decoder: str, audio: np.ndarray) -> str:
    """
    Transcribe float32 PCM array.
    - torch.inference_mode() — no autograd overhead, fastest on CPU and GPU
    - verbose=False           — kills tqdm "Transcribing: 100%|..." bars
    - return_hypotheses=False — stable across NeMo versions
    - Handles List[str] and List[Hypothesis] return types
    - Temp WAV always deleted in finally
    """
    if len(audio) / SAMPLE_RATE < 0.4:
        return ""
    wav_path = audio_to_wav_file(audio)
    try:
        model.cur_decoder = decoder
        with torch.inference_mode():
            results = model.transcribe(
                [wav_path],
                batch_size=1,
                language_id=lang_id,
                return_hypotheses=False,
                logprobs=False,
                verbose=False,
            )
        r = results[0]
        text = r.text if hasattr(r, "text") else str(r)
        return text.strip()
    except Exception as e:
        print(f"\n[transcribe error] {e}", file=sys.stderr)
        return ""
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def print_live(text: str):
    """Single continuously-updated cyan line — Google Voice Typing feel."""
    clear_line()
    sys.stdout.write(f"\r\U0001f3a4 \033[1;36m{text}\033[0m")
    sys.stdout.flush()


def print_utterance(text: str):
    """Bold green permanent line — utterance mode finalised result."""
    clear_line()
    print(f"\u2705 \033[1;32m{text}\033[0m")


# ====================== UI ======================

def select_language():
    """Numbered menu → (lang_name, lang_id, model_id)."""
    print("\n" + "═" * 65)
    print("  IndicConformer Real-Time Streaming ASR v7.2 — Definitive")
    print("═" * 65)
    for key, (name, code, _) in LANGUAGES.items():
        print(f" {key:>2}. {name:<12} [{code}]")
    print("═" * 65)
    while True:
        choice = input(" Enter number (1-22): ").strip()
        if choice in LANGUAGES:
            name, code, model_id = LANGUAGES[choice]
            print(f"\n \u2705 {name} ({code}) selected.\n")
            return name, code, model_id
        print(" Invalid choice. Try again.")


def select_decoder(device: str) -> str:
    """
    Device-aware decoder menu.
    GPU → RNNT default (fast enough, highest accuracy)
    CPU → CTC default (3-5× faster than RNNT on CPU)
    """
    print(" Choose decoder:\n")
    if device == "cuda":
        print(" 1. RNNT (recommended — highest accuracy on GPU)")
        print(" 2. CTC  (faster, slightly lower accuracy)\n")
        choice = input(" Enter 1 or 2 [default 1]: ").strip() or "1"
        dec = "rnnt" if choice == "1" else "ctc"
    else:
        print(" 1. CTC  (recommended — 3-5x faster on CPU)")
        print(" 2. RNNT (higher accuracy — slow on CPU)\n")
        choice = input(" Enter 1 or 2 [default 1]: ").strip() or "1"
        dec = "ctc" if choice == "1" else "rnnt"
    print(f" \u2705 {dec.upper()} decoder selected.\n")
    return dec


# ====================== MODEL ======================

def load_model(model_id: str):
    """
    Auto-detects CUDA. No forced CPU. No quantization.
    (Dynamic int8 quantization breaks NeMo conformers → NaN / garbage text.)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device     : {str(device).upper()}")
    print(f" Loading {model_id}... (first run ~1.8 GB)")
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    model.freeze()
    model = model.to(device)
    print(f" \u2705 Model loaded on {str(device).upper()}\n")
    return model, device


# ====================== STREAMING ENGINE ======================

class StreamingASR:
    """
    Three cooperating roles across two threads:

    Audio thread (sounddevice callback)
      → fills audio_q with raw float32 chunks; never blocked by inference.

    Decode thread (_decode_loop)
      → drains audio_q, manages rolling buffer, rate-limits decodes to
        partial_interval (1.25 s GPU / 2.0 s CPU), runs transcribe(),
        posts (text, is_final) tuples to result_q.

    Print loop (_print_loop, main thread)
      → drains result_q and updates the terminal display.
        time.sleep(0.01) in both the queue-empty and post-print paths
        keeps this thread from spinning wastefully.

    CONTINUOUS (default) — single growing cyan line, Google-style.
    UTTERANCE  (--utterance) — bold green finalised lines per utterance.
    """

    def __init__(self, model, lang_id: str, decoder: str,
                 continuous: bool, save_path: str | None, partial_interval: float):
        self.model            = model
        self.lang_id          = lang_id
        self.decoder          = decoder
        self.continuous       = continuous
        self.save_path        = save_path
        self.partial_interval = partial_interval

        self.audio_q  = queue.Queue()
        self.result_q = queue.Queue()
        self.running  = False

        self._last_saved_text = ""

    def _save(self, text: str):
        """Append timestamped line — only when text has changed."""
        if self.save_path and text and text != self._last_saved_text:
            ts = datetime.now().strftime("%H:%M:%S")
            with open(self.save_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {text}\n")
            self._last_saved_text = text

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_q.put(indata[:, 0].copy())

    def _decode_loop(self):
        buf              = np.array([], dtype=np.float32)
        last_decode      = 0.0
        last_speech_time = time.time()
        silence_cnt      = 0

        while self.running:
            try:
                chunk = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            now = time.time()
            buf = np.concatenate([buf, chunk])

            # Rolling buffer cap
            max_samples = int(MAX_BUFFER_SEC * SAMPLE_RATE)
            if len(buf) > max_samples:
                buf = buf[-max_samples:]

            # Trim to 3 s tail after long silence (continuous mode)
            if self.continuous and (now - last_speech_time > LONG_SILENCE_RESET_SEC):
                buf = buf[-int(SAMPLE_RATE * 3):]

            if now - last_decode < self.partial_interval:
                time.sleep(0.01)
                continue
            last_decode = now

            if self.continuous:
                # ── CONTINUOUS MODE ───────────────────────────────────────
                text = transcribe(self.model, self.lang_id, self.decoder, buf)
                if text:
                    last_speech_time = now
                    self.result_q.put((text, False))

            else:
                # ── UTTERANCE MODE ────────────────────────────────────────
                is_speech = rms(chunk) > SILENCE_RMS_THRESH
                if is_speech:
                    silence_cnt      = 0
                    last_speech_time = now
                    text = transcribe(self.model, self.lang_id, self.decoder, buf)
                    if text:
                        self.result_q.put((text, False))
                else:
                    silence_cnt += 1
                    # 32 chunks × 100 ms = ~3.2 s silence → finalise utterance
                    if silence_cnt >= 32 and len(buf) / SAMPLE_RATE >= 0.5:
                        final = transcribe(
                            self.model, self.lang_id, self.decoder, buf
                        )
                        if final:
                            self.result_q.put((final, True))
                        buf         = np.array([], dtype=np.float32)
                        silence_cnt = 0

            time.sleep(0.01)

    def _print_loop(self):
        print("═" * 65)
        mode = (
            "Continuous Live (Google Voice Typing style)"
            if self.continuous else
            "Utterance / VAD mode"
        )
        print(f" \U0001f3a4 {mode} — Speak now (Ctrl+C to quit)\n")

        while self.running:
            try:
                text, is_final = self.result_q.get(timeout=0.3)
            except queue.Empty:
                time.sleep(0.01)   # idle sleep — keeps print thread CPU near zero
                continue

            if is_final:
                print_utterance(text)
                self._save(text)
                self._last_saved_text = ""   # reset dedup for next utterance
            else:
                print_live(text)
                self._save(text)

            time.sleep(0.01)   # post-print sleep — CPU relief between updates

    def start(self):
        self.running = True
        decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        decode_thread.start()

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=self._audio_callback,
        )
        with stream:
            try:
                self._print_loop()
            except KeyboardInterrupt:
                self.running = False
                decode_thread.join(timeout=2)
                if self.save_path:
                    print(f"\n\n Transcript saved \u2192 {self.save_path}")
                print("\n \U0001f44b Stopped. Thank you for using IndicConformer ASR!\n")
            finally:
                self.running = False


# ====================== MAIN ======================

def main():
    parser = argparse.ArgumentParser(
        description="IndicConformer Real-Time Streaming ASR v7.2"
    )
    parser.add_argument(
        "--utterance", action="store_true",
        help="Use VAD utterance mode instead of continuous live mode"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save transcript to a timestamped .txt file"
    )
    args = parser.parse_args()

    # 1. Language selection
    lang_name, lang_id, model_id = select_language()

    # 2. Load model (auto GPU/CPU, no quantization)
    model, device = load_model(model_id)
    on_cpu = (str(device) == "cpu")

    # 3. Device-aware decoder selection
    decoder = select_decoder(str(device))

    # 4. Mic calibration with safe fallback
    print(" Calibrating mic... (stay quiet for 1 second)", end="", flush=True)
    try:
        noise = sd.rec(
            int(SAMPLE_RATE * 1.0), samplerate=SAMPLE_RATE,
            channels=1, dtype="float32"
        )
        sd.wait()
        global SILENCE_RMS_THRESH
        SILENCE_RMS_THRESH = max(0.008, min(0.05, rms(noise.flatten()) * 3.0))
        print(f" threshold = {SILENCE_RMS_THRESH:.4f}")
    except Exception:
        print(" using default")

    # 5. Optional transcript file — filename includes full language name
    save_path = None
    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"transcript_{lang_name.lower()}_{ts}.txt"
        print(f" Saving transcript \u2192 {save_path}")

    # 6. Pick partial interval based on device
    partial_sec = CPU_PARTIAL_INTERVAL_SEC if on_cpu else PARTIAL_INTERVAL_SEC

    # 7. Summary
    mode_label = "Continuous Live (Google-style)" if not args.utterance else "Utterance (VAD)"
    print(f"\n Language        : {lang_name} [{lang_id}]")
    print(f" Decoder         : {decoder.upper()}")
    print(f" Mode            : {mode_label}")
    print(f" Device          : {str(device).upper()}")
    print(f" Partial updates : every {partial_sec}s\n")

    # 8. Stream
    StreamingASR(model, lang_id, decoder,
                 continuous=not args.utterance,
                 save_path=save_path,
                 partial_interval=partial_sec).start()


if __name__ == "__main__":
    main()