from transformers import AutoModel
import torch, torchaudio

# Load the model
model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual", 
    trust_remote_code=True,
    map_location="cpu"
)

model = model.to("cpu")

# Load an audio file
wav, sr = torchaudio.load("sample_audio_infer_ready.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

target_sample_rate = 16000  # Expected sample rate
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
    wav = resampler(wav)

# Perform ASR with CTC decoding
with torch.no_grad():
    transcription_ctc = model(wav, "gu", "ctc")
    print("CTC Transcription:", transcription_ctc)

# Perform ASR with RNNT decoding
with torch.no_grad():
    transcription_rnnt = model(wav, "gu", "rnnt")
    print("RNNT Transcription:", transcription_rnnt)