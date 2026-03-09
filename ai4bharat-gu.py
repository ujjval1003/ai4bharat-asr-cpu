import torch
import nemo.collections.asr as nemo_asr

print("Loading AI4Bharat Gujarati model on CPU...")

model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large",
    map_location="cpu"
)

model.freeze()
model = model.to("cpu")

print("✅ Model loaded successfully!\n")

# CTC Decoder
model.cur_decoder = "ctc"
ctc_text = model.transcribe(['sample_audio_infer_ready.wav'], batch_size=1, language_id='gu')[0]
print("CTC Decoder  :", ctc_text)

# RNN-T Decoder
model.cur_decoder = "rnnt"
rnnt_text = model.transcribe(['sample_audio_infer_ready.wav'], batch_size=1, language_id='gu')[0]
print("RNN-T Decoder:", rnnt_text)