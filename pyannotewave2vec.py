import os
import torch
import torchaudio
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, BitsAndBytesConfig

mp3_path = "/content/C02_1009.mp3"
wav_path = "/content/C02_1009_16k_mono.wav"
slice_duration_sec = 100
output_dir = "/content/slices_output"
os.makedirs(output_dir, exist_ok=True)

# Convert MP3 to WAV
audio = AudioSegment.from_file(mp3_path)
resampled_audio = audio.set_frame_rate(16000).set_channels(1)
resampled_audio.export(wav_path, format="wav")
print(f"Resampled audio saved at: {wav_path}")

# Load WAV
waveform, sr = torchaudio.load(wav_path)
total_duration_sec = waveform.shape[1] / sr
num_slices = int(total_duration_sec // slice_duration_sec) + 1

slice_indices = [
    (int(i * slice_duration_sec * sr), int(min((i + 1) * slice_duration_sec * sr, waveform.shape[1])))
    for i in range(num_slices)
]

# Load Pyannote
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=""
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarization_pipeline.to(device)

# Load Wav2Vec2
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-960h-lv60-self",
    device_map="auto",
    quantization_config=bnb_config
).eval()

# Storage
full_words = []   # each word: (global_start, global_end, word)
full_diarization = []

for idx, (start_sample, end_sample) in enumerate(slice_indices):
    slice_waveform = waveform[:, start_sample:end_sample]
    slice_path = os.path.join(output_dir, f"slice_{idx+1}.wav")
    torchaudio.save(slice_path, slice_waveform, sr)

    # ---- Pyannote diarization ----
    with ProgressHook() as hook:
        diarization_output = diarization_pipeline(slice_path, hook=hook)
    
    for turn, speaker in diarization_output.speaker_diarization:
        global_start = turn.start + start_sample / sr
        global_end = turn.end + start_sample / sr
        full_diarization.append((global_start, global_end, speaker))

    # ---- Wav2Vec2 transcription ----
    input_values = processor(slice_waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt").input_values.to(device)
    input_values = input_values.to(dtype=wav2vec_model.dtype)
    with torch.no_grad():
        logits = wav2vec_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(predicted_ids)[0]

    # ---- Approximate word-level timestamps ----
    words = transcript.split()
    slice_duration = (end_sample - start_sample) / sr
    if words:
        word_duration = slice_duration / len(words)
        for i, word in enumerate(words):
            word_start = start_sample / sr + i * word_duration
            word_end = word_start + word_duration
            full_words.append((word_start, word_end, word))

# ---- Merge words with speakers ----
word_speaker_mapping = []
for w_start, w_end, word in full_words:
    # find the speaker whose segment contains the word midpoint
    mid = (w_start + w_end) / 2
    speaker_label = "UNK"
    for s_start, s_end, speaker in full_diarization:
        if s_start <= mid <= s_end:
            speaker_label = speaker
            break
    word_speaker_mapping.append((w_start, w_end, speaker_label, word))

# ---- Save final aligned transcript ----
final_transcript_path = os.path.join(output_dir, "aligned_transcription.txt")
with open(final_transcript_path, "w") as f:
    for w_start, w_end, speaker, word in word_speaker_mapping:
        f.write(f"{w_start:.2f}-{w_end:.2f}s {speaker}: {word}\n")

print(f"Aligned transcription with speakers saved at: {final_transcript_path}")