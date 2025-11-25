import os
import math
import ffmpeg
import pysrt
import sys
import types

# ---- Stub pyaudioop so pydub can import it without error ----
# Newer pydub versions do: "import pyaudioop as audioop".
# We don't need its functionality (playback/analysis), only file I/O,
# so a dummy module is enough.
if "pyaudioop" not in sys.modules:
    sys.modules["pyaudioop"] = types.ModuleType("pyaudioop")

from flask import Flask, render_template, request, send_file
from faster_whisper import WhisperModel
from translate import Translator
from gtts import gTTS
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------- Helper ----------

def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


# ---------- Routes ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dub", methods=["POST"])
def dub_video():
    if "video" not in request.files:
        return "No video uploaded"

    video = request.files["video"]
    target_lang = request.form.get("language", "ta")

    # Save uploaded video
    input_video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(input_video_path)

    base_name = os.path.splitext(video.filename)[0]

    extracted_audio = os.path.join(OUTPUT_FOLDER, f"{base_name}.wav")
    eng_srt = os.path.join(OUTPUT_FOLDER, f"{base_name}_en.srt")
    translated_srt = os.path.join(OUTPUT_FOLDER, f"{base_name}_{target_lang}.srt")
    final_audio = os.path.join(OUTPUT_FOLDER, f"{base_name}_{target_lang}.wav")
    final_video = os.path.join(OUTPUT_FOLDER, f"{base_name}_{target_lang}.mp4")

    # 1. Extract audio from video
    stream = ffmpeg.input(input_video_path)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)

    # 2. Transcribe using Whisper
    model = WhisperModel("small")
    segments, info = model.transcribe(extracted_audio)
    segments = list(segments)
    source_lang = info.language  # e.g. "en"

    # 3. Create original subtitle file
    text = ""
    for i, seg in enumerate(segments):
        text += f"{i + 1}\n"
        text += f"{format_time(seg.start)} --> {format_time(seg.end)}\n"
        text += seg.text + "\n\n"

    with open(eng_srt, "w", encoding="utf-8") as f:
        f.write(text)

    # 4. Translate subtitles
    subs = pysrt.open(eng_srt, encoding="utf-8")
    translator = Translator(from_lang=source_lang, to_lang=target_lang)

    for sub in subs:
        sub.text = translator.translate(sub.text)

    subs.save(translated_srt, encoding="utf-8")

    # 5. Generate dubbed audio from translated subtitles
    combined = AudioSegment.silent(duration=0)

    for sub in subs:
        start_time = sub.start.ordinal / 1000.0  # seconds
        tts = gTTS(sub.text, lang=target_lang)
        tts.save("temp.mp3")

        seg_audio = AudioSegment.from_mp3("temp.mp3")

        silent_duration = start_time * 1000 - len(combined)
        if silent_duration > 0:
            combined += AudioSegment.silent(duration=silent_duration)

        combined += seg_audio

    combined.export(final_audio, format="wav")
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")

    # 6. Replace the video audio track
    video_clip = VideoFileClip(input_video_path)
    audio_clip = AudioFileClip(final_audio)

    final = video_clip.set_audio(audio_clip)
    final.write_videofile(final_video, codec="libx264", audio_codec="aac")

    return send_file(final_video, as_attachment=True)


if __name__ == "__main__":
    # Local debug only – Render uses gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
