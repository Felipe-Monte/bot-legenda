import whisper
import subprocess
import torch
import os
import re
from deep_translator import GoogleTranslator
import gradio as gr

# Define se vai usar GPU ou CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega modelo Whisper
model = whisper.load_model("tiny").to(device)

# Formata timestamps em SRT
def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# Pega a duração total do vídeo
def get_video_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout.decode().strip())

# Monitora progresso do ffmpeg
def monitor_ffmpeg_progress(pipe, total_duration):
    pattern = re.compile(r'time=(\d+):(\d+):(\d+)\.(\d+)')
    for line in pipe:
        match = pattern.search(line)
        if match:
            h, m, s, ms = map(int, match.groups())
            current_time = h * 3600 + m * 60 + s + ms / 100
            percent = (current_time / total_duration) * 100
            print(f"🎞️ Inserindo legenda: {percent:.1f}%\r", end="")

# Função principal do app
def processar_video(video_path):
    try:
        if not video_path:
            return "⚠️ Nenhum vídeo selecionado."

        print("🔧 Iniciando processamento do vídeo...")
        total_duration = get_video_duration(video_path)

        # Transcrição
        print("[1/3] 🎤 Transcrevendo áudio...")
        result = model.transcribe(video_path, task="transcribe", verbose=False)

        with open("saida.srt", "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], start=1):
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
                percent = (segment["end"] / total_duration) * 100
                print(f"📝 Transcrevendo: {percent:.1f}%\r", end="")

        # Tradução
        print("\n[2/3] 🌍 Traduzindo legendas...")
        translator = GoogleTranslator(source='auto', target='pt')
        with open("saida_traduzida.srt", "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], start=1):
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                original_text = segment["text"].strip()
                translated_text = translator.translate(original_text)
                f.write(f"{i}\n{start} --> {end}\n{translated_text}\n\n")
                percent = (segment["end"] / total_duration) * 100
                print(f"🌍 Traduzindo: {percent:.1f}%\r", end="")

        # Inserção das legendas
        print("\n[3/3] 🎞️ Inserindo legendas no vídeo...")
        output_video = "video_com_legenda.mp4"

        process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"subtitles=saida_traduzida.srt",
                "-c:a", "copy", output_video
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            universal_newlines=True
        )

        monitor_ffmpeg_progress(process.stderr, total_duration)
        process.wait()

        print(f"\n✅ Legenda inserida com sucesso! Arquivo final: {output_video}")
        return output_video

    except Exception as e:
        print(f"❌ Erro: {e}")
        return f"❌ Erro: {e}"

# Interface Gradio
gr.Interface(
    fn=processar_video,
    inputs=gr.File(label="Selecione o vídeo"),
    outputs=gr.Video(label="Vídeo com legenda"),
    title="Bot de Legendas com Tradução",
    description="Este app transcreve, traduz e insere legendas hardcoded no vídeo."
).launch()
