import whisper
import subprocess
import torch
import re
import shutil
from deep_translator import GoogleTranslator
import gradio as gr
import uuid
import os

# Verifica se ffmpeg está instalado
if not shutil.which("ffmpeg"):
    raise EnvironmentError("❌ ffmpeg não está instalado no sistema.")

# Testa se a GPU está disponível
if torch.cuda.is_available():
    print(f"✅ CUDA está disponível! Usando GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️ CUDA não está disponível, usando CPU.")
    device = "cpu"

# Carrega modelo Whisper
model = whisper.load_model("tiny").to(device)

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def get_video_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout.decode().strip())

def run_ffmpeg_with_progress(input_path, subtitle_path, output_path, total_duration):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", f"subtitles={subtitle_path}",
        "-c:a", "copy",
        output_path,
    ]

    process = subprocess.Popen(command, stderr=subprocess.PIPE, universal_newlines=True)

    pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

    while True:
        line = process.stderr.readline()
        if not line:
            break
        match = pattern.search(line)
        if match:
            h, m, s = map(float, match.groups())
            current = h * 3600 + m * 60 + s
            progress = 80 + min(20, (current / total_duration) * 20)
            yield progress, f"[3/3] 🎮 Inserindo legendas no vídeo... {progress:.1f}%", None

    process.wait()
    yield 100, "✅ Legenda inserida com sucesso!", output_path

def processar_video(video_path):
    try:
        if not video_path:
            yield 0, "⚠️ Nenhum vídeo selecionado.", None
            return

        yield 0, "🔧 Iniciando processamento do vídeo...", None
        total_duration = get_video_duration(video_path)

        # Transcrição com barra de progresso
        segments = []
        result = model.transcribe(video_path, task="transcribe", verbose=False)
        segments = result["segments"]
        total_segments = len(segments)

        for i, segment in enumerate(segments, start=1):
            progress = (i / total_segments) * 40  # 0% a 40%
            yield progress, f"[1/3] 🎤 Transcrevendo áudio... {progress:.1f}%", None

        # Tradução
        legenda_traduzida = f"saida_traduzida_{uuid.uuid4().hex[:8]}.srt"
        translator = GoogleTranslator(source='auto', target='pt')

        with open(legenda_traduzida, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                original_text = segment["text"].strip()
                try:
                    translated_text = translator.translate(original_text)
                except Exception as e:
                    translated_text = "[Erro na tradução]"
                    print(f"⚠️ Erro ao traduzir: {e}")
                f.write(f"{i}\n{start} --> {end}\n{translated_text}\n\n")

                progress = 40 + ((i / total_segments) * 40)  # 40% a 80%
                yield progress, f"[2/3] 🌐 Traduzindo legendas... {progress:.1f}%", None

        # Inserção das legendas no vídeo
        output_video = f"video_com_legenda_{uuid.uuid4().hex[:8]}.mp4"
        for p, status, result in run_ffmpeg_with_progress(video_path, legenda_traduzida, output_video, total_duration):
            yield p, status, result

    except Exception as e:
        yield 0, f"❌ Erro: {e}", None

iface = gr.Interface(
    fn=processar_video,
    inputs=gr.File(label="Selecione o vídeo"),
    outputs=[
        gr.Slider(minimum=0, maximum=100, label="Progresso", interactive=False),
        gr.Textbox(label="Status"),
        gr.Video(label="Vídeo com legenda"),
    ],
    title="Bot de Legendas com Tradução",
    description="Este app transcreve, traduz e insere legendas hardcoded no vídeo.",
)

iface.launch(share=True)
