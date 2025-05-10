import whisper
import subprocess
import torch
from deep_translator import GoogleTranslator
import os
import re

# Caminho do vídeo (absoluto)
video_path = os.path.abspath("video.mp4")
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")

# Testa se a GPU está disponível
if torch.cuda.is_available():
    print(f"CUDA está disponível! Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA não está disponível, usando CPU.")

# Dispositivo apropriado
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega modelo Whisper
model = whisper.load_model("tiny").to(device)

# Função para obter duração total do vídeo
def get_video_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    try:
        return float(result.stdout.decode().strip())
    except ValueError:
        raise RuntimeError(f"Erro ao obter duração do vídeo: {result.stdout.decode().strip()}")

total_duration = get_video_duration(video_path)

# Função para formatar timestamp no estilo SRT
def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# Transcrevendo o vídeo
print("[Iniciando transcrição...]")
result = model.transcribe(video_path, task="transcribe", verbose=False)

# Salva arquivo de legenda original (saida.srt)
with open("saida.srt", "w", encoding="utf-8") as f:
    for i, segment in enumerate(result["segments"], start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()

        f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

        percent = (segment["end"] / total_duration) * 100
        print(f"📝 Transcrevendo: {percent:.1f}%\r", end="")

print("\n✅ Transcrição concluída! Arquivo 'saida.srt' gerado.")

# Traduzindo as legendas
print("🌍 Iniciando tradução...")
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

print("\n✅ Tradução concluída! Arquivo 'saida_traduzida.srt' gerado.")

# Inserindo legenda traduzida no vídeo (hardcoded com ffmpeg)
output_video = "video_com_legenda.mp4"

def monitor_ffmpeg_progress(pipe, total_duration):
    pattern = re.compile(r'time=(\d+):(\d+):(\d+)\.(\d+)')
    for line in pipe:
        try:
            match = pattern.search(line)
            if match:
                h, m, s, ms = map(int, match.groups())
                current_time = h * 3600 + m * 60 + s + ms / 100
                percent = (current_time / total_duration) * 100
                print(f"🎞️ Inserindo legenda: {percent:.1f}%\r", end="")
        except Exception:
            pass  # Ignora erros ao processar linha

print("⏳ Inserindo legenda no vídeo (isso pode demorar)...")

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
