import whisper
import subprocess
import torch
from deep_translator import GoogleTranslator
import os

# Testa se a GPU está disponível
if torch.cuda.is_available():
    print(f"CUDA está disponível! Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA não está disponível, usando CPU.")

# Caminho do vídeo
video_path = "video.mp4"

# Dispositivo adequado
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega o modelo Whisper
model = whisper.load_model("small").to(device)

# Duração total do vídeo para barra de progresso
def get_video_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout.decode().strip())

total_duration = get_video_duration(video_path)

# Transcreve o vídeo
result = model.transcribe(video_path, task="transcribe", verbose=False)

# Função para formatar timestamp SRT
def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# Gera legenda original
with open("saida.srt", "w", encoding="utf-8") as f:
    for i, segment in enumerate(result["segments"], start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()

        f.write(f"{i}\n")
        f.write(f"{start} --> {end}\n")
        f.write(f"{text}\n\n")

        percent = (segment["end"] / total_duration) * 100
        print(f"📝 Transcrevendo: {percent:.1f}%\r", end="")

print("\n✅ Transcrição concluída! Arquivo 'saida.srt' gerado.")

# Traduzindo a legenda
translator = GoogleTranslator(source='auto', target='pt')

with open("saida_traduzida.srt", "w", encoding="utf-8") as f:
    for i, segment in enumerate(result["segments"], start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        original_text = segment["text"].strip()
        translated_text = translator.translate(original_text)

        f.write(f"{i}\n")
        f.write(f"{start} --> {end}\n")
        f.write(f"{translated_text}\n\n")

        percent = (segment["end"] / total_duration) * 100
        print(f"🌍 Traduzindo: {percent:.1f}%\r", end="")

print("\n✅ Tradução concluída! Arquivo 'saida_traduzida.srt' gerado.")

# Inserindo legenda traduzida no vídeo (hardcoded)
output_video = "video_com_legenda.mp4"

subprocess.run([
    "ffmpeg", "-y", "-i", video_path, "-vf", f"subtitles=saida_traduzida.srt",
    "-c:a", "copy", output_video
])

print(f"\n🎬 Legenda inserida no vídeo com sucesso! Arquivo final: {output_video}")
