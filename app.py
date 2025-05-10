import whisper
import subprocess
import torch
import re
from deep_translator import GoogleTranslator
import gradio as gr
import uuid

# Testa se a GPU está disponível
if torch.cuda.is_available():
    print(f"✅ CUDA está disponível! Usando GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️ CUDA não está disponível, usando CPU.")
    device = "cpu"

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

# Função principal do app
def processar_video(video_path):
    try:
        if not video_path:
            return "⚠️ Nenhum vídeo selecionado."

        print("🔧 Iniciando processamento do vídeo...")
        total_duration = get_video_duration(video_path)

        # Transcrição
        print("[1/2] 🎤 Transcrevendo e traduzindo áudio...")
        result = model.transcribe(video_path, task="transcribe", verbose=False)

        legenda_traduzida = f"saida_traduzida_{uuid.uuid4().hex[:8]}.srt"
        translator = GoogleTranslator(source='auto', target='pt')

        with open(legenda_traduzida, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], start=1):
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                original_text = segment["text"].strip()
                try:
                    translated_text = translator.translate(original_text)
                except Exception as e:
                    translated_text = "[Erro na tradução]"
                    print(f"⚠️ Erro ao traduzir: {e}")
                f.write(f"{i}\n{start} --> {end}\n{translated_text}\n\n")
                percent = (segment["end"] / total_duration) * 100
                print(f"📝 Progresso: {percent:.1f}%\r", end="")

        # Inserção das legendas
        print("\n[2/2] 🎞️ Inserindo legendas no vídeo...")
        output_video = f"video_com_legenda_{uuid.uuid4().hex[:8]}.mp4"

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"subtitles={legenda_traduzida}",
                "-c:a", "copy", output_video
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print(f"\n✅ Legenda inserida com sucesso! Arquivo final: {output_video}")
        return output_video

    except Exception as e:
        print(f"❌ Erro: {e}")
        return f"❌ Erro: {e}"

# Interface Gradio com compartilhamento público
gr.Interface(
    fn=processar_video,
    inputs=gr.File(label="Selecione o vídeo"),
    outputs=gr.Video(label="Vídeo com legenda"),
    title="Bot de Legendas com Tradução",
    description="Este app transcreve, traduz e insere legendas hardcoded no vídeo."
).launch(share=True)
