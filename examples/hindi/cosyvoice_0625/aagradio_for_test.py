# coding=utf-8
import gradio as gr
import csv
import logging
import os
import json
from datetime import datetime

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------------------
# Configuration and Global State
# -------------------------------

wav_file = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp_yoyo_sft_basebbc240_epoch17/wav.scp"
csv_prefix = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp_yoyo_sft_basebbc240_epoch17/mos_results"

def load_wavscp(file_path):
    wav_paths = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            wav_path = line.strip()
            if wav_path:
                wav_paths.append(wav_path)
    return wav_paths

wav_paths = load_wavscp(wav_file)
total_num = len(wav_paths)
user_data = {}

# -------------------------------
# Helper Functions
# -------------------------------

def format_progress(idx):
    done = idx
    remain = total_num - done
    return f"Total: {total_num}, Completed: {done}, Remaining: {remain}"

def get_sample(idx):
    if 0 <= idx < total_num:
        wav_path = wav_paths[idx]
        prev_text = user_data.get(idx, {}).get("text", "")
        prev_acc = user_data.get(idx, {}).get("accuracy", 3)
        prev_real = user_data.get(idx, {}).get("realness", 3)
        progress = format_progress(idx)
        return wav_path, prev_text, prev_acc, prev_real, progress
    else:
        return None, "", 3, 3, "All audio files have been reviewed."

def save_and_next(text, acc, real, idx):
    user_data[idx] = {"text": text, "accuracy": float(acc), "realness": float(real)}
    logging.info("Next")
    logging.info(f"wav_path: {wav_paths[idx]}")
    logging.info(f"user_data[{idx}] = {user_data[idx]}")
    
    next_idx = idx + 1
    if next_idx >= total_num:
        return None, "", 3, 3, "All audio files have been reviewed.", next_idx
    return (*get_sample(next_idx), next_idx)

def save_and_prev(text, acc, real, idx):
    user_data[idx] = {"text": text, "accuracy": float(acc), "realness": float(real)}
    logging.info("Prev")
    logging.info(f"wav_path: {wav_paths[idx]}")
    logging.info(f"user_data[{idx}] = {user_data[idx]}")
    prev_idx = idx - 1
    if prev_idx < 0:
        return None, "", 3, 3, "This is the first audio sample.", prev_idx
    return (*get_sample(prev_idx), prev_idx)

def export_csv(start_str):
    end_time = datetime.now()
    end_str = end_time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"{csv_prefix}_{start_str}_to_{end_str}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "text", "accuracy", "realness"])
        for idx, wav_path in enumerate(wav_paths):
            entry = user_data.get(idx, {})
            writer.writerow([
                wav_path,
                entry.get("text", ""),
                entry.get("accuracy", ""),
                entry.get("realness", "")
            ])
    csv_name = csv_path.split("/")[-1]
    
    # 写成json 文件
    jsonl_path = csv_path.replace(".csv", ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for idx, wav_path in enumerate(wav_paths):
            entry = user_data.get(idx, {})
            line = {
                "wav_path": wav_path,
                "text": entry.get("text", ""),
                "accuracy": entry.get("accuracy", ""),
                "realness": entry.get("realness", "")
            }
            f_jsonl.write(json.dumps(line, ensure_ascii=False) + "\n")
    return f"✅ Exported to: {csv_name}"

# -------------------------------
# Gradio UI Layout
# -------------------------------

with gr.Blocks() as demo:
    start_time = datetime.now()
    start_str = start_time.strftime("%Y%m%d_%H%M%S")
    start_state = gr.State(start_str)  # ✅ 把它放进 gr.State
    
    gr.Markdown("<h2 style='text-align: center; color: #4a90e2;'>Audio MOS Evaluation Tool</h2>")
    idx_state = gr.State(0)

    # 第一行：Audio 播放 + 上/下一条
    with gr.Row():
        with gr.Column():
            audio_player = gr.Audio(label="Audio", type="filepath", interactive=True, elem_classes=["audio-container"])
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Transcription or Notes", placeholder="Enter transcription or notes...", lines=1, max_lines=1)
            
    with gr.Row():
        with gr.Column():
            mos_accuracy = gr.Slider(label="Pronunciation Accuracy", minimum=0, maximum=5, step=0.5, value=3)
            mos_realness = gr.Slider(label="Realness", minimum=0, maximum=5, step=0.5, value=3)
        with gr.Column():    
            btn_next = gr.Button("Next", variant="primary")
            btn_prev = gr.Button("Previous", variant="primary")
            btn_export = gr.Button("📥 Export CSV", variant="primary")
            
    with gr.Row():
        with gr.Column():
            export_status = gr.Textbox(label="Export Status", interactive=False)

    # 最下方：进度
    with gr.Row():
        progress_text = gr.Text(label="Progress", interactive=False)

    # Load initial sample
    def load_first(idx):
        return (*get_sample(idx), idx)

    demo.load(load_first, inputs=idx_state, outputs=[audio_player, text_input, mos_accuracy, mos_realness, progress_text, idx_state])
    btn_prev.click(save_and_prev, inputs=[text_input, mos_accuracy, mos_realness, idx_state],
                   outputs=[audio_player, text_input, mos_accuracy, mos_realness, progress_text, idx_state])
    btn_next.click(save_and_next, inputs=[text_input, mos_accuracy, mos_realness, idx_state],
                   outputs=[audio_player, text_input, mos_accuracy, mos_realness, progress_text, idx_state])
    btn_export.click(export_csv, inputs=[start_state], outputs=export_status)

# Launch the app
demo.launch(
    server_port=7680,
    server_name="127.0.0.1",
    share=False
)
