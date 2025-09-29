
# coding=utf-8
import gradio as gr
import csv
import logging
import os
import json
from datetime import datetime
import random
import string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

playlist_file = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp_yoyo_sft_basebbc240_epoch17/playlist.csv"
save_result_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/output_exp_yoyo_sft_basebbc240_epoch17/mos_results"

def load_wavscp_with_text(file_path):
    wav_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "wav_path" in row and "text" in row:
                wav_data.append((row["wav_path"], row["text"]))
    return wav_data

wav_entries = load_wavscp_with_text(playlist_file)
total_num = len(wav_entries)

user_data_all = {}

def get_user_data(username):
    if username not in user_data_all:
        user_data_all[username] = {}
    return user_data_all[username]

def format_progress(idx):
    done = idx
    remain = total_num - done
    return f"Total: {total_num}, Completed: {done}, Remaining: {remain}"

def get_sample(idx, username):
    user_data = get_user_data(username)
    if 0 <= idx < total_num:
        wav_path, ref = wav_entries[idx]
        entry = user_data.get(idx, {})
        return (
            wav_path,
            ref,
            entry.get("unclear", ""),
            entry.get("missing", ""),
            entry.get("extra", ""),
            entry.get("user_text", ""),
            entry.get("accuracy", 3),
            entry.get("realness", 3),
            format_progress(idx)
        )
    else:
        return None, "", "", "", "", "", 3, 3, "All audio files have been reviewed."

def save_and_next(ref_text, unclear, missing, extra, user_text, acc, real, idx, username):
    user_data = get_user_data(username)
    user_data[idx] = {
        "ref_text": ref_text,
        "unclear": unclear,
        "missing": missing,
        "extra": extra,
        "user_text": user_text,
        "accuracy": float(acc),
        "realness": float(real)
    }
    logging.info(f"Next - {username} - wav_path: {wav_entries[idx][0]} - data: {user_data[idx]}")
    next_idx = idx + 1
    if next_idx >= total_num:
        return None, "", "", "", "", "", 3, 3, "All audio files have been reviewed.", next_idx
    return (*get_sample(next_idx, username), next_idx)

def save_and_prev(ref_text, unclear, missing, extra, user_text, acc, real, idx, username):
    user_data = get_user_data(username)
    user_data[idx] = {
        "ref_text": ref_text,
        "unclear": unclear,
        "missing": missing,
        "extra": extra,
        "user_text": user_text,
        "accuracy": float(acc),
        "realness": float(real)
    }
    logging.info(f"Prev - {username} - wav_path: {wav_entries[idx][0]} - data: {user_data[idx]}")
    prev_idx = idx - 1
    if prev_idx < 0:
        return None, "", "", "", "", "", 3, 3, "This is the first audio sample.", prev_idx
    return (*get_sample(prev_idx, username), prev_idx)

def export_csv(username):
    user_data = get_user_data(username)
    if not user_data:
        return "⚠️ No data to export for this user."
    csv_path = f"{save_result_dir}/res_{username}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "accuracy", "realness", "ref_text", "unclear_text", "missing_text", "extra_text", "user_text"])
        for idx, (wav_path, ref_text) in enumerate(wav_entries):
            entry = user_data.get(idx, {})
            writer.writerow([
                wav_path,
                entry.get("accuracy", ""),
                entry.get("realness", ""),
                ref_text,
                entry.get("unclear", ""),
                entry.get("missing", ""),
                entry.get("extra", ""),
                entry.get("user_text", "")
            ])
    jsonl_path = csv_path.replace(".csv", ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for idx, (wav_path, ref_text) in enumerate(wav_entries):
            entry = user_data.get(idx, {})
            line = {
                "wav_path": wav_path,
                "accuracy": entry.get("accuracy", ""),
                "realness": entry.get("realness", ""),
                "ref_text": ref_text,
                "unclear_text": entry.get("unclear", ""),
                "missing_text": entry.get("missing", ""),
                "extra_text": entry.get("extra", ""),
                "user_text": entry.get("user_text", "")
            }
            f_jsonl.write(json.dumps(line, ensure_ascii=False) + "\n")
    return f"✅ Exported to: {os.path.basename(csv_path)}"

with gr.Blocks() as demo:
    name_state = gr.State("")
    idx_state = gr.State(0)

    gr.Markdown("<h2 style='text-align: center; color: #4a90e2;'>Audio MOS Evaluation Tool</h2>")

    with gr.Row():
        name_input = gr.Textbox(label="Your Name (Required)", placeholder="Enter your name", interactive=True)
        btn_confirm = gr.Button("Confirm", variant="primary")

    with gr.Row():
        audio_player = gr.Audio(label="Audio", type="filepath", interactive=True)

    with gr.Column():
        ref_text = gr.Textbox(label="Ref Text", placeholder="Enter reference transcription...", lines=1, max_lines=1, interactive=False)
        unclear_text = gr.Textbox(label="Some words are not clearly or correctly pronounced.", placeholder="Enter unclear words...", lines=1, max_lines=1)
        missing_text = gr.Textbox(label="Some words from the reference text are missing.", placeholder="Enter missing words...", lines=1, max_lines=1)
        extra_text = gr.Textbox(label="Extra words not in the reference text are spoken.", placeholder="Enter extra words...", lines=1, max_lines=1)
        user_text = gr.Textbox(label="If audio doesn't match reference text, what do you hear?", placeholder="What do you actually hear...", lines=1, max_lines=1)

    with gr.Row():
        with gr.Column():
            mos_accuracy = gr.Slider(label="Pronunciation Accuracy", minimum=0, maximum=5, step=0.5, value=3)
            mos_realness = gr.Slider(label="Realness", minimum=0, maximum=5, step=0.5, value=3)
        with gr.Column():
            btn_next = gr.Button("Next", variant="primary", interactive=False)
            btn_prev = gr.Button("Previous", variant="primary", interactive=False)
            btn_export = gr.Button("📥 Export CSV", variant="primary", interactive=False)

    export_status = gr.Textbox(label="Export Status", interactive=False)
    progress_text = gr.Text(label="Progress", interactive=False)
    

    def generate_suffix(length=6):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def confirm_name(name):
        if not name.strip():
            return (
                gr.update(interactive=True),
                "Please enter your name.",
                "", None,
                None, "", "", "", "", "", 3, 3, "Please enter your name.", 0,
                gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
            )
        suffix = generate_suffix()
        full_name = f"{name}_{suffix}"
        
        idx0 = 0
        return (
            gr.update(interactive=False), "", full_name, 
            *get_sample(idx0, name), idx0,
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
        )

    btn_confirm.click(fn=confirm_name, inputs=[name_input], outputs=[
        name_input, export_status, name_state, 
        audio_player, ref_text, unclear_text, missing_text, extra_text, user_text,
        mos_accuracy, mos_realness, progress_text, idx_state,
        btn_next, btn_prev, btn_export
    ])

    btn_prev.click(save_and_prev, inputs=[ref_text, unclear_text, missing_text, extra_text, user_text, mos_accuracy, mos_realness, idx_state, name_state],
                   outputs=[audio_player, ref_text, unclear_text, missing_text, extra_text, user_text, mos_accuracy, mos_realness, progress_text, idx_state])

    btn_next.click(save_and_next, inputs=[ref_text, unclear_text, missing_text, extra_text, user_text, mos_accuracy, mos_realness, idx_state, name_state],
                   outputs=[audio_player, ref_text, unclear_text, missing_text, extra_text, user_text, mos_accuracy, mos_realness, progress_text, idx_state])

    btn_export.click(export_csv, inputs=[name_state], outputs=[export_status])

demo.launch(server_port=7680, server_name="127.0.0.1", share=False)
