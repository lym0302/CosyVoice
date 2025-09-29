# coding=utf-8
import gradio as gr
import csv
import logging
import os
import json
import random
import string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

#root_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/test300_v2/"
# root_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/output_test1min/to_eval"
root_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/filelists_v3/yoyo_v2_choose2100"
playlist_file = os.path.join(root_dir, "playlist.csv")
save_result_dir = os.path.join(root_dir, "save_res")
os.makedirs(save_result_dir, exist_ok=True)
spk_save_dir = {}

user_data_all = {}

def get_user_data(username):
    if username not in user_data_all:
        user_data_all[username] = {}
    return user_data_all[username]


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



def get_evaluated_audio_prefixes(save_res_spk_dir):
    """
    读取用户保存目录下的所有json文件，提取音频文件的前缀（文件名去掉扩展名）
    """
    evaluated_prefixes = set()
    if not os.path.exists(save_res_spk_dir):
        return evaluated_prefixes
    
    for filename in os.listdir(save_res_spk_dir):
        if filename.endswith(".json"):
            prefix = os.path.splitext(filename)[0]  # 去掉 .json 后缀
            evaluated_prefixes.add(prefix)
    return evaluated_prefixes


def format_progress(idx):
    done = idx
    remain = total_num - done
    return f"Total: {total_num}, Completed: {done}, Remaining: {remain}"


def get_sample(idx, username):
    user_data = get_user_data(username)
    if 0 <= idx < total_num:
        wav_path, ref_text = wav_entries[idx]
        entry = user_data.get(idx, {})
        return (
            wav_path,
            # ref_text,
            entry.get("sentence_type", None),
            # entry.get("has_paralinguistic", "No"),
            entry.get("paralinguistic_type", None),
            format_progress(idx)
        )
    else:
        return None, None, [], "All audio files have been evaluated."


def save_and_next(sentence_type, paralinguistic_type, idx, username):
    if idx >= total_num:
        return None, None, [], "All audio files have been evaluated.", idx
        
    if username not in spk_save_dir:
        save_res_spk_dir = os.path.join(save_result_dir, username)
        os.makedirs(save_res_spk_dir, exist_ok=True)
        spk_save_dir[username] = save_res_spk_dir

    user_data = get_user_data(username)
    wav_path, ref_text = wav_entries[idx]
    wav_name = wav_path.split("/")[-1].replace(".wav", "")
    user_data[idx] = {
        "wav_path": wav_path,
        "ref_text": ref_text,
        "sentence_type": sentence_type,
        # "has_paralinguistic": has_paralinguistic,
        "paralinguistic_type": paralinguistic_type,
    }
    logging.info(f"Next - {username} - data: {user_data[idx]}")
    save_json_file = os.path.join(spk_save_dir[username], wav_name+".json")
    with open(save_json_file, "w", encoding="utf-8") as f:
        json.dump(user_data[idx], f, ensure_ascii=False, indent=2)
        
    next_idx = idx + 1
    if next_idx >= total_num:
        return None, None, [], "All audio files have been evaluated.", next_idx
    return (*get_sample(next_idx, username), next_idx)


def save_and_prev(sentence_type, paralinguistic_type, idx, username):
    if idx >= total_num:
        return None, None, [], "All audio files have been evaluated.", idx
    
    if username not in spk_save_dir:
        save_res_spk_dir = os.path.join(save_result_dir, username)
        os.makedirs(save_res_spk_dir, exist_ok=True)
        spk_save_dir[username] = save_res_spk_dir

    user_data = get_user_data(username)
    wav_path, ref_text = wav_entries[idx]
    wav_name = wav_path.split("/")[-1].replace(".wav", "")
    user_data[idx] = {
        "wav_path": wav_path,
        "ref_text": ref_text,
        "sentence_type": sentence_type,
        # "has_paralinguistic": has_paralinguistic,
        "paralinguistic_type": paralinguistic_type,
    }
    logging.info(f"Prev - {username} - data: {user_data[idx]}")
    save_json_file = os.path.join(spk_save_dir[username], wav_name+".json")
    with open(save_json_file, "w", encoding="utf-8") as f:
        json.dump(user_data[idx], f, ensure_ascii=False, indent=2)
        
    prev_idx = idx - 1
    if prev_idx < 0:
        return None, None, [], "This is the first audio sample.", prev_idx
    return (*get_sample(prev_idx, username), prev_idx)


def get_start_idx(name):
    start_idx = 0
    save_res_spk_dir = os.path.join(save_result_dir, name)
    os.makedirs(save_res_spk_dir, exist_ok=True)
    spk_save_dir[name] = save_res_spk_dir
    evaluated_prefixes = get_evaluated_audio_prefixes(save_res_spk_dir)
    start_idx = len(evaluated_prefixes)
    return start_idx



def confirm_name(name):
    if not name.strip():
        return (
            gr.update(interactive=True), gr.update(value=""),   # name_state 保持为空
            None, None, [], "Please enter your name.", 0,
            gr.update(interactive=False), gr.update(interactive=False),
        )
    idx0 = get_start_idx(name)
    return (
        gr.update(interactive=False), gr.update(value=name),   # 关键：更新 name_state
        *get_sample(idx0, name), idx0,
        gr.update(interactive=True), gr.update(interactive=True)
    )


with gr.Blocks() as demo:
    name_state = gr.State("")
    idx_state = gr.State(0)

    gr.Markdown("<h2 style='text-align: center; color: #4a90e2;'>Audio MOS Evaluation Tool</h2>")

    with gr.Row():
        name_input = gr.Textbox(label="Your Name (Required)", placeholder="Enter your name", interactive=True)
        btn_confirm = gr.Button("Confirm", variant="primary")

    with gr.Row():
        audio_player = gr.Audio(label="Audio", type="filepath", interactive=True)

    with gr.Row():
        with gr.Column():
            sentence_type = gr.Radio(
                choices=["Declarative", "Interrogative", "Exclamatory", "Other"],
                label="Choose one per clip based only on audio; if unsure, pick Other — this decides the sentence-ending punctuation.",
                value=None  # 默认值
            )
            # has_paralinguistic = gr.Radio(
            #     choices=["Yes", "No"],
            #     label="Select Yes if audio contains paralinguistic sounds (e.g., breath, laughter, noise, cough); otherwise select No.",
            #     value="No"  # 默认选否
            # )
            paralinguistic_type = gr.CheckboxGroup(
                choices=["Laught", "Cough", "Thinking sound (emm...)", "Breath", "Strong emphasis", "Other"],
                label="If audio contains paralinguistic sounds, select all that apply (multiple choices allowed)",
                value=[]  # 默认不选
            )
    
    with gr.Row():
        with gr.Column():
            btn_next = gr.Button("Next", variant="primary", interactive=False)
            btn_prev = gr.Button("Previous", variant="primary", interactive=False)
            

    # export_status = gr.Textbox(label="Export Status", interactive=False)
    progress_text = gr.Text(label="Progress", interactive=False)
    

    btn_confirm.click(
        fn=confirm_name, 
        inputs=[name_input], 
        outputs=[name_input, name_state, 
        audio_player, sentence_type, paralinguistic_type, progress_text, idx_state,
        btn_next, btn_prev
    ])

    btn_prev.click(save_and_prev, inputs=[sentence_type, paralinguistic_type, idx_state, name_input],
                   outputs=[audio_player, sentence_type, paralinguistic_type, progress_text, idx_state])

    btn_next.click(save_and_next, inputs=[sentence_type, paralinguistic_type, idx_state, name_input],
                   outputs=[audio_player, sentence_type, paralinguistic_type, progress_text, idx_state])


demo.launch(server_port=7680, server_name="127.0.0.1", share=False,
            allowed_paths=["/data/liangyunming/dataset/hindi/data_v2_sft/audio_loudnorm_-16_choose_mos3_snr30",
                           "/data/liangyunming/dataset/hindi/data_v2/audio_loudnorm_-16"])
