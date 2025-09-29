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
root_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/paralang/to_eval"
playlist_file = os.path.join(root_dir, "playlist.csv")
save_result_dir = os.path.join(root_dir, "save_res")
os.makedirs(save_result_dir, exist_ok=True)
spk_save_dir = {}

user_data_all = {}

user_ranges = {}  # name -> (start_id, end_id)


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

examples = [
    {
        "example_type": "example for laughter",
        "audio": "/data/liuchao/annotation/audio/17003974_60ca96128c591300011365db_1753027612.wav",
        "ref_text": "<laughter>गले पे फिट करा दे गले पे।</laughter>",
        "correct_text": "[laughter]<laughter>गले पे फिट करा दे गले पे।</laughter>"
    },
    {
        "example_type": "example for breath",
        "audio": "/data/liuchao/annotation/audio/36657274_5f7560863b0f850001cebe77_1751944963.wav",
        "ref_text": "चलो अच्छी बात है। [breath]",
        "correct_text": "चलो अच्छी बात है। [breath]"
    },
    {
        "example_type": "example for cough",
        "audio": "/data/liuchao/annotation/audio/32018318_62cfbef4a6b9c30001545c11_1752575493.wav",
        "ref_text": "[cough] गर्मी तो है। [noise]",
        "correct_text": "[cough] गर्मी तो है।"
    },
    {
        "example_type": "example for strong",
        "audio": "/data/liuchao/annotation/audio/29629591_5fa21209d2e8a8000135d234_1752662171.wav",
        "ref_text": "मतलब <strong>क्या</strong> है तेरा?",
        "correct_text": "मतलब <strong>क्या</strong> है तेरा?[laughter]"
    }
    
]

# 定义显示函数（这里只是直接返回示例数据）
def show_example(example):
    return example["audio"], example["ref_text"], example["correct_text"]
    


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


def format_progress(idx, start_id, end_id):
    total = end_id - start_id
    done = idx - start_id
    remain = total - done
    return f"Total: {total}, Completed: {done}, Remaining: {remain}"


def get_sample(idx, username):
    user_data = get_user_data(username)
    start_id, end_id = user_ranges.get(username, (0, total_num))
    
    if start_id <= idx < end_id:
        wav_path, ref_text = wav_entries[idx]
        entry = user_data.get(idx, {})
        return (
            wav_path,
            ref_text,
            entry.get("correct_text", ""),
            format_progress(idx, start_id, end_id)
        )
    else:
        return None, None, None, "All audio files have been evaluated."


def save_and_next(correct_text, idx, username):
    start_id, end_id = user_ranges.get(username, (0, total_num))
    
    if idx >= end_id:
        return None, None, None, "All audio files have been evaluated.", idx
        
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
        "correct_text": correct_text,
    }
    logging.info(f"Next - {username} - data: {user_data[idx]}")
    save_json_file = os.path.join(spk_save_dir[username], wav_name+".json")
    with open(save_json_file, "w", encoding="utf-8") as f:
        json.dump(user_data[idx], f, ensure_ascii=False, indent=2)
        
    next_idx = idx + 1
    if next_idx >= end_id:
        return None, None, None, "All audio files have been evaluated.", next_idx
    return (*get_sample(next_idx, username), next_idx)


def save_and_prev(correct_text, idx, username):
    start_id, end_id = user_ranges.get(username, (0, total_num))
    
    if idx >= end_id:
        return None, None, None, "All audio files have been evaluated.", idx
    
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
        "correct_text": correct_text,
    }
    logging.info(f"Prev - {username} - data: {user_data[idx]}")
    save_json_file = os.path.join(spk_save_dir[username], wav_name+".json")
    with open(save_json_file, "w", encoding="utf-8") as f:
        json.dump(user_data[idx], f, ensure_ascii=False, indent=2)
        
    prev_idx = idx - 1
    if prev_idx < start_id:
        return None, None, None, "This is the first audio sample.", prev_idx
    return (*get_sample(prev_idx, username), prev_idx)


def get_start_idx(name):
    start_id, end_id = user_ranges.get(name, (0, total_num))
    save_res_spk_dir = os.path.join(save_result_dir, name)
    os.makedirs(save_res_spk_dir, exist_ok=True)
    spk_save_dir[name] = save_res_spk_dir
    evaluated_prefixes = get_evaluated_audio_prefixes(save_res_spk_dir)
    start_idx = start_id + len(evaluated_prefixes)
    return start_idx


def confirm_name(name, start_id, end_id):
    start_id = int(start_id)
    end_id = int(end_id)
    name = name.strip()
    if not name:
        return (
            gr.update(interactive=True), gr.update(value=""),   # name_state 保持为空
            None, None, None, "Please enter your name.", 0,
            gr.update(interactive=False), gr.update(interactive=False),
        )
        
    user_ranges[name] = (start_id, min(total_num, end_id))
    idx0 = get_start_idx(name)
    
    return (
        gr.update(interactive=False), gr.update(value=name),   # 关键：更新 name_state
        *get_sample(idx0, name), idx0,
        gr.update(interactive=True), gr.update(interactive=True)
    )


with gr.Blocks() as demo:
    name_state = gr.State("")
    idx_state = gr.State(0)

    gr.Markdown("<h2 style='text-align: center; color: #4a90e2;'> Checks and Corrections </h2>")

    with gr.Row():
        name_input = gr.Textbox(label="Your Name (Required)", placeholder="Enter your name", interactive=True)
        start_id = gr.Textbox(label="Start ID (Required)", placeholder="Enter Start ID", interactive=True)
        end_id = gr.Textbox(label="End ID (Required)", placeholder="Enter End ID", interactive=True)
        btn_confirm = gr.Button("Confirm", variant="primary")

    with gr.Row():
        audio_player = gr.Audio(label="Audio", type="filepath", interactive=True)

    with gr.Row():
        with gr.Column():
            ref_text = gr.Textbox(label="Ref Text", placeholder="Ref Text", interactive=False)  # interactive=False 表示只读
            correct_text = gr.Textbox(label="Correct Text", placeholder="Enter Correct Text", interactive=True)

        with gr.Column():
            btn_next = gr.Button("Next", variant="primary", interactive=False)
            btn_prev = gr.Button("Previous", variant="primary", interactive=False)
    

    progress_text = gr.Text(label="Progress", interactive=False)
    

    btn_confirm.click(
        fn=confirm_name, 
        inputs=[name_input, start_id, end_id], 
        outputs=[name_input, name_state, 
        audio_player, ref_text, correct_text, progress_text, idx_state,
        btn_next, btn_prev
    ])

    btn_prev.click(save_and_prev, inputs=[correct_text, idx_state, name_input],
                   outputs=[audio_player, ref_text, correct_text, progress_text, idx_state])

    btn_next.click(save_and_next, inputs=[correct_text, idx_state, name_input],
                   outputs=[audio_player, ref_text, correct_text, progress_text, idx_state])
    
    with gr.Column():
        gr.Markdown("### Examples\n")

        for example in examples:
            with gr.Row():  # 整行
                # 左边音频
                gr.Audio(value=example["audio"], label=example["example_type"], type="filepath")
                
                # 右边两行文本
                with gr.Column():
                    gr.Textbox(value=example["ref_text"], label="Ref Text")
                    gr.Textbox(value=example["correct_text"], label="Correct Text")

demo.launch(server_port=7682, server_name="127.0.0.1", share=False, root_path="/page1/",
            allowed_paths=["/data/liuchao/annotation/audio"])
