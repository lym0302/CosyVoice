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
#root_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/output/compare_to_minimax"
root_dir = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/output_test1min/to_eval"
playlist_file = os.path.join(root_dir, "playlist.csv")
save_result_dir = os.path.join(root_dir, "save_res")
os.makedirs(save_result_dir, exist_ok=True)
spk_save_dir = {}

user_data_all = {}

spk_ref_audio = {"31036304": "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/others/out_spk_ref/aa/31036304_ref_16k_part2.wav",
                 "37863166": "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice2/others/out_spk_ref/aa/37863166_ref_16k_part.wav",}

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
        wav_path, ref = wav_entries[idx]
        spk = wav_path.split("/")[-1].split("_")[1]
        ref_audio = spk_ref_audio[spk]
        entry = user_data.get(idx, {})
        return (
            wav_path,
            ref_audio,
            ref,
            entry.get("unclear", ""),
            entry.get("missing", ""),
            entry.get("extra", ""),
            entry.get("user_text", ""),
            entry.get("accuracy", 0),
            entry.get("realness", 0),
            entry.get("speaker similarity", 0),
            format_progress(idx)
        )
    else:
        return None, None, "", "", "", "", "", 0, 0, 0, "All audio files have been evaluated."


def save_and_next(ref_text, unclear, missing, extra, user_text, acc, real, sim, idx, username):
    if idx >= total_num:
        return None, None, "", "", "", "", "", 0, 0, 0, "All audio files have been evaluated.", idx
        
    if username not in spk_save_dir:
        save_res_spk_dir = os.path.join(save_result_dir, username)
        os.makedirs(save_res_spk_dir, exist_ok=True)
        spk_save_dir[username] = save_res_spk_dir

    user_data = get_user_data(username)
    wav_path = wav_entries[idx][0]
    wav_name = wav_path.split("/")[-1].replace(".wav", "")
    user_data[idx] = {
        "wav_path": wav_path,
        "ref_text": ref_text,
        "unclear": unclear,
        "missing": missing,
        "extra": extra,
        "user_text": user_text,
        "accuracy": float(acc),
        "realness": float(real),
        "speaker similarity": float(sim),
    }
    logging.info(f"Next - {username} - data: {user_data[idx]}")
    save_json_file = os.path.join(spk_save_dir[username], wav_name+".json")
    with open(save_json_file, "w", encoding="utf-8") as f:
        json.dump(user_data[idx], f, ensure_ascii=False, indent=2)
        
    next_idx = idx + 1
    if next_idx >= total_num:
        return None, None, "", "", "", "", "", 0, 0, 0, "All audio files have been evaluated.", next_idx
    return (*get_sample(next_idx, username), next_idx)


def save_and_prev(ref_text, unclear, missing, extra, user_text, acc, real, sim, idx, username):
    if idx >= total_num:
        return None, None, "", "", "", "", "", 0, 0, 0, "All audio files have been evaluated.", idx
    
    if username not in spk_save_dir:
        save_res_spk_dir = os.path.join(save_result_dir, username)
        os.makedirs(save_res_spk_dir, exist_ok=True)
        spk_save_dir[username] = save_res_spk_dir

    user_data = get_user_data(username)
    wav_path = wav_entries[idx][0]
    wav_name = wav_path.split("/")[-1].replace(".wav", "")
    user_data[idx] = {
        "wav_path": wav_path,
        "ref_text": ref_text,
        "unclear": unclear,
        "missing": missing,
        "extra": extra,
        "user_text": user_text,
        "accuracy": float(acc),
        "realness": float(real),
        "speaker similarity": float(sim)
    }
    logging.info(f"Prev - {username} - data: {user_data[idx]}")
    save_json_file = os.path.join(spk_save_dir[username], wav_name+".json")
    with open(save_json_file, "w", encoding="utf-8") as f:
        json.dump(user_data[idx], f, ensure_ascii=False, indent=2)
        
    prev_idx = idx - 1
    if prev_idx < 0:
        return None, None, "", "", "", "", "", 0, 0, 0, "This is the first audio sample.", prev_idx
    return (*get_sample(prev_idx, username), prev_idx)


def get_start_idx(name):
    start_idx = 0
    save_res_spk_dir = os.path.join(save_result_dir, name)
    os.makedirs(save_res_spk_dir, exist_ok=True)
    spk_save_dir[name] = save_res_spk_dir
    evaluated_prefixes = get_evaluated_audio_prefixes(save_res_spk_dir)
    start_idx = len(evaluated_prefixes)
    return start_idx


# def confirm_name(name):
#     if not name.strip():
#         return (
#             gr.update(interactive=True), "",
#             None, "", "", "", "", "", 0, 0, "Please enter your name.", 0,
#             gr.update(interactive=False), gr.update(interactive=False),
#         )
#     idx0 = get_start_idx(name)
#     return (
#         gr.update(interactive=False), name, 
#         *get_sample(idx0, name), idx0,
#         gr.update(interactive=True), gr.update(interactive=True)
#     )


def confirm_name(name):
    if not name.strip():
        return (
            gr.update(interactive=True), gr.update(value=""),   # name_state 保持为空
            None, None, "", "", "", "", "", 0, 0, 0, "Please enter your name.", 0,
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
        ref_audio_player = gr.Audio(label="Speaker Ref Audio", type="filepath", interactive=True)

    with gr.Column():
        ref_text = gr.Textbox(label="Ref Text", placeholder="Enter reference transcription...", lines=1, max_lines=1, interactive=False)
        unclear_text = gr.Textbox(label="Enter the unclear or mispronounced words.", placeholder="Enter unclear words...", lines=1, max_lines=1)
        missing_text = gr.Textbox(label="Enter the missing words.", placeholder="Enter missing words...", lines=1, max_lines=1)
        extra_text = gr.Textbox(label="Enter the extra words.", placeholder="Enter extra words...", lines=1, max_lines=1)
        user_text = gr.Textbox(label="If audio doesn't match reference text, write what you heard.", placeholder="What do you actually hear...", lines=1, max_lines=1)

    with gr.Row():
        with gr.Column():
            mos_accuracy = gr.Slider(label="Pronunciation Accuracy", minimum=0, maximum=5, step=0.5, value=0)
            mos_realness = gr.Slider(label="Naturalness", minimum=0, maximum=5, step=0.5, value=0)
            sim = gr.Slider(label="Speaker Similarity", minimum=0, maximum=5, step=0.5, value=0)
        with gr.Column():
            btn_next = gr.Button("Next", variant="primary", interactive=False)
            btn_prev = gr.Button("Previous", variant="primary", interactive=False)
            

    # export_status = gr.Textbox(label="Export Status", interactive=False)
    progress_text = gr.Text(label="Progress", interactive=False)
    

    btn_confirm.click(
        fn=confirm_name, 
        inputs=[name_input], 
        outputs=[name_input, name_state, 
        audio_player, ref_audio_player, ref_text, unclear_text, missing_text, extra_text, user_text,
        mos_accuracy, mos_realness, sim, progress_text, idx_state,
        btn_next, btn_prev
    ])

    btn_prev.click(save_and_prev, inputs=[ref_text, unclear_text, missing_text, extra_text, user_text, mos_accuracy, mos_realness, sim, idx_state, name_input],
                   outputs=[audio_player, ref_audio_player, ref_text, unclear_text, missing_text, extra_text, user_text, mos_accuracy, mos_realness, sim, progress_text, idx_state])

    btn_next.click(save_and_next, inputs=[ref_text, unclear_text, missing_text, extra_text, user_text, mos_accuracy, mos_realness, sim, idx_state, name_input],
                   outputs=[audio_player, ref_audio_player, ref_text, unclear_text, missing_text, extra_text, user_text, mos_accuracy, mos_realness, sim, progress_text, idx_state])


demo.launch(server_port=7680, server_name="127.0.0.1", share=False,
            allowed_paths=["/data/liangyunming/dataset/hindi/data_v2_sft/audio_loudnorm_-16_choose_mos3_snr30",
                           "/data/liangyunming/dataset/hindi/data_v2/audio_loudnorm_-16"])
