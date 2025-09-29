# coding=utf-8
import gradio as gr
import csv

wav_file = "/data/liangyunming/tts_20250618/CosyVoice/examples/hindi/cosyvoice_0625/output/tttemp/wav.scp"

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

mos_options = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

def format_progress(idx):
    done = idx  # 已评数（索引从0开始，当前条算“待评”，所以done = idx）
    remain = total_num - done
    return f"共 {total_num} 条，已评 {done} 条，剩余 {remain} 条"

def get_sample(idx):
    if 0 <= idx < total_num:
        wav_path = wav_paths[idx]
        prev_text = user_data.get(idx, {}).get("text", "")
        prev_mos = user_data.get(idx, {}).get("mos", 3)
        progress = format_progress(idx)
        return wav_path, prev_text, str(prev_mos), progress
    else:
        return None, "", "3", "已完成所有音频"

def save_and_next(text, mos, idx):
    user_data[idx] = {"text": text, "mos": float(mos)}
    next_idx = idx + 1
    if next_idx >= total_num:
        return None, "", "3", "已完成所有音频", next_idx
    wav_path = wav_paths[next_idx]
    prev_text = user_data.get(next_idx, {}).get("text", "")
    prev_mos = user_data.get(next_idx, {}).get("mos", 3)
    progress = format_progress(next_idx)
    return wav_path, prev_text, str(prev_mos), progress, next_idx

def save_and_prev(text, mos, idx):
    user_data[idx] = {"text": text, "mos": float(mos)}
    prev_idx = idx - 1
    if prev_idx < 0:
        return None, "", "3", "第一条音频", prev_idx
    wav_path = wav_paths[prev_idx]
    prev_text = user_data.get(prev_idx, {}).get("text", "")
    prev_mos = user_data.get(prev_idx, {}).get("mos", 3)
    progress = format_progress(prev_idx)
    return wav_path, prev_text, str(prev_mos), progress, prev_idx

def export_csv():
    csv_path = "mos_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "text", "mos"])
        for idx, wav_path in enumerate(wav_paths):
            text = user_data.get(idx, {}).get("text", "")
            mos = user_data.get(idx, {}).get("mos", "")
            writer.writerow([wav_path, text, mos])
    return f"导出完成，文件路径：{csv_path}"

with gr.Blocks() as demo:
    gr.Markdown("# 音频听感评分工具")

    idx_state = gr.State(0)

    audio_player = gr.Audio(label="播放音频", interactive=False)
    text_input = gr.Textbox(label="填写文本内容", lines=2, placeholder="请填写文本或备注")
    mos_dropdown = gr.Dropdown(label="选择 MOS 分数", choices=[str(x) for x in mos_options], value="3")
    progress_text = gr.Text(label="进度", interactive=False)

    btn_prev = gr.Button("上一条")
    btn_next = gr.Button("下一条")
    btn_export = gr.Button("导出CSV")

    def load_first(idx):
        return get_sample(idx)

    btn_prev.click(save_and_prev, inputs=[text_input, mos_dropdown, idx_state], outputs=[audio_player, text_input, mos_dropdown, progress_text, idx_state])
    btn_next.click(save_and_next, inputs=[text_input, mos_dropdown, idx_state], outputs=[audio_player, text_input, mos_dropdown, progress_text, idx_state])
    demo.load(load_first, inputs=idx_state, outputs=[audio_player, text_input, mos_dropdown, progress_text])

    btn_export.click(export_csv, inputs=None, outputs=gr.Textbox(label="导出状态"))

# demo.launch(server_port=7680, server_name="0.0.0.0")
demo.launch(
    server_port=7680,
    server_name="127.0.0.1",  # 只监听本地，避免直接外网暴露
    #inbrowser=True,
    share=False,
    #root_path="/tag"
)
