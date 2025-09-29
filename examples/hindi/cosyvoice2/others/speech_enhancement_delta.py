#coding=utf-8
# conda activate ClearerVoice-Studio
import os
import argparse
from clearvoice import ClearVoice

import os
import subprocess
from tqdm import tqdm

def convert_to_16k(input_dir, output_dir):
    """
    将 input_dir 下的所有音频文件（wav, mp3）转换为 16kHz，并保存到 output_dir
    保留原始目录结构
    """
    for root, _, files in os.walk(input_dir):
        for f in tqdm(files):
            if not f.lower().endswith(('.wav', '.mp3')):
                continue

            infile = os.path.join(root, f)
            rel_path = os.path.relpath(infile, input_dir)  # 相对路径
            out_path = os.path.join(output_dir, rel_path)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 创建输出目录
            # 使用 ffmpeg 转换采样率
            subprocess.run(["ffmpeg", "-y", "-i", infile, "-ar", "16000", out_path], check=True)


# convert_to_16k("/data2/nginx_files/annotation/audio", "/data2/nginx_files/annotation/audio_16k")
# exit()


def enhance_directory(input_root, output_root):
    print("==> 初始化模型...")
    #cv_se = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    cv_se = ClearVoice(task="speech_enhancement", model_names=["MossFormerGAN_SE_16K"])
    print("==> 模型初始化完成")

    for dirpath, _, filenames in os.walk(input_root):
        for fname in filenames:me
            if not fname.lower().endswith(".wav"):
                continue

            input_path = os.path.join(dirpath, fname)
            relative_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, relative_path)

            # ✅ 跳过已存在的文件
            if os.path.exists(output_path):
                print(f"⏩ 已存在，跳过: {output_path}")
                continue

            print(f"==> 开始处理音频: {input_path}")
            try:
                output_wav = cv_se(input_path=input_path, online_write=False)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv_se.write(output_wav, output_path=output_path)
                print(f"✅ 已保存至: {output_path}")
            except Exception as e:
                print(f"❌ 处理失败: {input_path}\n错误信息: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch enhance wav files using ClearVoice")
    parser.add_argument("-i", "--input_dir", help="Path to input directory containing .wav files")
    parser.add_argument("-o", "--output_dir", help="Path to output directory for enhanced files")

    args = parser.parse_args()
    enhance_directory(args.input_dir, args.output_dir)

