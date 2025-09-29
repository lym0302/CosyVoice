import os
import shutil
import argparse
from tqdm import tqdm

def copy_speaker_folders(in_dir, out_dir):
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    # 遍历输入目录
    for root, dirs, files in os.walk(in_dir):
        for dir_name in dirs:
            if dir_name.startswith("speaker_"):
                full_dir_path = os.path.join(root, dir_name)
                rel_path = os.path.relpath(full_dir_path, in_dir)

                # 提取中间部分作为新文件夹名
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    base = parts[-2].replace("BBC-News-Hindi---Playlists", "BBCNews")
                    speaker = parts[-1].replace("speaker_", "spk")
                    new_dir_name = f"{base}_{speaker}"
                    target_dir = os.path.join(out_dir, new_dir_name)

                    os.makedirs(target_dir, exist_ok=True)

                    files_to_copy = [
                        f for f in os.listdir(full_dir_path)
                        if os.path.isfile(os.path.join(full_dir_path, f))
                    ]

                    for file in tqdm(files_to_copy, desc=f"Copying to {new_dir_name}"):
                        src = os.path.join(full_dir_path, file)
                        dst = os.path.join(target_dir, file)
                        # print("src: ", src)
                        # print("dst: ", dst)
                        shutil.copy2(src, dst)

def main():
    parser = argparse.ArgumentParser(description="Copy speaker folders to flat structure.")
    parser.add_argument("-i", "--in_dir", type=str, required=True, help="Input directory")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    copy_speaker_folders(args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()
