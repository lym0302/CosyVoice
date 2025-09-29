import os
import shutil
import argparse
from tqdm import tqdm

def extract_spk(filename):
    # 提取前缀下划线
    pre_ext = ''
    for ch in filename:
        if ch == '_':
            pre_ext += '_'
        else:
            break

    # 去掉前缀下划线后的部分
    rest = filename.lstrip('_')
    spk_temp = rest.split('__')[0] if rest else ''
    
    return pre_ext + spk_temp

def organize_by_spk(input_dir, output_dir, ext, dataname):
    # 确保扩展名前带点
    ext = ext if ext.startswith(".") else "." + ext
    ext = ext.lower()

    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir)):
        if not fname.lower().endswith(ext):
            continue
        if dataname == "VL107":
            spk = extract_spk(fname)
            # spk = fname.split("_")[0]
        elif dataname == "mucs":
            spk = fname.replace(f".{ext}", "").split("_")[1]
        else:
            print("eeeeeeeeeeeeeeeerror dataname: {dataname}")
        src_path = os.path.join(input_dir, fname)

        spk_dir = os.path.join(output_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)

        dst_path = os.path.join(spk_dir, fname)
        shutil.copy2(src_path, dst_path)

    print(f"✅ 所有以 {ext} 结尾的文件已按 spk 分类拷贝到 {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="将音频按 spk 拷贝到子目录中")
    parser.add_argument("-i", "--input_dir", required=True, help="原始音频文件夹路径")
    parser.add_argument("-o", "--output_dir", required=True, help="输出文件夹路径")
    parser.add_argument("-e", "--ext", default="wav", help="音频文件扩展名（默认：wav）")
    parser.add_argument("-n", "--dataname", default="VL107", choices=['VL107', 'mucs'], help="数据名称")

    args = parser.parse_args()
    organize_by_spk(args.input_dir, args.output_dir, args.ext, args.dataname)

if __name__ == "__main__":
    main()
