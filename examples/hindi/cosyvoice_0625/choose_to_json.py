import json
import random

def choose(input_list, output_json, choose_num=200):
    # 读取全部数据
    with open(input_list, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) > choose_num:
        selected_lines = random.sample(lines, choose_num)
    else:
        selected_lines = lines

    # 构建输出字典
    result = {}
    for line in selected_lines:
        wav_path, spk, text, dur, asr_conf = line.strip().split("\t")
        utt = spk + "_" + wav_path.split("/")[-1].replace(".wav", "")
        result[utt] = [text]

    # 保存为 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(selected_lines)} 条数据到 {output_json}")


# choose("filelists_temp/yoyo_sft/test.list", "data_yoyo_sft/test.json")
# exit()


def merge_json_files(path1, path2, output_path, keep='last'):
    """
    合并两个 JSON 文件并去重（按键），结果保存到 output_path。

    参数:
        path1 (str): 第一个 JSON 文件路径。
        path2 (str): 第二个 JSON 文件路径。
        output_path (str): 合并结果保存路径。
        keep (str): 'last' 表示 path2 的值覆盖 path1，'first' 表示保留 path1 的值。
    """
    with open(path1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(path2, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    if keep == 'last':
        merged = {**data1, **data2}
    elif keep == 'first':
        merged = {**data2, **data1}
    else:
        raise ValueError("参数 keep 应为 'first' 或 'last'")
    
    print("llllllllllllllllllllllll: ", len(merged))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"已合并 {path1} 和 {path2}，保存至 {output_path}")

# input_list = "filelists/test.list"
# output_json = "test200.json"
# choose(input_list, output_json)
# merge_json_files("test200.json", "test80.json", "test280.json")

