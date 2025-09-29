import pandas as pd

def get_choose_audio_paths(csv_file, path_col="audio_path"):
    """
    从csv文件中读取所有audio_path，返回一个列表
    """
    df = pd.read_csv(csv_file)
    choose_audio_paths = df[path_col].tolist()
    return choose_audio_paths


def filter_data_list(data_list_file, output_file, choose_audio_paths):
    """
    从 data.list 文件中过滤出 audio_path 在 choose_audio_paths 中的行，
    保存到新的文件 data_choose.list
    """
    choose_audio_set = set(choose_audio_paths)  # 用 set 加速查找
    
    with open(data_list_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    selected_lines = []
    for line in lines:
        audio_path = line.strip().split("\t")[0]
        if audio_path in choose_audio_set:
            selected_lines.append(line)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(selected_lines)
    
    print(f"筛选完成，共找到 {len(selected_lines)} 条，已保存到 {output_file}")


if __name__ == "__main__":
    data_list_file = "filelists/bbc07230902_yoyo0904/data.list"
    
    thress = ["300.0", "480.0", "600.0"]
    for thres in thress:
        csv_file = f"filelists/bbc07230902_yoyo0904/snr_mos_tag_choose_final_{thres}.csv"
        output_file = f"filelists/bbc07230902_yoyo0904/data_choose_{thres}.list"
        choose_audio_paths = get_choose_audio_paths(csv_file, path_col="audio_path")
        filter_data_list(data_list_file, output_file, choose_audio_paths)
        
    
    

    
