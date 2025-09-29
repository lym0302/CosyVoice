# coding=utf-8

import pandas as pd
import argparse
import os

human_voice_labels = {
    'Speech', 'woman speaking', 'man speaking', 'Male speech', 'Female speech',
    'Child speech', 'kid speaking', 'monologue', 'Narration', 'Conversation',
    'Speech synthesizer',
    'Giggle', 'Chuckle', 'chortle', 'Snicker', 'Wail', 'moan', 'Groan', 'Whimper',
    'Breathing', 'Gasp', 'Pant', 'Sigh', 'Throat clearing',
    'Cough', 'Hiccup', 'Burping',
    'eructation'
}


def filter_csv_with_stats(inp_csv_path, out_csv_path, snr_thres=30, mos_thres=3.5):
    """
    筛选 CSV 中满足条件的行，并保存到新 CSV，同时统计保留行 duration 占比。

    条件：
        snr_db > snr_thres
        p808_mos > mos_thres
        top1_label == 'Speech'
    
    参数：
        inp_csv_path : 输入 CSV 文件路径
        out_csv_path : 输出 CSV 文件路径
    """
    # 读取 CSV
    df = pd.read_csv(inp_csv_path)

    # 计算原始总 duration
    total_duration = df["duration"].sum()
    
    def all_tags_in_human_voice(top_tags_str):
        """
        判断 top_tags 字符串中所有标签是否都在 human_voice_labels 集合中
        top_tags_str 格式示例: "Speech: 0.678, Speech synthesizer: 0.622"
        """
        if pd.isna(top_tags_str):
            return False
        tags = [t.split(":")[0].strip() for t in top_tags_str.split(",")]
        return all(tag in human_voice_labels for tag in tags)

    # 筛选条件
    filtered_df = df[
        (df["snr_db"] > snr_thres)
        & (df["p808_mos"] > mos_thres)
        & (df["top1_label"] == "Speech")
        & df["top_tags"].apply(all_tags_in_human_voice)
    ]

    # 保存到输出文件
    filtered_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    # 统计保留行 duration 总和及比例
    filtered_duration = filtered_df["duration"].sum()
    duration_ratio = filtered_duration / total_duration if total_duration > 0 else 0.0
    
    
    from collections import Counter
    # 统计每个标签数量
    tag_counter = Counter()
    for tags_str in filtered_df["top_tags"].dropna():
        tags = tags_str.split(",")
        for t in tags:
            label = t.split(":")[0].strip()
            tag_counter[label] += 1

    # # 按数量从大到小排序显示
    # print(f"{len(tag_counter)}标签数量统计（按数量从大到小）：")
    # for tag, count in tag_counter.most_common():
    #     print(f"{tag}: {count}")
        

    print(f"保留行数: {len(filtered_df)} / {len(df)}")
    print(f"保留行 duration 总和: {filtered_duration:.3f} s, {filtered_duration/3600:.3f} h")
    print(f"原始总 duration: {total_duration:.3f} s, {total_duration/3600:.3f} h")
    print(f"duration 占比: {duration_ratio:.3%}")
    # print(f"所有 top_tags 类型 ({len(top_tags_set)} 个): {top_tags_set}")


def summarize_by_spk(inp_csv_path, out_csv_path):
    # 读取 CSV
    df = pd.read_csv(inp_csv_path)
    
    # 提取 spk
    df['spk'] = df['audio_path'].apply(lambda x: x.split("/")[-2])
    
    # 按 spk 统计总时长和条数
    summary = df.groupby('spk')['duration'].agg(['sum', 'count']).reset_index()
    summary.rename(columns={'sum': 'total_duration', 'count': 'num_samples'}, inplace=True)
    
    # 按总时长降序排序
    summary = summary.sort_values(by='total_duration', ascending=False)
    
    # 保存 CSV
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    summary.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved summary to {out_csv_path}")


def get_final_csv(out_csv_path, duration_thres, spk_count_path, final_csv_path):
    final_csv_path = final_csv_path.replace(".csv", "") + "_" + str(duration_thres) + ".csv"  # 改名附上threshold
    
    """
    根据 spk 的总时长阈值筛选对应的音频行，并统计总 duration 占比
    """
    # 读取 spk 统计 CSV
    spk_df = pd.read_csv(spk_count_path)
    
    # 筛选出总时长大于阈值的 spk
    # valid_spks = spk_df[spk_df['total_duration'] >= duration_thres]['spk'].tolist()
    valid_spks = spk_df.loc[spk_df['total_duration'] >= duration_thres, 'spk'].astype(str).tolist()
    print(f"Total valid spk: {len(valid_spks)}")
    
    # 读取 out_csv_path
    df = pd.read_csv(out_csv_path)
    
    # 计算原始总 duration
    total_duration = df['duration'].sum()
    
    # 保留 valid spk 对应的行
    final_df = df[df['audio_path'].apply(lambda x: x.split('/')[-2] in valid_spks)]
    
    # 保存最终 CSV
    os.makedirs(os.path.dirname(final_csv_path), exist_ok=True)
    final_df.to_csv(final_csv_path, index=False, encoding='utf-8-sig')
    
    # 统计总时长和占比
    final_duration = final_df['duration'].sum()
    duration_ratio = final_duration / total_duration if total_duration > 0 else 0.0
    
    print(f"保留行数: {len(final_df)} / {len(df)}")
    print(f"保留行 duration 总和: {final_duration:.3f} s, {final_duration/3600:.3f} h")
    print(f"原始总 duration: {total_duration:.3f} s, {total_duration/3600:.3f} h")
    print(f"duration 占比: {duration_ratio:.3%}")


    
    
def main():
    parser = argparse.ArgumentParser(description="筛选 CSV 并统计 duration 占比")
    # parser.add_argument("-i", "--inp_csv_path", type=str, required=True, help="输入 CSV 文件路径")
    # parser.add_argument("-o", "--out_csv_path", type=str, required=True, help="输出 CSV 文件路径")
    # parser.add_argument("-s", "--spk_count_path", type=str, required=True, help="输出统计 spk 的 CSV 文件路径")
    # parser.add_argument("-f", "--final_csv_path", type=str, required=True, help="输出最终的csv文件")
    # parser.add_argument("-n", "--data_name", type=str, required=True, help="data name")
    # parser.add_argument("-t", "--duration_thres", type=float, default=600, help="每个人的时长不低于这个阈值")


    # args = parser.parse_args()
    # duration_thres = args.duration_thres
    # data_name = args.data_name
    # inp_csv_path = f"filelists/{data_name}/snr_mos_tag.csv"
    # out_csv_path = f"filelists/{data_name}/snr_mos_tag_choose.csv"
    # spk_count_path = f"filelists/{data_name}/snr_mos_tag_choose_spkcount.csv"
    # final_csv_path = f"filelists/{data_name}/snr_mos_tag_choose_final.csv"
    

    # filter_csv_with_stats(inp_csv_path, out_csv_path)
    # summarize_by_spk(out_csv_path, spk_count_path)
    # get_final_csv(out_csv_path, duration_thres, spk_count_path, final_csv_path)
    
    
    
    data_names = ["bbc_0723_0811",  "bbc_0827_0830",  "bbc_0901_0902", "yoyo_20250904"]
    duration_thress = [300.0, 480.0, 600.0]
    
    for data_name in data_names:
        print(f"####################### {data_name} ############################")
        inp_csv_path = f"filelists/{data_name}/snr_mos_tag.csv"
        out_csv_path = f"filelists/{data_name}/snr_mos_tag_choose.csv"
        spk_count_path = f"filelists/{data_name}/snr_mos_tag_choose_spkcount.csv"
        final_csv_path = f"filelists/{data_name}/snr_mos_tag_choose_final.csv"
        filter_csv_with_stats(inp_csv_path, out_csv_path)
        summarize_by_spk(out_csv_path, spk_count_path)
        for duration_thres in duration_thress:
            get_final_csv(out_csv_path, duration_thres, spk_count_path, final_csv_path)
        
        

if __name__ == "__main__":
    main()