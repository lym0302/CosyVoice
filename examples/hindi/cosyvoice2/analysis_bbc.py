import pandas as pd

def stats_over_threshold(csv_path, threshold_seconds):
    """
    统计时长大于指定阈值的 folder_path 数量、总秒数和总小时数
    
    参数：
        csv_path (str): CSV 文件路径
        threshold_seconds (float): 时长阈值（秒）
    
    返回：
        dict: 包含数量、总秒数、总小时数
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)
    
    # 筛选大于阈值的行
    filtered_df = df[df['total_duration_seconds'] > threshold_seconds]
    
    # 统计数量
    count = len(filtered_df)
    
    # 计算总秒数和小时数
    total_seconds = filtered_df['total_duration_seconds'].sum()
    total_hours = total_seconds / 3600
    
    return {
        "count": count,
        "total_seconds": round(total_seconds, 3),
        "total_hours": round(total_hours, 3)
    }

dirs = ['0723', '0728', '0806', '0807', '0808', '0809', '0810', '0811']
results = {}
dur_count = {0: 0.0, 100: 0.0, 120: 0.0, 180: 0.0, 300: 0.0}
for dd in dirs:
    csv_path = f"/data/youtube_dataset/{dd}/valid-rename-summary.csv"
    raw_result = stats_over_threshold(csv_path, 0)

    print(f"{dd} raw result: ", raw_result)
    results[dd] = {}
    for dur in [0, 100, 120, 180, 300]:
        result = stats_over_threshold(csv_path, dur)
        results[dd][dur] = result
        dur_count[dur] += result["total_hours"]
        print(f"{dd} result that dur more than {dur}: {result}")
    print("############################################################")

print("dur_count: ", dur_count)
