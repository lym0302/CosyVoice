from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def trim_silence_with_padding(input_file, output_file, silence_thresh=-40, max_padding_ms=300):
    """
    去掉开头和结尾静音，但保留不超过 max_padding_ms 的静音
    """
    audio = AudioSegment.from_file(input_file, format="wav")

    # 检测非静音区间
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)

    if not nonsilent_ranges:
        # 整段都是静音
        audio[:0].export(output_file, format="wav")
        return

    # 取第一段非静音和最后一段非静音
    start = nonsilent_ranges[0][0]
    end = nonsilent_ranges[-1][1]

    # 前后保留 padding，最大不超过 max_padding_ms
    start = max(0, start - min(start, max_padding_ms))
    end = min(len(audio), end + min(len(audio)-end, max_padding_ms))

    trimmed_audio = audio[start:end]
    trimmed_audio.export(output_file, format="wav")
    print(f"处理完成：{output_file}")

import sys
input_file = sys.argv[1]
output_file = sys.argv[2]
trim_silence_with_padding(input_file, output_file)
