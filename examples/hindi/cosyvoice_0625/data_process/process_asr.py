# coding=utf-8
import difflib
import re


def get_itn_lexical_mapping(lexical, itn):
    lex_words = lexical.split()
    itn_words = itn.split()
    sm = difflib.SequenceMatcher(a=itn_words, b=lex_words)
    itn_to_lex_mapping = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue  # 相同的部分不用记录
        elif tag in ('replace', 'delete', 'insert'):
            # 获取对应片段
            itn_segment = " ".join(itn_words[i1:i2]) if i2 > i1 else None
            lex_segment = " ".join(lex_words[j1:j2]) if j2 > j1 else None
            if itn_segment is not None and lex_segment is not None:
                itn_to_lex_mapping.append((itn_segment, lex_segment))
    return itn_to_lex_mapping

def restore_masked_itn_by_order(masked_itn, itn_to_lex_mapping):
    """
    按照 mapping 列表顺序还原 masked_itn 中的片段
    """
    restored = masked_itn
    for itn_seg, lex_seg in itn_to_lex_mapping:
        # 使用正则全词匹配
        pattern = r'\b{}\b'.format(re.escape(itn_seg))
        restored = re.sub(pattern, lex_seg, restored, count=1)  # 每次只替换第一个出现
    return restored


def restore_display_from_masked(display, masked_words):
    restored = display
    for lex_seg in [l for _, l in masked_words]:
        # 只替换第一个 ***
        restored = restored.replace("***", lex_seg, 1)    
        
    return restored


lexical = "today i have nine hundren dollars so i get the first place may have a lot of money to buy a car"
itn = "today i have 900 dollars so i get 1st place may have a lot of money to buy a car"
masked_itn = "today i have 900 dollars so i get 1st place may have a lot *** to buy ***"
display = "today i have 900 dollars, so i get 1st place. may have a lot *** to buy ***."



# lexical = "लड़ खाती है जैसे कोई कहानी कह रही हो दूर से आती मिट्टी की खुशबू बचपन की यादों की जगह देते हैं बचपन की यादों को जगा देते हैं कभी कभी ऐसे ही पलों के पलों में ज़िन्दगी की"
# itn = "लड़ खाती है जैसे कोई कहानी कह रही हो दूर से आती मिट्टी की खुशबू बचपन की यादों की जगह देते हैं बचपन की यादों को जगा देते हैं कभी कभी ऐसे ही पलों के पलों में ज़िन्दगी की"
# masked_itn = "*** खाती है जैसे कोई कहानी कह रही हो। दूर से आती मिट्टी की खुशबू बचपन की यादों की जगह देते हैं बचपन की यादों को जगा देते हैं कभी कभी ऐसे ही पलों के पलों में ज़िन्दगी की।"
# display = "*** खाती है जैसे कोई कहानी कह रही हो। दूर से आती मिट्टी की खुशबू बचपन की यादों की जगह देते हैं, बचपन की यादों को जगा देते हैं कभी कभी ऐसे ही पलों के पलों में ज़िन्दगी की।"

itn_to_lex_mapping = get_itn_lexical_mapping(lexical, itn) 
print(itn_to_lex_mapping)
restore_masked_itn = restore_masked_itn_by_order(masked_itn, itn_to_lex_mapping)
print(restore_masked_itn)
masked_words = get_itn_lexical_mapping(lexical, restore_masked_itn)
masked_words = [pair for pair in masked_words if pair[0] == "***"]  # 只保留 ***
print(masked_words)
restore_display = restore_display_from_masked(display, masked_words)
print(restore_display)
restore_display_tn = restore_masked_itn_by_order(restore_display, itn_to_lex_mapping)
print(restore_display_tn)