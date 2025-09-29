import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name="zicsx/Hindi-Punk"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(model_name)
model.to(device)
model.eval()
id2label = model.config.id2label

def predict_punctuated_text(text):
    """
    输入印地语文本，输出带标点的文本
    """

    # 分词
    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)

    # 模型预测
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape: [1, seq_len, num_labels]
        predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()  # [seq_len]

    # 将 token 对应的标点转换为文本
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    punctuated_text = ""
    for token, pred_idx in zip(tokens, predictions):
        # 跳过特殊 token
        if token in tokenizer.all_special_tokens:
            continue
        punct_label = id2label[pred_idx]
        if token.startswith("##"):
            # subword 拼接
            punctuated_text += token[2:]
        else:
            punctuated_text += " " + token
        # 添加预测标点
        if punct_label and punct_label != "":
            punctuated_text += punct_label

    # 去掉开头多余空格
    punctuated_text = punctuated_text.strip()
    return punctuated_text


if __name__ == "__main__":
    text = "घुड़सवारी और तीरंदाजी शामिल है वो कई भाषाओं की विद्वान थी और फ़्रेंच अंग्रेजी और उर्दू जैसी भाषाओं में भी कुशल थी रानी का विवाह शिवगंगाई के राजा से हुआ वेलु नचियार के पति मुथु वधुगनाथ पेरिया वुडिया थेवर सत्रह सौ अस्सी में ईस्ट इंडिया कंपनी के सैनिकों के साथ एक लड़ाई में मारे गए"

    result = predict_punctuated_text(text)
    print("aaaaaaaaaaaaaaaa: ", text)
    print("bbbbbbbbbbbbbbbb: ", result)
