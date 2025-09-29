# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Generator
import json
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import inflect
try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation
from transformers import AutoTokenizer, AutoModelForTokenClassification

class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option,
                                                                     providers=["CUDAExecutionProvider" if torch.cuda.is_available() else
                                                                                "CPUExecutionProvider"])
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
            spks = self.spk2info.keys()
            logging.info(f"Support spks: {spks}")
        else:
            logging.error(f'error to get spk2info: {spk2info}')
            self.spk2info = {}
        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=True)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()
            
            
        # add hindi punc predict
        # hindi_punc_model_name="zicsx/Hindi-Punk"
        # hindi_punc_model_name = "/home/ec2-user/.cache/huggingface/hub/models--zicsx--Hindi-Punk/snapshots/9dc5cc1cf14c4686c03d308b9d4d5eb68d2e2c94/"

        # # 加载 tokenizer 和模型
        # self.hindi_punc_tokenizer = AutoTokenizer.from_pretrained(hindi_punc_model_name, use_fast=True)
        # self.hindi_punc_model = AutoModelForTokenClassification.from_pretrained(hindi_punc_model_name)
        # self.hindi_punc_model.to(self.device)
        # self.hindi_punc_model.eval()
        # self.hindi_punc_id2label = self.hindi_punc_model.config.id2label

    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will return _extract_text_token_generator!')
            # NOTE add a dummy text_token_len for compatibility
            return self._extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32).to(self.device)
        else:
            text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i: i + 1]

    def _extract_speech_token(self, speech):
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None,
                                                         {self.speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          self.speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len
    
    
    def hindi_punc_predict(self, text):
        encoding = self.hindi_punc_tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.hindi_punc_model(input_ids)
            logits = outputs.logits  # shape: [1, seq_len, num_labels]
            predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()  # [seq_len]
        tokens = self.hindi_punc_tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        punctuated_text = ""
        for token, pred_idx in zip(tokens, predictions):
            # 跳过特殊 token
            if token in self.hindi_punc_tokenizer.all_special_tokens:
                continue
            punct_label = self.hindi_punc_id2label[pred_idx]
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
    
    # def text_split_hindi(self, text):
        """
        印地语文本切分函数：
        1. 小于10个字直接返回
        2. 首先按照句号、问号、感叹号及非数字间的小数点切分
        3. 如果切分后的片段长度 > 30，按照逗号进行切分
        4. 如果逗号切分之后还大于30字，则先进行标点预测，再根据标点重新切分
        5. 保留原始标点
        """
        # import pdb; pdb.set_trace()
        text = re.sub(r'\s+', ' ', text).strip()   # 将多个空格统一为一个空格
        if len(text.split(" ")) < 10:  # 小于10个词直接返回
            return [text.strip()]

        # 第一步切分（保留标点，排除小数点）
        parts = re.split(r'((?<!\d)[।!?\.](?!\d))', text)
        # 将文本和标点组合回句子
        sentences = []
        if len(parts) == 1:
            sentences = [parts[0].strip()]
        else:    
            for i in range(0, len(parts)-1, 2):
                sent = parts[i].strip()
                punct = parts[i+1] if i+1 < len(parts) else ''
                if sent or punct:  # 避免空字符串
                    sentences.append(f"{sent}{punct}")

        final_texts = []

        for sent in sentences:
            if len(sent.split(" ")) <= 30:
                final_texts.append(sent)
            else:
                # 先按原始逗号切分
                comma_parts_raw = re.split(r'(,)', sent)
                comma_parts = []
                i = 0
                while i < len(comma_parts_raw):
                    p = comma_parts_raw[i].strip()
                    # 如果下一个是逗号，则合并
                    if i + 1 < len(comma_parts_raw) and comma_parts_raw[i+1] == ',':
                        p = f"{p},"
                        i += 1
                    if p:
                        comma_parts.append(p)
                    i += 1
                
                for part in comma_parts:
                    if len(part.split(" ")) <= 30:
                        final_texts.append(part)
                    else:
                        # 逗号切分后仍 >30词，做标点预测再切分
                        predicted_text = self.hindi_punc_predict(part)
                        print("##################0000000000000000000: ", part)
                        print("##################1111111111111111111: ", predicted_text)

                        # 根据标点再次切分（保留标点）
                        sub_parts = re.split(r'((?<!\d)[।!?\.](?!\d))', predicted_text)
                        sub_sentences = []
                        if len(sub_parts) == 1:
                            sub_sentences = [sub_parts[0].strip()]
                        else:
                            for j in range(0, len(sub_parts)-1, 2):
                                sub_sent = sub_parts[j].strip()
                                sub_punct = sub_parts[j+1] if j+1 < len(sub_parts) else ''
                                if sub_sent or sub_punct: # 避免空字符串
                                    sub_sentences.append(f"{sub_sent}{sub_punct}")

                        final_texts.extend(sub_sentences)

        return final_texts
    

    def text_split_hindi(self, text):
        """
        印地语文本切分函数：
        1. 小于10个字直接返回
        2. 按句号、问号、感叹号切分
        3. 长句子 >30 -> 先逗号切，再预测切，再逗号切
        4. 保留原始标点
        """
        def split_by_punct(text, pattern):
            """按给定正则切分并保留标点"""
            parts = re.split(f'({pattern})', text)
            sentences = []
            if len(parts) == 1:
                return [parts[0].strip()]
            for i in range(0, len(parts)-1, 2):
                sent = parts[i].strip()
                punct = parts[i+1] if i+1 < len(parts) else ''
                if sent or punct:
                    sentences.append(f"{sent}{punct}")
            return sentences
        
        def split_long_sentence(text, max_len=30):
            """递归处理：先按逗号，再标点预测，再按逗号"""
            if len(text.split()) <= max_len:
                return [text]

            # 先按逗号切
            comma_parts = []
            raw = re.split(r'(,)', text)
            i = 0
            while i < len(raw):
                p = raw[i]
                if i + 1 < len(raw) and raw[i+1] == ',':
                    p = f"{p},"
                    i += 1
                p = p.strip()
                if p:
                    comma_parts.append(p)
                i += 1

            results = []
            for part in comma_parts:
                if len(part.split()) <= max_len:
                    results.append(part)
                else:
                    # 标点预测
                    predicted_text = self.hindi_punc_predict(part)
                    subs = split_by_punct(predicted_text, r'(?<!\d)[।!?\.](?!\d)')
                    for sub in subs:
                        if len(sub.split()) <= max_len:
                            results.append(sub)
                        else:
                            # 标点切分后仍然过长 -> 再按逗号切
                            results.extend(split_long_sentence(sub, max_len))
            return results
    
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text.split()) < 20:
            return [text]

        sentences = split_by_punct(text, r'(?<!\d)[।!?\.](?!\d)')
        final_texts = []
        for sent in sentences:
            if len(sent.split()) <= 30:
                final_texts.append(sent)
            else:
                final_texts.extend(split_long_sentence(sent, max_len=30))
        return final_texts


    
    

    def text_normalize(self, text, split=True, text_frontend=True, lang=None):
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will skip text_normalize!')
            return [text]
        
        if lang == 'hindi' or lang == 'hi':
            text = text.strip()
            texts = self.text_split_hindi(text)
            texts = [i for i in texts if not is_only_punctuation(i)]
            return texts if split is True else text
            
        if text_frontend is False or text == '':
            return [text] if split is True else text
        text = text.strip()
        
        
        if self.use_ttsfrd:
            texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
            text = ''.join(texts)
        else:         
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                            token_min_n=60, merge_len=20, comma_split=False))
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                            token_min_n=60, merge_len=20, comma_split=False))
            
                
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]['embedding']
        # print("11111111111111111111: ", embedding, embedding.shape)
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        if zero_shot_spk_id == '':
            prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
            prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
            speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
            if resample_rate == 24000:
                # cosyvoice2, force speech_feat % speech_token = 2
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
                speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
                
            embedding = self._extract_spk_embedding(prompt_speech_16k)
            model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                           'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                           'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                           'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                           'llm_embedding': embedding, 'flow_embedding': embedding}
        else:
            model_input = self.spk2info[zero_shot_spk_id]
        model_input['text'] = tts_text_token
        model_input['text_len'] = tts_text_token_len
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k, resample_rate, zero_shot_spk_id)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

    def frontend_instruct2(self, tts_text, instruct_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        # model_input = self.frontend_zero_shot(tts_text, instruct_text + '<|endofprompt|>', prompt_speech_16k, resample_rate, zero_shot_spk_id)
        model_input = self.frontend_zero_shot(tts_text, instruct_text + ' <|endofprompt|> ', prompt_speech_16k, resample_rate, zero_shot_spk_id)
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        model_input = {'source_speech_token': source_speech_token, 'source_speech_token_len': source_speech_token_len,
                       'flow_prompt_speech_token': prompt_speech_token, 'flow_prompt_speech_token_len': prompt_speech_token_len,
                       'prompt_speech_feat': prompt_speech_feat, 'prompt_speech_feat_len': prompt_speech_feat_len,
                       'flow_embedding': embedding}
        return model_input
