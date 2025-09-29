# from cosyvoice.tokenizer.tokenizer import QwenTokenizer

# model_path = "../../../pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN"

# tokenizer = QwenTokenizer(model_path)
# tokens = tokenizer.encode("नमस्ते दुनिया")  # Hello world in Hindi
# print("11111111111111: ", tokens, len(tokens))
# print("22222222222222: ", tokenizer.decode(tokens))




# from cosyvoice.tokenizer.tokenizer import get_tokenizer

# tokenizer = get_tokenizer(multilingual=True, num_languages=100, language='hi', task='transcribe')
# tokens = tokenizer.encode("नमस्ते दुनिया")  # Hello world in Hindi
# print("33333333333333333: ", tokens, len(tokens))
# print("44444444444444444: ", tokenizer.decode(tokens))

import torchaudio.compliance.kaldi as kaldi
import torch

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
    
    
    

