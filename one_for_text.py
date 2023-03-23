"""
被试者文本特征提取
"""
import warnings
import torch
from transformers import BertTokenizer  # 这个pip 一下,transformers
from transformers import BertModel
import numpy as np
import os

warnings.filterwarnings("ignore")
topics = ['positive', 'neutral', 'negative']
answers = {}
text_features = []
text_targets = []
ss = []


def extract_features(text_features, path):
    """
    range里面放多少个被试者
    """
    for index in range(1):
        """
        这里是存放被试者的信息的目录，和EATD一样,path就是子目录的一个前缀
        """
        if os.path.isdir("E:/大创/Depression-recognition/EATD-Corpus/" + str(path) + str(index + 1)):
            answers[index + 1] = []
            for topic in topics:
                with open("E:/大创/Depression-recognition/EATD-Corpus/" + str(path) + str(index + 1) + "/" + '%s.txt' % (topic), 'r',
                          encoding="utf-8") as f:
                    lines = f.readlines()[0]
                    marked_text = "[CLS] " + lines + " [SEP]"
                    print(marked_text)
                    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                    tokenized_text = tokenizer.tokenize(marked_text)
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    segments_ids = [1] * len(tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens])
                    segments_tensors = torch.tensor([segments_ids])
                    model = BertModel.from_pretrained('bert-base-chinese',
                                                      output_hidden_states=True)
                    model.eval()
                    with torch.no_grad():
                        outputs = model(tokens_tensor, segments_tensors)
                        hidden_states = outputs[2]
                    token_embeddings = torch.stack(hidden_states, dim=0)
                    token_embeddings = torch.squeeze(token_embeddings, dim=1)
                    token_embeddings = token_embeddings.permute(1, 0, 2)
                    token_vecs = hidden_states[-2][0]
                    sentence_embedding = torch.mean(token_vecs, dim=0)
                    answers[index + 1].append(sentence_embedding)
            temp = []
            for i in range(3):
                temp.append(np.array(answers[index + 1][i]))
            text_features.append(temp)


extract_features(text_features, 't_')
print("Saving npz file locally...")
"""
predict_text.npz存放被试者提取的三类文本向量
"""
np.savez('E:/大创/Depression-recognition/reg_feature/predict_text.npz', text_features)
