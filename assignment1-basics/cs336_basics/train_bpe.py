from heapq import merge
import os
import regex as re
from collections import defaultdict                                                                                                                                                                          
   
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$|\s+(?!\S)|\s"""

def train_bpe(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 初始化词表
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
          vocab[len(vocab)] = token.encode('utf-8')

    # 合并列表：记录合并顺序
    merges = []
    # 读文件，预分词，统计每个词的出现频率
    word_freqs = defaultdict(int)
    with open(input_path, mode='rb') as f:
        text = f.read().decode('utf-8', errors='ignore')
        # 先按特殊 token 分割，再对每段做正则预分词
        special_pat = '|'.join(re.escape(t) for t in special_tokens)
        segments = re.split(special_pat, text) if special_tokens else [text]
        for segment in segments:
            for word in re.findall(PAT, segment):
                word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
                word_freqs[word_bytes] += 1

    # 第三步：反复合并，直到词表大小达到 vocab_size
    while len(vocab) < vocab_size:

        # 统计所有词内部的字节对频率
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            # 遍历 word 里的相邻对，累加频率
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i+1])] += freq
        if not pair_freqs:
            break

        # 找最高频的对
        max_freq = max(pair_freqs.values())
        best_pair = max(p for p in pair_freqs if pair_freqs[p] == max_freq)

        # 合并：把所有词里的 best_pair 合并
        new_token = best_pair[0] + best_pair[1]
        new_id = len(vocab)
        vocab[new_id] = new_token
        merges.append(best_pair)

        # 更新 word_freqs
        new_word_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            # 把 word 里的 best_pair 替换成 new_token
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            new_word_freqs[new_word] += freq
        word_freqs = new_word_freqs

    return vocab, merges

    
