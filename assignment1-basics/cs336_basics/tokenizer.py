import regex as re
from typing import Iterable, Iterator

# GPT-2 / tiktoken 的预分词正则，把文本切成有意义的小块（word）
# 规则按优先级从高到低：
#   1. 英语缩写：'s  't  'll  've  're  'd  'm
#   2. 可选前导空格 + 一串字母（ hello  world）
#   3. 可选前导空格 + 一串数字（ 42  200）
#   4. 可选前导空格 + 标点/符号（,  .  !?）
#   5. 行尾空白
#   6. 后面不跟非空白的空白（尾部空格）
#   7. 单个空白字符
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$|\s+(?!\S)|\s"""


class Tokenizer:
    """
    BPE 分词器（推理阶段使用）。

    训练阶段由 train_bpe() 产出 vocab 和 merges，
    本类负责用这两个产物对任意文本做 encode / decode。

    核心接口：
        encode(text)          str  → list[int]
        decode(ids)           list[int] → str
        encode_iterable(f)    可迭代文本流 → 逐个 yield int（省内存）
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Args:
            vocab:          token id → bytes 的映射，由 train_bpe 产出
            merges:         合并规则列表，顺序即优先级，由 train_bpe 产出
            special_tokens: 特殊 token 字符串列表，如 ["<|endoftext|>"]
                            这些 token 永远作为整体，不会被 BPE 拆开
        """
        self.vocab = vocab
        self.special_tokens = special_tokens or []

        # encode 时需要反向查表：bytes → id
        # 例如：b'hello' → 1234
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        # encode 时按 merges 列表顺序合并，index 越小优先级越高
        # 用字典存是为了 O(1) 查询，而不是每次线性搜索
        # 例如：(b'h', b'e') → 0，表示这是第 1 个合并规则
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

    # ------------------------------------------------------------------ #
    #  公开接口                                                             #
    # ------------------------------------------------------------------ #

    def encode(self, text: str) -> list[int]:
        """
        把字符串编码成 token id 列表。

        流程：
            1. 用特殊 token 把文本切成若干段
            2. 特殊 token 段 → 直接查 id
            3. 普通文本段 → PAT 预分词 → 每个 word 做 BPE 合并 → 查 id
        """
        ids: list[int] = []

        for segment in self._split_on_special_tokens(text):
            if segment in self.special_tokens:
                # 特殊 token 不走 BPE，直接映射到它的 id
                ids.append(self.bytes_to_id[segment.encode('utf-8')])
            else:
                # 普通文本：先用 PAT 切词，再对每个 word 做 BPE
                for word in re.findall(PAT, segment):
                    # word 是字符串，先拆成单字节列表
                    # 例如 "he" → [b'h', b'e']
                    word_bytes = [bytes([b]) for b in word.encode('utf-8')]
                    # 按 merges 规则合并，例如 [b'h', b'e'] → [b'he']
                    merged = self._apply_merges(word_bytes)
                    # 每个合并后的 bytes 查 id
                    ids.extend(self.bytes_to_id[b] for b in merged)

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        把 token id 列表解码回字符串。

        做法：查 vocab 得到每个 id 对应的 bytes，全部拼起来，
        再统一 decode 成 UTF-8 字符串。

        注意：必须先拼 bytes 再统一 decode，不能逐个 decode——
        因为一个 Unicode 字符可能被拆成多个 token，每个 token
        单独 decode 是非法 UTF-8 序列，会出错。
        """
        return b"".join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对可迭代文本流（如打开的文件）逐行 encode，逐个 yield token id。

        与 encode() 的区别：encode() 一次性返回所有 id（需要把整个文本
        加载到内存）；encode_iterable() 是生成器，处理一行产出一行的 id，
        适合处理超大文件，内存占用极低。
        """
        for line in iterable:
            for token_id in self.encode(line):
                yield token_id

    # ------------------------------------------------------------------ #
    #  内部辅助方法                                                         #
    # ------------------------------------------------------------------ #

    def _split_on_special_tokens(self, text: str) -> list[str]:
        """
        用特殊 token 把文本切成若干段，并保留特殊 token 本身。

        例如：
            text = "hello<|endoftext|>world"
            → ["hello", "<|endoftext|>", "world"]

        两个细节：
            - 较长的特殊 token 排在正则前面，防止短的先匹配把长的拆散
              例如有 ["<|end|>", "<|end|><|end|>"] 时，双的应该整体匹配
            - re.split 加括号（捕获组）才会把分隔符本身保留在结果里
        """
        if not self.special_tokens:
            return [text]

        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = '(' + '|'.join(re.escape(t) for t in sorted_specials) + ')'
        # 过滤掉 re.split 在首尾可能产生的空字符串
        return [seg for seg in re.split(pattern, text) if seg]

    def _apply_merges(self, word: list[bytes]) -> list[bytes]:
        """
        对一个 word（单字节列表）反复应用 BPE 合并规则，直到无法再合并。

        每轮：
            1. 扫描所有相邻 pair，找出 merge_rank 最小（最优先）的
            2. 把 word 里所有这个 pair 的出现都合并掉
            3. 重复，直到没有 pair 在 merge_rank 里

        例如 word = [b'h', b'e', b'l', b'l', b'o']：
            轮1：best=(b'h',b'e') → [b'he', b'l', b'l', b'o']
            轮2：best=(b'l',b'l') → [b'he', b'll', b'o']
            轮3：没有可合并的 pair，停止
        """
        while True:
            # 找本轮优先级最高（rank 最小）的相邻 pair
            best_rank = float('inf')
            best_pair = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merge_rank.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # 没有任何 pair 在 merge_rank 里，合并结束
            if best_pair is None:
                break

            # 把 word 里所有 best_pair 的出现替换成合并后的 token
            merged_token = best_pair[0] + best_pair[1]
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(merged_token)
                    i += 2  # 跳过被合并的两个元素
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        return word
