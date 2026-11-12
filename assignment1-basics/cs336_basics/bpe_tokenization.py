import os
from typing import BinaryIO
from collections import defaultdict
import regex as re
#PAT = r"""'(?:ve|re|ll|[smdt])| ?\p{L}+| ?\p{N}+|\s+(?!\s)|\s+| ?[^\s\p{N}\p{L}]+"""
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
from collections import Counter

class bpe_tokenizer():
    def __init__(self, input_file_path=None, special_token_list=None,vocab=None, merges=None):
        self.input_file_path = input_file_path
        self.special_token_list = special_token_list
        self.counts_total = Counter()
        self.pair_to_tuples: dict[tuple, set] = defaultdict(set)  # (a,b) -> {tuple}
        if self.special_token_list is None:
            self.special_token_list = ["<|endoftext|>"]
        self.special_bytes = {
        t.encode("utf-8") if isinstance(t, str) else t
        for t in self.special_token_list
    }
        if vocab is not None:
            self.vocab = vocab
            self.token_to_id = {v: k for k, v in vocab.items()}
            self.id_to_token = {k: v for k, v in vocab.items()}
        else:
            self.vocab = {i: bytes([i]) for i in range(256)}
        if merges is not None:
            self.merge_list = merges
        else:
            self.merge_list = []
        self.merge_rank = {p: i for i, p in enumerate(self.merge_list)}
        self._pat = self._build_pat()


    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))



    def run_pre_tokenization(self):
        with open(self.input_file_path, "rb") as f:
            num_processes = 4
            boundaries = self.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            #chunk_dict: dict[tuple[bytes], int] = {}
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                storys = chunk.split("<|endoftext|>")
                # Process each story as needed
                counts_chunk = Counter()
                for story in storys:
                    matches = re.finditer(PAT, story)
                    for match in matches:
                        single_byte_tuple = tuple(bytes([b]) for b in match.group().encode("utf-8"))
                        counts_chunk[single_byte_tuple] += 1
                self.counts_total.update(counts_chunk)


    def build_index(self):
        """（重）建倒排索引，训练开始时调用一次"""
        self.pair_to_tuples = defaultdict(set)
        for tup in self.counts_total:
            for i in range(len(tup) - 1):
                self.pair_to_tuples[(tup[i], tup[i+1])].add(tup)


#拿到vocab_counts后，统计每个pair的出现次数，找到出现次数最多的pair，进行merge
    def get_stats_merge(self, vocab_counts: Counter):
        """
        Get statistics about the vocabulary counts.
        """
        max_pair = max(self.pairs, key=lambda k: (self.pairs[k], k))
        if self.pairs[max_pair] <= 0:
            return vocab_counts, None
        new_token = max_pair[0] + max_pair[1]
        affected = list(self.pair_to_tuples.get(max_pair, ()))
        for tup in affected:
            cnt = vocab_counts.pop(tup)
            # 1. 旧 tuple 的所有相邻 pair：减计数、移出索引
            for i in range(len(tup) - 1):
                p = (tup[i], tup[i+1])
                self.pairs[p] -= cnt
                self.pair_to_tuples[p].discard(tup)
            # 2. 只替换相邻且不重迭的 (a,b)
            merged, i, n = [], 0, len(tup)
            while i < n:
                if i < n - 1 and tup[i] == max_pair[0] and tup[i+1] == max_pair[1]:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(tup[i])
                    i += 1
            new_tup = tuple(merged)
            vocab_counts[new_tup] = vocab_counts.get(new_tup, 0) + cnt
            # 3. 新 tuple 的所有相邻 pair：加计数、加索引
            for i in range(len(new_tup) - 1):
                p = (new_tup[i], new_tup[i+1])
                self.pairs[p] = self.pairs.get(p, 0) + cnt
                self.pair_to_tuples[p].add(new_tup)

        self.pairs.pop(max_pair, None)   # 该 pair 已不存在，顺便清掉
        return vocab_counts, max_pair

    def train_bpe(self,  target_vocab_size:int) -> Counter:
        """
        Run the BPE main loop until the target vocabulary size is reached.
        """
        merge_list=[]
        vocab_counts = self.counts_total
        vocab = dict(self.vocab) 
        self.pairs = {}
        for token_tuple, count in vocab_counts.items():
            for token in range(len(token_tuple)-1):
                pair = (token_tuple[token], token_tuple[token+1])
                if pair in self.pairs:
                    self.pairs[pair] += count
                else:
                    self.pairs[pair] = count 

        self.build_index()  # 建倒排索引
        while len(vocab) < target_vocab_size - len(self.special_token_list):
            if not self.pairs:
                break
            vocab_counts,max_pair = self.get_stats_merge(vocab_counts)
            if max_pair is None:
                break
            vocab[len(vocab)] = max_pair[0] + max_pair[1]
            merge_list.append(max_pair)
        j=0
        while j<len(self.special_token_list):
            vocab[len(vocab)] = self.special_token_list[j].encode('utf-8') if isinstance(self.special_token_list[j], str) else self.special_token_list[j]
            j += 1
        self.merge_list = merge_list
        self.vocab = vocab
        self.merge_rank = {p: i for i, p in enumerate(merge_list)}
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.id_to_token = {k: v for k, v in vocab.items()}
        return vocab_counts,vocab,merge_list

    def encode(self, text: str, allowed_special=frozenset()) -> list[int]:
        if not hasattr(self, "merge_rank"):
            self.merge_rank = {p: i for i, p in enumerate(self.merge_list)}
        if not hasattr(self, "_pat"):
            self._pat = self._build_pat()

        # allowed_special 统一成 bytes，和 special_bytes 对齐
        allowed = {a.encode("utf-8") if isinstance(a, str) else a for a in allowed_special}
        if not allowed:
            allowed = self.special_bytes

        out = []
        for match in self._pat.finditer(text):
            g = match.group()
            gb = g.encode("utf-8")

            if gb in self.special_bytes:              # ← special 分支
                if gb not in allowed:
                    raise ValueError(f"disallowed special token {g!r}")
                out.append(self.token_to_id[gb])      # 直接查 id，不过 BPE
                continue

            # 普通文本段：字节化 + rank 合并
            word = tuple(bytes([b]) for b in gb)
            while len(word) > 1:
                best = min(zip(word, word[1:]),
                        key=lambda p: self.merge_rank.get(p, float("inf")))
                if best not in self.merge_rank:
                    break
                merged, new_word, i = best[0] + best[1], [], 0
                while i < len(word):
                    if i < len(word) - 1 and word[i:i + 2] == best:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = tuple(new_word)
            out.extend(self.token_to_id[t] for t in word)

        return out

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of byte tokens back into a string.
        """
        # Convert token IDs back to byte tokens
        byte_tokens = [self.id_to_token[token_id] for token_id in tokens]
        # Join the byte tokens and decode to a string
        return b''.join(byte_tokens).decode('utf-8', errors='ignore')

    def _build_pat(self):
        if not self.special_bytes:
            return re.compile(PAT)
        srt = sorted(self.special_bytes, key=lambda b: (-len(b), b))  # 坑2：长优先
        alt = "|".join(re.escape(b.decode("utf-8")) for b in srt)     # 坑3：escape
        punct = f" ?(?:(?!{alt})[^\\s\\p{{L}}\\p{{N}}])+"             # 坑1：防贪婪吞 special
        return re.compile(
            f"(?:{alt})|'(?:[sdmt]|ll|ve|re)"
            f"| ?\\p{{L}}+| ?\\p{{N}}+|{punct}"
            f"|\\s+(?!\\S)|\\s+"
        )

    def encode_iterable(self,iterable):
        for text in iterable:
            list_yield=self.encode(text)
            i=0
            while i<len(list_yield):
                yield list_yield[i]
                i += 1