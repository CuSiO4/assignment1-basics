from tests.common import gpt2_bytes_to_unicode
import json
import regex as re
class Tokenizer():
    def __init__(self, vocab:dict[int, bytes], merges:list[tuple[bytes,bytes]], special_tokens: list[str] = None ):
        self.vocab = vocab
        self.id_to_bytes = vocab
        self.bytes_to_id = {v:k for k,v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens | []
        
        if special_tokens:
            sorted_special_tokens = sort(special_tokens, key = len, reversed = True)
            spacial_pat = "|".join(re.escape(s) for s in sorted_special_tokens)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # 备选构造
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        with open(vocab_filepath,"r") as f:
            raw_vocab = json.read(f)
        raw_merges = []
        with open(merges_filepath,"r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    raw_merges.append(tuple(parts))
        reverse_map = {a : b for b, a in gpt2_bytes_to_unicode().items()}
        vocab = {}
        for key_str, idx in raw_vocab.itmes():
            vocab[idx] = bytes(reverse_map[ch] for ch in key_str)
        merges = []
        for a_str, b_str in raw_merges:
            a_bytes = bytes(reverse_map[ch] for ch in a_str)
            b_bytes = bytes(reverse_map[ch] for ch in b_str)
            merges.append((a_bytes, b_bytes))

        cls(vocab, merges, special_tokens)
    def encode(self, text:str)->list[int]:
        text = re.sub(self.special_regex, "", text)
        words = text.split()
        words_bpe_encode = []
        for word in words:
            tokens = [bytes([ch]) for ch in word.encode("utf-8")]
            merged = True
            while len(tokens) != 1 and merged:
                for a, b in merged_list:
                    merged = False
                    for i in range(len(tokens) - 1):
                        if tokens[i]== a and tokens[i+1]==b:
                            tokens[i:i+2] = [a + b]
                            merged = True
                            break
                        if merged:
                            break
            token_id_list = [bytes_to_id[t] for t in tokens]
            words_bpe_encode.append(token_id_list)
        print(words_bpe_encode)
        return words_bpe_encode
        pass
    def encode_iterable(iterable)->Iterator[int]:

        pass
    def decode(self, ids)->str:
        
        pass
