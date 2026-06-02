import os
import regex as re
# from pretokenization_example import find_chunk_boundaries
import multiprocessing
from collections import Counter, defaultdict
import heapq
from pprint import pprint
# 1.初始化利用正则预分词 2.统计每个预分词出现的频率，用字典统计
# def train_bpe(
#     input_str : str,
#     vocab_size : int,
#     special_tokens : list[str]
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # '''
    # 输入：
    # input_str: 训练数据的路径
    # vocab_size: 最大的最终词汇量
    # special_tokens: 字符边界
    # 输出：
    # vocab：从整数到字节的映射
    # merges：每个列表项都是一个字节元组 (<token1>, <token2>)，表示 <token1> 与 <token2> 被合并。合并项应按创建顺序排列。
    # '''
    # assert vocab_size > 256
    # #读取文件-并行分为多个chunk 按照特数token拆分 正则预分词
    # pre_tokenizer_frequency = {}
    # tokens = []
    # with open(input_str,'rb') as f:
    #     num_processes = 4
    #     #multiprocessing.Pool(num_processes)
    #     boundaries = find_chunk_boundaries(f,num_processes,b"<|endoftext|>")
    #     for start, end in zip(boundaries[:-1],boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8",errors="ignore")
    #         delimiter = "|".join(re.escape(t) for t in special_tokens)
    #         # re.split收到反斜杠的影响
    #         segments = re.split(delimiter, chunk)
    #         for segment in segments:
    #             if segment == "":   # 跳过空片段
    #                 continue
    #             PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #             for iter in re.finditer(PAT,segment):
    #                 token = iter.group()
    #                 # 因为有的字符有多个Bytes，为了拆分最小单位
    #                 encode_token = token.encode("utf-8")
    #                 token_list = tuple(bytes([x]) for x in encode_token) 
    #                 tokens.append(token_list)
    #                 if token_list in pre_tokenizer_frequency:
    #                     pre_tokenizer_frequency[token_list] +=1
    #                 else:
    #                     pre_tokenizer_frequency[token_list] = 1
    # #初始化词汇表
    # vocab = {}
    # for i in range(256):
    #     vocab[i] = bytes([i])
    # for i,str in enumerate(special_tokens):
    #     vocab[256 + i] = str.encode("utf-8")
    # #初始化合并列表
    # merges = []
    # while(len(vocab) != vocab_size):
    #     #初始化pair_frequency
    #     pair_frequency = {}
    #     for idx,pre_tokenizer in enumerate(pre_tokenizer_frequency):
    #         key = pre_tokenizer
    #         for i in range(len(key) - 1):
    #             pair = tuple([key[i],key[i+1]])
    #             if pair in pair_frequency:
    #                 pair_frequency[pair] += pre_tokenizer_frequency[key]
    #             else:
    #                 pair_frequency[pair] = pre_tokenizer_frequency[key]
    #     # 找到最大的pair
    #     max_pair = max(pair_frequency,key= lambda p: (pair_frequency[p],p))
    #     #print(max_pair,pair_frequency[max_pair])
    #     # 合并
    #     a,b = max_pair
    #     vocab[len(vocab)] = a+b
    #     merges.append(tuple([a,b]))
    #     # 遍历所有tokens
    #     new_tokens = []
    #     for token in tokens:
    #         lst = list(token)
    #         new_list = []
    #         i = 0
    #         while i < len(lst):
    #             if i < len(lst) - 1 and lst[i] == a and lst[i+1] == b:
    #                 new_list.append(a + b)
    #                 i += 2
    #             else:
    #                 new_list.append(lst[i])
    #                 i += 1
    #         new_tokens.append(tuple(new_list))
    #     # 根据new_tokens 然后呢创建新的频率表
    #     tokens = new_tokens
    #     pre_tokenizer_frequency = {}
    #     for token in tokens:
    #         if token in pre_tokenizer_frequency:
    #             pre_tokenizer_frequency[token] += 1
    #         else:
    #             pre_tokenizer_frequency[token] = 1
    # return vocab, merges
    
class ReverseByte():
    def __init__(self,b):
        self.b = b
    def __lt__(self,other):
        return self.b > other.b
    def __eq__(self,other):
        return self.b == other.b
def init_pair_heap(pair_count_dict, id_to_bytes):
    heap = []
    for pair, cnt in pair_count_dict.items():
        a, b = pair
        heapq.heappush(heap, (-cnt, ReverseByte(id_to_bytes[a]),ReverseByte(id_to_bytes[b]), pair))
    return heap
def get_best_pair(pair_heap, pair_count_dict):
    while pair_heap:
        neg_cnt,_,_,best_pair = heapq.heappop(pair_heap)
        if -neg_cnt == pair_count_dict[best_pair] and neg_cnt < 0:
            return best_pair, -neg_cnt
    return None, 0
def init_vocabs(special_tokens):
    special_vocabs = {i : x.encode("utf-8") for i, x in enumerate(special_tokens)}
    #special_tokens = {}
    offset = len(special_vocabs)
    vocabs = {i + offset : bytes([i]) for i in range(256)}
    return special_vocabs | vocabs
def init_bytes_to_id(id_to_bytes):
    return {x : y for y, x in id_to_bytes.items()}

def init_node_linklist(words, bytes_to_id):
    node_list = []
    next_node_list = []
    pre_node_list = []
    live_node_list = []
    word_of_node_list = []

    for word_id, word in enumerate(words):
        tokens = [bytes([x]) for x in word.encode("utf-8")]
        pre_node_id = -1
        for token in tokens:
            node_id = len(node_list)
            node_list.append(bytes_to_id[token])
            pre_node_list.append(pre_node_id)
            next_node_list.append(-1)
            if pre_node_id != -1:
                next_node_list[pre_node_id] = node_id
            live_node_list.append(True)
            word_of_node_list.append(word_id)
            pre_node_id = node_id
    return node_list, next_node_list, pre_node_list, live_node_list, word_of_node_list

def init_pair_linkList(node_list, next_node_list, word_of_node_list, word_count_list, live_node_list):
    pair_list = []
    latest_pair_dict = {}
    pre_pair_list = []
    pair_count_dict = defaultdict(int)

    for left_node_id, left_token_id in enumerate(node_list):
        if not live_node_list[left_node_id]:
            continue
        right_node_id = next_node_list[left_node_id]
        if right_node_id == -1:
            continue
        if not live_node_list[right_node_id]:
            continue
        right_token_id = node_list[right_node_id]
        pair = (left_token_id,right_token_id)
        pair_id = len(pair_list)
        pair_list.append(left_node_id)
        pre_pair_list.append(latest_pair_dict.get(pair, -1))
        latest_pair_dict[pair] = pair_id

        frq = word_count_list[word_of_node_list[left_node_id]]
        pair_count_dict[pair] += frq
    return pair_list, latest_pair_dict, pre_pair_list, pair_count_dict 

def train_bpe(path, vocab_size,special_tokens ):
    # 1.读取path，然后进行正则预先分词
    # 2.分词后，得到words， word_count_list, 初始化node， pair，和heap
    # 3.进行vocabs_size - len 循环
    # 4.每次找到一个best pair. 删除三条pair 合并node 增加两条pair， 三步每一步都要heappush操作
    # 5.返回vocabs， merged_list
    with open(path, "r",encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        split_PAT = "|".join(re.escape(special) for special in special_tokens)
        parts = re.split(split_PAT, text)
        train_segments = [p for p in parts if p not in special_tokens]
    else:
        train_segments = [text]
    gpt_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words = []
    for segment in train_segments:
        # extend
        words.extend(re.findall(gpt_PAT, segment))
    words_count_dict = Counter(words)
    id_to_words = [word for word in words_count_dict.keys()]
    word_count_list = [fre for fre in words_count_dict.values()]
    vocabs = init_vocabs(special_tokens)
    bytes_to_id = init_bytes_to_id(vocabs)
    node_list, next_node_list, pre_node_list, live_node_list, word_of_node_list = init_node_linklist(id_to_words, bytes_to_id)
    pair_list, latest_pair_dict, pre_pair_list, pair_count_dict  = init_pair_linkList(node_list, next_node_list, word_of_node_list, word_count_list, live_node_list)
    merged_list = []
    merged_times = max(0, vocab_size - len(vocabs))
    pair_heap = init_pair_heap(pair_count_dict, vocabs)
    for step in range(merged_times):
        best_pair, real_cnt = get_best_pair(pair_heap, pair_count_dict)
        if best_pair == None:
            continue
        if real_cnt <= 0:
            continue
        left_token_id, right_token_id = best_pair
        new_token_id = len(vocabs)
        vocabs[new_token_id] = vocabs[left_token_id] + vocabs[right_token_id]
        merged_list.append((vocabs[left_token_id],vocabs[right_token_id]))
        if step >= 42 and step <= 46:
            print(f"Step {step}: selected {(vocabs[best_pair[0]], vocabs[best_pair[1]])} with count {real_cnt}")
        pair_id = latest_pair_dict.get(best_pair,-1)
        best_pair_node_id_list = []
        while pair_id != -1: 
            best_pair_node_id_list.append(pair_id)
            pair_id = pre_pair_list[pair_id]
            
        for pair_id in reversed(best_pair_node_id_list):
            left_node_id = pair_list[pair_id]
            # 删除两条边
            if not live_node_list[left_node_id]:
                continue
            right_node_id = next_node_list[left_node_id]
            if right_node_id == -1:
                continue
            if not live_node_list[right_node_id]:
                continue
            # 缺一行代码，结果出错
            if node_list[left_node_id] != best_pair[0] or node_list[right_node_id] != best_pair[1]:
                continue
            word_id = word_of_node_list[left_node_id]
            pair_count_dict[best_pair] -= word_count_list[word_id]
            cnt = pair_count_dict[best_pair]
            if cnt > 0:
                a, b = best_pair
                heapq.heappush(pair_heap,(-cnt, ReverseByte(vocabs[a]), ReverseByte(vocabs[b]), best_pair))
            
            left_neighbor_node_id = pre_node_list[left_node_id]
            right_neighbor_node_id = next_node_list[right_node_id]
            # 删左
            if left_neighbor_node_id != -1 and live_node_list[left_neighbor_node_id]:
                old_left_pair = (node_list[left_neighbor_node_id],node_list[left_node_id])
                word_id = word_of_node_list[left_neighbor_node_id]
                pair_count_dict[old_left_pair] -= word_count_list[word_id]
                cnt = pair_count_dict[old_left_pair]
                update_pair = old_left_pair
                if cnt > 0:
                    a, b = update_pair
                    heapq.heappush(pair_heap,(-cnt, ReverseByte(vocabs[a]), ReverseByte(vocabs[b]), update_pair))
            # 删右
            if right_neighbor_node_id != -1 and live_node_list[right_neighbor_node_id]:
                old_right_pair = (node_list[right_node_id],node_list[right_neighbor_node_id])
                word_id = word_of_node_list[right_node_id]
                pair_count_dict[old_right_pair] -= word_count_list[word_id]
                cnt = pair_count_dict[old_right_pair]
                update_pair = old_right_pair
                if cnt > 0:
                    a, b = update_pair
                    heapq.heappush(pair_heap,(-cnt, ReverseByte(vocabs[a]), ReverseByte(vocabs[b]), update_pair))
            # a b c d
            # a bc c d
            node_list[left_node_id] = new_token_id
            next_node_list[left_node_id] = right_neighbor_node_id
            if right_neighbor_node_id != -1:
                pre_node_list[right_neighbor_node_id] = left_node_id
            live_node_list[right_node_id] = False

            # 新pair(a, bc) (bc, d)
            if left_neighbor_node_id != -1 and live_node_list[left_neighbor_node_id]:
                new_left_pair = (node_list[left_neighbor_node_id], node_list[left_node_id])
                pair_id = len(pair_list)
                pair_list.append(left_neighbor_node_id)
                pre_pair_list.append(latest_pair_dict.get(new_left_pair, -1))
                latest_pair_dict[new_left_pair] = pair_id

                word_id = word_of_node_list[left_neighbor_node_id]
                pair_count_dict[new_left_pair] += word_count_list[word_id]
                cnt = pair_count_dict[new_left_pair]
                update_pair = new_left_pair
                if cnt > 0:
                    a, b = update_pair
                    heapq.heappush(pair_heap,(-cnt, ReverseByte(vocabs[a]), ReverseByte(vocabs[b]), update_pair))
            # 新pair(a, bc) (bc, d)
            if right_neighbor_node_id != -1 and live_node_list[right_neighbor_node_id]:
                new_right_pair = (node_list[left_node_id], node_list[right_neighbor_node_id])
                pair_id = len(pair_list)
                pair_list.append(left_node_id)
                pre_pair_list.append(latest_pair_dict.get(new_right_pair, -1))
                latest_pair_dict[new_right_pair] = pair_id
                word_id = word_of_node_list[left_node_id]
                pair_count_dict[new_right_pair] += word_count_list[word_id]
                cnt = pair_count_dict[new_right_pair]
                update_pair = new_right_pair
                if cnt > 0:
                    a, b = update_pair
                    heapq.heappush(pair_heap,(-cnt, ReverseByte(vocabs[a]), ReverseByte(vocabs[b]), update_pair))
    return vocabs, merged_list
def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转换为可见字符串
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v) 
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    
    # 合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt" # 你的原始文本路径
    vocab_size = 10000 # 作业要求的词表大小
    # input_path = "data/owt_train.txt" 
    # input_path = "data/chinese.txt" 
    # vocab_size = 1000 # 作业要求的词表大小
    
    special_tokens = ["<|endoftext|>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    print("这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率。")
    
    # 调用你之前写好的逻辑
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    # 保存结果
    save_tokenizer_files(vocab, merges, output_dir)


            
if __name__ == "__main__":
    main()