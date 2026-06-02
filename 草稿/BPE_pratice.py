from pprint import pprint
from collections import Counter, defaultdict
import heapq
import random
random.seed(42)

'''
1. pair count 
2. max pair
3. merge
--------删3条
--------merge
--------加2条
--------heap 惰性删除
'''
# 反转Bytes  用于建立大根堆
class ReverseBytes:
    def __init__(self,b):
        self.b = b
    def __lt__(self,other):
        return self.b > other.b
    def __eq__(self,other):
        return self.b == other.b
    pass
# 初始化heap表
def init_heap(pair_count_dict,id_to_tokens):
    heap = []
    for pair, cnt in pair_count_dict.items():
        a, b = pair
        heapq.heappush(heap,(-cnt, ReverseBytes(id_to_tokens[a]), ReverseBytes(id_to_tokens[b]),pair))
    return heap
def get_best_pair(heap, pair_count_dict):
    while heap:
        neg_cnt, _, _, pair = heapq.heappop(heap)
        real_cnt = pair_count_dict.get(pair, -1)
        if real_cnt < 0 or real_cnt != -neg_cnt:
            continue
        return pair, real_cnt
    return None, 0
# 初始化词表  序列：tokens
def init_id_to_bytes_vocabs(speical_words):
    special_token_vocabs = {i : tok.encode("utf-8") for i,tok in enumerate(speical_words)}
    offset = len(speical_words)
    vocabs = {i + offset : bytes([i]) for i in range(256)}
    return special_token_vocabs | vocabs
def init_bytes_to_id(vocabs):
    return {a : b for b, a in vocabs.items()}
def init_node_linkedList(id_to_word, bytes_to_id):
    node_list = []  #节点列表
    next_node_list = [] #指向下一个节点
    prev_node_list = [] #指向上一个节点
    live_node_list = [] #是否有效
    node_word_of_list = [] # 表示当前token属于哪个word
    for word_id, word in enumerate(id_to_word):
        tokens = [bytes([x]) for x in word.encode("utf-8")]
        prev_node_id = -1
        for token in tokens:
            # token id 和node id
            # 前者表示当前node 的token 是什么
            # 后者表示序号
            token_id = bytes_to_id[token]
            node_id = len(node_list)
            node_list.append(token_id)
            node_word_of_list.append(word_id)
            live_node_list.append(True)
            prev_node_list.append(prev_node_id)
            next_node_list.append(-1)
            if (prev_node_id != -1):
                next_node_list[prev_node_id] = node_id 
            prev_node_id = node_id
    return node_list, next_node_list, prev_node_list, live_node_list, node_word_of_list
    # pprint(node_list)
    # pprint(next_node_list)
    # pprint(prev_node_list)
    # pprint(live_node_list)
    pass
def init_pair_linkedList(node_list, next_node_list, live_node_list, node_word_of_list, word_fre_list):
    # 初始化pair链---需要三个数据结构
    pair_occ_list = [] # 当前list表
    pair_lastest_occ_dict = {} # pair ： index
    pair_prev_occ_list = [] # 当前index相同pair的上一个出现位置
    pair_count_dict = defaultdict(int)
    # 想明白node_id 和 token_id 的关系
    for node_id, left_token_id in enumerate(node_list):
        # 检查合法性
        if not live_node_list[node_id]:
            continue
        right_node_id = next_node_list[node_id]
        if right_node_id == -1:
            continue
        if not live_node_list[right_node_id] :
            continue
        pair = (node_list[node_id], node_list[right_node_id])
        word_id = node_word_of_list[node_id]
        pair_count_dict[pair] += word_fre_list[word_id]

        pair_id = len(pair_occ_list)
        pair_occ_list.append(node_id)
        prev_pair_left_id = pair_lastest_occ_dict.get(pair, -1)
        pair_prev_occ_list.append(prev_pair_left_id)
        pair_lastest_occ_dict[pair] = pair_id

    return pair_occ_list, pair_lastest_occ_dict, pair_prev_occ_list, pair_count_dict
def train_bpe(words, special_tokens, vocab_size):
    '''
    输入：words [word...]
         special_tokens [分割词]
         vocab_size 词表大小
    输出：vocabs 字典 {id：bytes} bytes为初始化的字典+后序合并的每一个token
         merge_list 列表 [pair] 每一次合并结果
    '''
    word_count = Counter(words)
    id_to_word = list(word_count.keys())
    word_fre_list = [word_count[word] for word in id_to_word]

    vocabs = init_id_to_bytes_vocabs(special_tokens)
    bytes_to_id = init_bytes_to_id(vocabs)
    node_list, next_node_list, prev_node_list, live_node_list, node_word_of_list = init_node_linkedList(id_to_word,bytes_to_id)
    pair_occ_list, pair_lastest_occ_dict, pair_prev_occ_list, pair_count_dict = init_pair_linkedList(node_list, next_node_list, live_node_list, node_word_of_list, word_fre_list)
    pair_heap = init_heap(pair_count_dict, vocabs)
    merge_list = []
    merge_times = max(0, vocab_size - len(vocabs))
    next_new_pair_id = len(vocabs)
    for _ in range(merge_times):
        best_pair, real_cnt = get_best_pair(pair_heap,pair_count_dict)
        if best_pair == None:
            break
        left_token_id, right_token_id = best_pair
        new_token_id = next_new_pair_id
        vocabs[new_token_id] = vocabs[left_token_id] + vocabs[right_token_id]
        merge_list.append((vocabs[left_token_id] , vocabs[right_token_id]))
        next_new_pair_id += 1
        # 寻找best_pair出现次数
        occ_id_list = []
        occ_id = pair_lastest_occ_dict.get(best_pair, -1)
        while occ_id != -1:
            occ_id_list.append(occ_id)
            occ_id = pair_prev_occ_list[occ_id]
        # 遍历每一个best_pair
        for occ_id in reversed(occ_id_list):
            node_id = pair_occ_list[occ_id]
            if not live_node_list[node_id]:
                continue
            right_node_id = next_node_list[node_id]
            if right_node_id == -1:
                continue
            if not live_node_list[right_node_id]:
                continue
            if (node_list[node_id], node_list[right_node_id]) != best_pair:
                continue
            word_id = node_word_of_list[node_id]
            fre = word_fre_list[word_id]
            pair_count_dict[best_pair] -= fre
            # 三条边的删除 ---删掉pair_count_dict中三条边，同时heap插入更新后的元组
            cnt = pair_count_dict[best_pair]
            if cnt > 0:
                a, b = best_pair    # a,b are token_id
                heapq.heappush(pair_heap, (-cnt, ReverseBytes(vocabs[a]),ReverseBytes(vocabs[b]), best_pair))

            left_neighbor_node_id = prev_node_list[node_id]
            right_neighbor_node_id = next_node_list[right_node_id]

            if left_neighbor_node_id != -1 and live_node_list[left_neighbor_node_id]:
                old_left_pair = (node_list[left_neighbor_node_id],node_list[node_id])
                pair_count_dict[old_left_pair] -= fre
                cnt = pair_count_dict[old_left_pair]
                update_pair = old_left_pair
                if cnt > 0:
                    a, b = update_pair   # a,b are token_id
                    heapq.heappush(pair_heap, (-cnt, ReverseBytes(vocabs[a]),ReverseBytes(vocabs[b]), update_pair))

            if right_neighbor_node_id != -1 and live_node_list[right_neighbor_node_id]:
                old_right_pair = (node_list[right_node_id],node_list[right_neighbor_node_id])
                pair_count_dict[old_right_pair] -= fre
                cnt = pair_count_dict[old_right_pair]
                update_pair = old_right_pair
                if cnt > 0:
                    a, b = update_pair   # a,b are token_id
                    heapq.heappush(pair_heap, (-cnt, ReverseBytes(vocabs[a]),ReverseBytes(vocabs[b]), update_pair))
            
            '''
            before: a,b,c,d
            now: a,bc,c,d
            修改node内容:
            1.live_node_list 
            2.指向 next和prev
            3.添加两条新的pair:(a,bc), (bc,d)
            '''

            # node 相关修改
            live_node_list[right_node_id] = False
            next_node_list[node_id] = right_neighbor_node_id
            if right_neighbor_node_id != -1:
                prev_node_list[right_neighbor_node_id] = node_id
            node_list[node_id] = new_token_id

            # 新增两条pair
            if left_neighbor_node_id != -1 and live_node_list[left_neighbor_node_id]:
                new_left_pair = (node_list[left_neighbor_node_id],node_list[node_id])
                pair_count_dict[new_left_pair] += fre

                occ_id = len(pair_occ_list)
                pair_occ_list.append(left_neighbor_node_id)
                pair_prev_occ_list.append(pair_lastest_occ_dict.get(new_left_pair,-1)) 
                pair_lastest_occ_dict[new_left_pair] = occ_id

                cnt = pair_count_dict[new_left_pair]
                update_pair = new_left_pair
                if cnt > 0:
                    a, b = update_pair   # a,b are token_id
                    heapq.heappush(pair_heap, (-cnt, ReverseBytes(vocabs[a]),ReverseBytes(vocabs[b]), update_pair))

            if right_neighbor_node_id != -1 and live_node_list[right_neighbor_node_id]:
                new_right_pair = (node_list[node_id],node_list[right_neighbor_node_id])
                pair_count_dict[new_right_pair] += fre

                occ_id = len(pair_occ_list)
                pair_occ_list.append(node_id)
                pair_prev_occ_list.append(pair_lastest_occ_dict.get(new_right_pair,-1)) 
                pair_lastest_occ_dict[new_right_pair] = occ_id

                cnt = pair_count_dict[new_right_pair]
                update_pair = new_right_pair
                if cnt > 0:
                    a, b = update_pair   # a,b are token_id
                    heapq.heappush(pair_heap, (-cnt, ReverseBytes(vocabs[a]),ReverseBytes(vocabs[b]), update_pair))
    return vocabs, merge_list


if __name__ == "__main__":
    text = "banana apple peach grape bear car candante anicode anchar anitip"
    special_tokens = []
    vocab_size = 400

    words = text.split()
    vocabs, merge_list = train_bpe(words,special_tokens,vocab_size)
    pprint(merge_list)
    pprint(vocabs)
    # pprint(pair_count)
