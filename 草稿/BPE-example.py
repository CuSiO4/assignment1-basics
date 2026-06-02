'''
cProfile： python -m cProfile -s cumulative
           ↑        ↑         ↑    ↑
        py方式运行   模块名    sort   按累计时间
'''

from collections import Counter,defaultdict
from pprint import pprint
import heapq
# 实现pdf中BPEEAMPLE的实现
def for_get_max_pair(df):
    return max(df.items(),key=lambda x: (x[1],x[0]))
def heap_get_max_pair(w:dict):
    heap = []
    for pair,fre in w.items():
        heapq.heappush(heap,(-fre,Reverse_Bytes(pair[0]),Reverse_Bytes(pair[1]),pair))
    return heapq.heappop(heap)
class Reverse_Bytes():
    def __init__(self,b):
        self.b = b
    def __lt__(self,other):
        return self.b > other.b
    def __eq__(self,other):
        return self.b == other.b
demo_data= "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
words = demo_data.split(" ")
vocabs = Counter(words)
word_tokens = defaultdict(list)
for word in words:
    word_tokens[word] = tuple(word)
print(word_tokens)

merge_step = 1
merge_funct = []
# 暴力实现
# heap快速找到max pair
# 链表
for step in range(merge_step):
    pair_frequency = defaultdict(int)
    relevent_word = defaultdict(list)
    for word in words:
        token = word_tokens[word]
        for i in range(len(token) - 1):
            pair = (token[i],token[i+1])
            pair_frequency[pair] += vocabs[word]
            if (word  not in relevent_word[pair]):
                relevent_word[pair].append(word)
    # merge
    max_pair = get_max_pair(pair_frequency)[0]
    merge_funct.append(''.join(max_pair))
    pprint(merge_funct)
    for w in relevent_word[max_pair]:
        tokens = word_tokens[w]
        new_tokens = []
        j = 0
        while j < len(tokens):
            if (j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == max_pair):
                new_tokens.append(tokens[j] + tokens[j+1])
                j += 2
            else:
                new_tokens.append(token[j]) 
                j += 1 
        word_tokens[w] = new_tokens
    pprint(word_tokens)
