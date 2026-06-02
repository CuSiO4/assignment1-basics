from collections import defaultdict,Counter
import random
from pprint import pprint
import heapq
# import time
# start_time = time.time()
# end_time = time.time()
# 构造一份数据
def for_get_max_pair(w:dict):
    for step in range(max_step):
        best_pair = max(w.items(),key =lambda x : (x[1],x[0]))[0]
        w[best_pair] = 0
    return 
def heap_get_max_pair(w:dict):
    heap = []
    for pair,fre in w.items():
        heapq.heappush(heap,(-fre,Reverse_Bytes(pair[0]),Reverse_Bytes(pair[1]),pair))
    for step in range(max_step):
        if not heap:
            break
        _,_,_,pair = heapq.heappop(heap)
        w[pair] = 0
    return 
class Reverse_Bytes():
    def __init__(self,b):
        self.b = b
    def __lt__(self,other):
        return self.b > other.b
    def __eq__(self,other):
        return self.b == other.b
with open("./data/TinyStoriesV2-GPT4-valid.txt","r",encoding= "utf-8") as f:
    text = f.read()
words = text.split(" ")
vocabs = defaultdict(int)
random.seed(42)
for word in words:
    vocabs[word] = random.randint(2,10)
word_tokens = defaultdict(list)
for word in words:
    word_tokens[word] = list(word.encode("utf-8"))
max_step = int(1e4)
pair_frequency = defaultdict(int)
relavent_words = defaultdict(list)
for word,fre in vocabs.items():
    tokens = word_tokens[word]
    for i in range(len(tokens) - 1):
        pair = (tokens[i],tokens[i + 1])
        pair_frequency[pair] += fre
        relavent_words[pair].append(word)
#for_get_max_pair(pair_frequency) # 3.661
heap_get_max_pair(pair_frequency) # 0.04
# for step in range(max_step):
#     pair_frequency = defaultdict(int)
#     relavent_words = defaultdict(list)
#     for word,fre in vocabs.items():
#         tokens = word_tokens[word]
#         for i in range(len(tokens) - 1):
#             pair = (tokens[i],tokens[i + 1])
#             pair_frequency[pair] += fre
#             relavent_words[pair].append(word)
#     _,_,_,best_pair = heap_get_max_pair(pair_frequency)
#     print(best_pair)