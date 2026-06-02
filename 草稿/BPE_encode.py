
'''
╔══════════════════════════════════════════════════════════════╗
║                    BPE Encode 笔记                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  【整体流程】                                                 ║
║  1. 预分词：先剔除 special_tokens（用 re.sub 删掉）           ║
║     或用 re.split 保留它们（作业要求保留）                    ║
║  2. 对每个 pre-token 独立编码：                               ║
║     a. str → encode("utf-8") → 遍历得到 int                  ║
║     b. int → bytes([int]) → 单字节 tokens 列表                ║
║     c. 循环应用 merges，直到无可用的 merge 为止                ║
║  3. 查 bytes_to_id 字典，把 bytes tokens 转成整数 ID           ║
║                                                              ║
║  【核心 merge 循环：while + 双重 break】                      ║
║  - 每次成功 merge 后必须回到 merges[0] 重新扫描                ║
║  - 因为先前的 merge 可能让更早的 merge 变得可用                ║
║  - break 的作用：中断内层遍历，回到 while 开头                ║
║                                                              ║
║  【切片赋值一行完成 merge】                                    ║
║  tokens[i:i+2] = [a + b]                                     ║
║  等价于：删除 i,i+1 并在 i 处插入新元素                        ║
║  break 立即跳出 → 不会因序列变短导致索引问题                   ║
║                                                              ║
║  【内存考虑（大文件分块处理）】                                ║
║  - 大文件无法全装入内存 → 分 chunk 处理                        ║
║  - 每个 chunk 边界必须对齐到 token 边界                       ║
║  - 用 find_chunk_boundaries() 在特殊词边界切分                 ║
║  - 保证分块处理结果 == 全量处理结果的拼接                      ║
║                                                              ║
║  【用 list 不用链表的原因】                                    ║
║  - 每个 pre-token 只有几个到几十个字节                         ║
║  - merge 次数很少（几十条）                                    ║
║  - 切片赋值已经足够简洁高效                                    ║
║  - 链表的复杂度在这里不划算                                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
'''
#str ---> bytes  encode
#int ---> bytes bytes([])
import regex as re
mlist = [('t','h'),('th', 'e'),('a', 't')]
merged_list = [(merge[0].encode("utf-8"), merge[1].encode("utf-8")) for merge in mlist]
special_tokens = ["<|endoftext|>"]
text = "the cat ate  <|endoftext|>"
vocabs = {
    0: b' ',
    1: b'a',
    2: b'c',
    3: b'e',
    4: b'h',
    5: b't',
    6: b'th',
    7: b'c',
    8: b'a',
    9: b'the',
    10: b'at',
}
bytes_to_id ={ x : y for y, x in vocabs.items()}
# 能够处理specialtokens
def init_bpe_encode(text, merged_list, special_tokens, bytes_to_id):
    PAT = "|".join(re.escape(token) for token in special_tokens)
    text = re.sub(PAT, "", text)
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
init_bpe_encode(text, merged_list, special_tokens, bytes_to_id)

    


    
