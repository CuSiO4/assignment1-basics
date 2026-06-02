import regex as re
#Q：为什么允许token前面带空格 A：token 自带空格，decode 时不用插空格。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pre_token_list = re.findall(PAT, "some test that i'll pre-tokenize")
print(pre_token_list)
#实际使用finditer而不是findall，findall一次性将所有内容存在内存里
#finditer每次返回一个batchsiz

