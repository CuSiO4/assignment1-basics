#********************** utf-8 ********************************
# test_string = "hello! こんにちは!"
# utf8_encoded = test_string.encode("utf-8")
# utf16_encoded = test_string.encode("utf-16")
# utf32_encoded = test_string.encode("utf-32")
# print(len(utf32_encoded))
# print(len(utf16_encoded))
# print(len(utf8_encoded))


#************************decode_utf8_bytes_to_str_wrong*****************
# def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
#     #列表拼装字符串
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
# str = decode_utf8_bytes_to_str_wrong("你".encode("utf-8"))
# print(str)


#***************给出一个无法解码的两字节序列********************
# utf-8 的起始字节决定该字符长度，连续字节不能单独出现
#print(bytes([255,255]).decode("utf-8"))

# int 编程bytes是用bytes（）， str变成bytes 用 encode（）