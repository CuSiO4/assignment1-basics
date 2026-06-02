import argparse
# 创建解析器
parser = argparse.ArgumentParser(description="训练我的语言模型")
# 添加参数
parser.add_argument("--batch_size", type = int, default=32, help="一次性处理多少样本")
parser.add_argument("--no_use_rope", action="store_false", help = "静止使用rope")
parser.add_argument("--norm_mode", type = "str",choices=["pre_norm", "post_norm"], help = "norm的两种模式选择")
# 所有参数放在args里
args = parser.parse_args()
print(f"batch size = {args.batch_size}")