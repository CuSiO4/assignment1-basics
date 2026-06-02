from modules import Linear, Embedding, RMSNorm, SwiGLU, RoPE, softmax,\
                    Scaled_dot_product_attention, Multihead_self_attention,Transform_block,\
                    Transformer_lm, Cross_entropy, AdamW, get_lr_cosine, gradient_clipping, \
                    data_loading, save_checkpoint, load_checkpoint

from BPE_tokenizer import train_bpe
import argparse, torch, os
import numpy as np
import wandb
def main():
    parser = argparse.ArgumentParser()
    '''
    模型超参数
    '''
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--vocabs_size", type = int, default = 10000)
    parser.add_argument("--context_length", type = int, default = 256)
    parser.add_argument("--d_model", type = int, default = 512)
    parser.add_argument("--num_layers", type = int, default = 512)
    parser.add_argument("--num_heads", type = int, default = 8)
    parser.add_argument("--d_ff", type = int, default = 2048)
    parser.add_argument("--theta", type = float, default = 10000)
    '''
    优化器超参数---余弦退火算法---梯度衰减
    lr, max_iters, warmup_iters, min_lr, max_norm
    '''
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--min_lr", type = float, default = 6e-5)
    parser.add_argument("--max_iters", type = int, default = 1000)
    parser.add_argument("--warmup_iters", type = int, default = 1000)
    parser.add_argument("--max_norm", type = float, default = 1.0)

    # ---路径与系统
    # required表示必须提供这个参数
    parser.add_argument("--train_data_path", type = str, required = True)
    parser.add_argument("--valid_data_path", type = str, required = True)
    parser.add_argument("--out_dir", type = str, default = "out")
    parser.add_argument("--device", type = str, default = "cuda" if torch.cuda.is_available() else "cpu")

    # --- WandB 设置 ---
    parser.add_argument("--wandb_project", type=str, default="cs336-pretraining")
    parser.add_argument("--run_name", type=str, default=None, help="WandB 实验名称")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok = True)

    # 1.加载数据--判断路径是否存在 + memap延迟加载数据
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"找不到训练数据在该路径上{args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"找不到验证数据在该路径上{args.valid_data_path}")
    # 加载大文件用np.memmap 内存-磁盘映射
    train_data = np.memmap(args.train_data_path, dtype = np.uint16, mode = 'r')
    valid_data = np.memmap(args.valid_data_path, dtype = np.uint16, mode = 'r')

    print(f"训练集大小：{len(train_data)}, 测试集大小：{len(valid_data)}")

    # 初始化模型
    model = Transformer_lm(
        vocab_size= args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff = args.d_ff,
        theta=args.theta
    ).to(args.device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay= 0.1)

    # 恢复断点
    start_iter = 0
    checkpoint_path = os.path.join(args.out_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        start_iter = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"从{start_iter}处开始恢复迭代")
    # 初始化 WandB 监控
    wandb.init(
        project=args.wandb_project,
        name=args.run_name, 
        config=args
    )
    # 主循环训练
    for it in range(start_iter, args.max_iters):
        lr = get_lr_cosine(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        model.train()
        x, y = data_loading(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = Cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()

        gradient_clipping(model.parameters(), args.max_norm)

        optimizer.step()

        # C. 验证与日志记录
        if it % 100 == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)
                print(f"Iter {it}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f}, lr {lr:.2e}")
                wandb.log({
                    "train/loss": loss.item(), 
                    "val/loss": v_loss.item(), 
                    "lr": lr, 
                    "iter": it + 1
                })

        # D. 保存检查点 (每 1000 步保存一次)
        if it % 1000 == 0 and it > 0:
            save_checkpoint(model, optimizer, it, ckpt_path)

    # 训练结束保存最终模型
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))
    wandb.finish()
    
    
if __name__ == "__main__":
    main()

# uv run cs336_basics\main_train.py --train_data_path data\TinyStoriesV2-GPT4-train.txt  --valid_data_path data\TinyStoriesV2-GPT4-valid.txt
