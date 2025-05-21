import argparse
from methods.edit_stego import EditStego
from methods.lstm_stego import LSTMStego
from methods.huffman_stega import HuffmanStega
from methods.neural_stego import NeuralStego
from methods.discop_stego import DiscopStego

def main():
    parser = argparse.ArgumentParser(description='文本隐写加密/解密工具')
    parser.add_argument('--action', default='encrypt', choices=['encrypt','decrypt'], help='选择操作：encrypt 加密；decrypt 解密')
    parser.add_argument('--method', default='neural', choices=['edit','lstm','huffman','neural','discop'], help='隐写方法')
    parser.add_argument('--model', default='gpt2', help='HuggingFace 模型名')
    parser.add_argument('--cover', default='cover.txt', help='载体文本文件路径 (encrypt 时必填)')
    parser.add_argument('--stego', default='stego.txt', help='隐写文本文件路径 (decrypt 时必填)')
    parser.add_argument('--payload', default='payload.txt', help='待加密消息文件路径 (encrypt 时必填)')
    parser.add_argument('--device', default='cpu', choices=['cuda','cpu'], help='设备')
    # edit_stego 特有参数
    parser.add_argument('--mask_interval', type=int, default=2, help='掩码间隔 (仅 edit_stego 有效)')
    parser.add_argument('--score_threshold', type=float, default=0.1, help='分数阈值 (仅 edit_stego 有效)')
    # lstm_stego 特有参数
    parser.add_argument('--bit_block_size', type=int, default=2, help='比特块大小 (仅 lstm_stego 有效)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (仅 lstm_stego 有效)')
    # rnn_stega 特有参数
    parser.add_argument('--bits_per_word', type=int, default=2, help='每个字符的比特数 (仅 rnn_stega 有效)')
    parser.add_argument('--encoding_method', default='vlc', choices=['flc','vlc'], help='编码方法 (仅 rnn_stega 有效)')
    # neural_stego 特有参数
    parser.add_argument('--top_k', type=int, default=1024, help='top-k 采样 (仅 neural_stego 有效)')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度 (仅 neural_stego 有效)')
    args = parser.parse_args()
    # 实例化对应方法类
    cls_map = {
        'edit': EditStego,
        'lstm': LSTMStego,
        'huffman': HuffmanStega,
        'neural': NeuralStego,
        'discop': DiscopStego,
    }

    if args.method == 'edit':
        stego = cls_map[args.method](
            model_name=args.model,
            mask_interval=args.mask_interval,
            score_threshold=args.score_threshold
        )
    elif args.method == 'lstm':
        stego = cls_map[args.method](
            model_name=args.model,
            bit_block_size=args.bit_block_size,
            seed=args.seed
        )
    elif args.method == 'huffman':
        stego = cls_map[args.method](
            model_name=args.model,
            bits_per_word=args.bits_per_word,
            encoding_method=args.encoding_method
        )
    elif args.method == 'neural':
        stego = cls_map[args.method](
            model_name=args.model,
            topk=args.top_k,
            temp=args.temperature
        )
        
    if args.action == 'encrypt':
        # 读取载体文本和秘密消息
        with open(args.cover, 'r', encoding='utf-8') as f:
            cover_text = f.read()
        with open(args.payload, 'rb') as f:
            data = f.read()
        # 隐写
        stego_text = stego.encrypt(cover_text, data)
        print(stego_text)
        print("----")
        # 保存隐写文本
        with open(args.stego, 'w', encoding='utf-8') as f:
            f.write(stego_text)
        secret = stego.decrypt(cover_text, stego_text)
        print("[DEBUG] Decrypted secret:", secret.decode('utf-8', errors='ignore'))

    else:  # decrypt
        with open(args.cover, 'r', encoding='utf-8') as f:
            cover_text = f.read()
        with open(args.stego, 'r', encoding='utf-8') as f:
            stego_text = f.read()
        secret = stego.decrypt(cover_text, stego_text)
        print(secret.decode('utf-8', errors='ignore'))

if __name__ == '__main__':
    main()
