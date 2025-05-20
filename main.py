import argparse
from methods.edit_stego import EditStego
from methods.lstm_stego import LSTMStego
from methods.rnn_stega import RNNStega
from methods.neural_stego import NeuralStego
from methods.discop_stego import DiscopStego

def main():
    parser = argparse.ArgumentParser(description='文本隐写加密/解密工具')
    parser.add_argument('--action', default='encrypt', choices=['encrypt','decrypt'], help='选择操作：encrypt 加密；decrypt 解密')
    parser.add_argument('--method', default='edit', choices=['edit','lstm','rnn','neural','discop'], help='隐写方法')
    parser.add_argument('--model', default='bert-base-cased', help='HuggingFace 模型名')
    parser.add_argument('--cover', default='cover.txt', help='载体文本文件路径 (encrypt 时必填)')
    parser.add_argument('--stego', default='stego.txt', help='隐写文本文件路径 (decrypt 时必填)')
    parser.add_argument('--payload', default='payload.txt', help='待加密消息文件路径 (encrypt 时必填)')
    
    # edit_stego 特有参数
    parser.add_argument('--mask_interval', type=int, default=2, help='掩码间隔 (仅 edit_stego 有效)')
    parser.add_argument('--score_threshold', type=float, default=0.01, help='分数阈值 (仅 edit_stego 有效)')
    # lstm_stego 特有参数

    args = parser.parse_args()

    # 实例化对应方法类
    cls_map = {
        'edit': EditStego,
        'lstm': LSTMStego,
        'rnn': RNNStega,
        'neural': NeuralStego,
        'discop': DiscopStego,
    }
    if args.method == 'edit':
        stego = cls_map[args.method](
            model_name=args.model,
            mask_interval=args.mask_interval,
            score_threshold=args.score_threshold
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
        # 保存隐写文本
        with open(args.stego, 'w', encoding='utf-8') as f:
            f.write(stego_text)
        secret = stego.decrypt(stego_text)
        print("[DEBUG] Decrypted secret:", secret.decode('utf-8', errors='ignore'))

    else:  # decrypt
        with open(args.stego, 'r', encoding='utf-8') as f:
            stego_text = f.read()
        secret = stego.decrypt(stego_text)
        print(secret.decode('utf-8', errors='ignore'))

if __name__ == '__main__':
    main()
