# SteganographyLibrary: 文本隐写算法库

`SteganographyLibrary` 是一个集成了多种经典文本隐写算法的Python库。本项目旨在为文本隐写技术的研究人员和开发者提供一个统一、灵活且易于扩展的实验框架。

本库实现了多种基于生成式和编辑式方法的文本隐写模型，支持使用不同的预训练语言模型（如BERT、GPT-2等）作为后端，从而探索现代深度学习技术在信息隐藏领域的应用。

## 功能特性

  - **统一的算法接口**：所有隐写模型均遵循统一的 `encrypt` 和 `decrypt` 接口，方便调用和替换。
  - **模块化设计**：每种隐写算法在独立的模块中实现，结构清晰，易于维护和扩展。
  - **支持多种模型**：集成了多种主流的文本隐写算法，包括：
      - **编辑式隐写 (Edit-based)**:
          - [《Frustratingly Easy Edit-based Linguistic Steganography with a Masked Language Model》](https://aclanthology.org/2021.naacl-main.436/)
      - **生成式隐写 (Generative)**:
          - [《Generating Steganographic Text with LSTMs》](https://aclanthology.org/P17-3017/) (基于Bins)
          - [《RNN-Stega: Linguistic Steganography Based on Recurrent Neural Networks》](https://www.google.com/search?q=https://ieeexplore.ieee.org/document/8489781) (基于哈夫曼编码)
          - [《Neural Linguistic Steganography》](https://arxiv.org/abs/1909.09294) (基于算术编码)
          - [《Discop: Provably Secure Steganography in Practice Based on “Distribution Copies”》](https://www.google.com/search?q=https://www.usenix.org/conference/usenixsecurity22/presentation/huang-haoyu) (基于分布式副本)
  - **灵活的语言模型后端**：支持通过Hugging Face `transformers`库加载不同的预训练语言模型（如BERT, GPT-2, RoBERTa等）作为隐写算法的基础。
  - **命令行工具**：提供一个功能强大的命令行接口 (`main.py`)，方便用户快速进行加密和解密操作，并调整各项参数。

## 项目结构

```bash
SteganographyLibrary
├─ cover.txt            # 示例载体文本
├─ LICENSE              # 项目许可证
├─ main.py              # 主程序入口和命令行工具
├─ methods/             # 隐写算法实现目录
│  ├─ edit_stego.py      # 模型一：编辑式隐写
│  ├─ lstm_stego.py      # 模型二：基于Bins的生成式隐写
│  ├─ huffman_stega.py   # 模型三：基于哈夫曼编码的生成式隐写
│  ├─ neural_stego.py    # 模型四：基于算术编码的生成式隐写
│  └─ discop_stego.py    # 模型五：基于分布式副本的生成式隐写
├─ payload.txt          # 示例秘密信息
├─ stego.txt            # 默认隐写文本输出文件
├─ utils.py             # 工具函数，如模型加载等
└─ requirements.txt     # 项目依赖
```

## 安装

1.  **克隆本仓库**:

    ```bash
    git clone https://github.com/your-username/SteganographyLibrary.git
    cd SteganographyLibrary
    ```

2.  **创建虚拟环境 (推荐)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # on Windows, use `venv\Scripts\activate`
    ```

3.  **安装依赖**:
    本项目依赖于PyTorch和Hugging Face Transformers。请先根据您的硬件情况（CPU/GPU）安装PyTorch。

      * 访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取适合您系统的安装命令。

      * 然后安装其他依赖：

    <!-- end list -->

    ```bash
    pip install -r requirements.txt
    ```

4.  **下载NLTK数据 (首次使用)**:
    部分模型可能需要NLTK的数据（如停用词表）。请在Python解释器中运行：

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## 使用说明

本项目提供了一个强大的命令行工具 `main.py` 用于执行隐写操作。

### 基本参数

  * `--action`: 选择操作，`encrypt` (加密) 或 `decrypt` (解密)。默认为 `encrypt`。
  * `--method`: 选择隐写方法，可选 `edit`, `lstm`, `rnn`, `neural`, `discop`。
  * `--model`: 指定使用的Hugging Face预训练模型名称（如 `bert-base-cased`, `gpt2`）。
  * `--cover`: 载体文本文件路径（加密时使用）。
  * `--payload`: 秘密信息文件路径（加密时使用）。
  * `--stego`: 隐写文本的输入/输出文件路径。

### 加密示例

**使用编辑式方法 (`edit`) 加密：**

```bash
python main.py \
    --action encrypt \
    --method edit \
    --model bert-base-cased \
    --cover cover.txt \
    --payload payload.txt \
    --stego stego_edit_out.txt \
    --mask_interval 3 \
    --score_threshold 0.01
```

**使用生成式方法 (`rnn` / 哈夫曼编码) 加密：**

对于生成式方法，`--cover` 参数通常用作生成文本的初始提示(prompt)。

```bash
python main.py \
    --action encrypt \
    --method rnn \
    --model gpt2 \
    --cover "The weather is" \
    --payload payload.txt \
    --stego stego_rnn_out.txt \
    --candidate_pool_size 50
```

### 解密示例

**解密由编辑式方法生成的文本：**

```bash
python main.py \
    --action decrypt \
    --method edit \
    --model bert-base-cased \
    --stego stego_edit_out.txt \
    --mask_interval 3 \
    --score_threshold 0.01
```

**解密由生成式方法 (`rnn`) 生成的文本：**

```bash
python main.py \
    --action decrypt \
    --method rnn \
    --model gpt2 \
    --stego stego_rnn_out.txt \
    --candidate_pool_size 50
```

**注意**: 解密时必须使用与加密时完全相同的参数（如 `model`, `mask_interval`, `candidate_pool_size` 等），否则无法正确还原秘密信息。

## 贡献

我们欢迎任何形式的贡献！如果您有兴趣改进现有算法、添加新模型或修复bug，请随时提交Pull Request或创建Issue。

1.  Fork本仓库
2.  创建您的新分支 (`git checkout -b feature/AmazingFeature`)
3.  提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4.  将您的分支推送到远程仓库 (`git push origin feature/AmazingFeature`)
5.  提交一个Pull Request

## 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](https://www.google.com/search?q=LICENSE) 文件。

## 致谢

本项目的实现参考了以下学术论文，感谢原作者的开创性工作：

  - Ueoka et al., "Frustratingly Easy Edit-based Linguistic Steganography with a Masked Language Model"
  - Fang et al., "Generating Steganographic Text with LSTMs"
  - Yang et al., "RNN-Stega: Linguistic Steganography Based on Recurrent Neural Networks"
  - Ziegler et al., "Neural Linguistic Steganography"
  - Huang et al., "Discop: Provably Secure Steganography in Practice Based on 'Distribution Copies'"
  - Hugging Face团队提供的 `transformers` 库。