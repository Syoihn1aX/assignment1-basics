# CS336 Assignment 1：从零实现 Transformer 语言模型

个人仓库：<https://github.com/Syoihn1aX/assignment1-basics>

本项目基于 Stanford CS336 Assignment 1 框架，完成字节级 BPE 分词器、Transformer 语言模型核心模块以及训练所需的基础组件。官方作业文档见 [`cs336_assignment1_basics.pdf`](./cs336_assignment1_basics.pdf)。

## 当前版本信息

- 开发分支：`lab5-work`
- 最近一个已提交的 Transformer 检查点：`109f661`
- GitHub 仓库：<https://github.com/Syoihn1aX/assignment1-basics>
- 说明：README 编写时仍有部分训练组件处于开发中；完成全部任务后，应将最终 commit ID 更新到本节。

## 已新增和修改的文件

| 文件 | 类型 | 主要内容 | 当前状态 |
| --- | --- | --- | --- |
| `cs336_basics/tokenizer.py` | 新增 | 字节级 BPE 训练、特殊 token 切分、Tokenizer 编码、解码和流式编码 | 已完成并通过 tokenizer 相关测试 |
| `cs336_basics/nn.py` | 新增/修改 | Linear、Embedding、SiLU、RMSNorm、SwiGLU、scaled dot-product attention、RoPE、softmax、cross-entropy | 核心模型模块已完成；训练工具仍在补充测试 |
| `cs336_basics/model.py` | 新增 | 普通多头自注意力、带 RoPE 的多头自注意力、Transformer Block、Transformer LM | 已完成并通过模型测试 |
| `cs336_basics/optimizer.py` | 新增 | 梯度裁剪实现 | 已新增；AdamW 和学习率调度仍需完成 |
| `tests/adapters.py` | 修改 | 将官方测试适配到个人实现，连接 tokenizer、神经网络模块、模型和部分优化器功能 | 已连接已完成模块，仍有训练基础设施适配器待完成 |

## 已实现功能

### BPE Tokenizer

- 以 256 个 UTF-8 字节作为初始词表；
- 使用 GPT-2 风格正则进行预分词；
- 统计相邻 token pair 并进行 BPE merge；
- 对相同频率的 pair 进行确定性 tie-breaking；
- 特殊 token 作为硬边界，不参与普通 merge；
- 支持 `encode`、`decode` 和 `encode_iterable`；
- 支持 Unicode 文本和非法 UTF-8 的替换解码。

### Transformer 模型

- Linear 和 Embedding；
- RMSNorm；
- SiLU 和 SwiGLU；
- 数值稳定的 scaled dot-product attention；
- causal mask；
- Rotary Positional Embedding（RoPE）；
- Multi-Head Self-Attention；
- pre-norm Transformer Block；
- Transformer Language Model 输出 logits。

## 当前验证情况

目前已经验证通过的部分包括：

- BPE tokenizer 训练测试；
- tokenizer 编码、解码和流式编码测试；
- Linear 和 Embedding；
- RMSNorm、SiLU 和 SwiGLU；
- scaled dot-product attention；
- RoPE；
- 普通多头自注意力；
- 带 RoPE 的多头自注意力；
- Transformer Block；
- Transformer Language Model。

训练相关组件完成后，应继续运行完整测试套件，并将最终测试结果补充到这里。

## 环境配置

本项目使用 `uv` 管理 Python 环境。项目要求 Python 3.12 或 3.13。

```bash
uv sync
```

如果本机 `uv` 默认缓存目录没有写权限，可以指定临时缓存目录：

```bash
UV_CACHE_DIR=/tmp/cs336-uv-cache uv sync
```

## 运行测试

运行全部测试：

```bash
UV_CACHE_DIR=/tmp/cs336-uv-cache uv run pytest -q
```

按模块运行测试：

```bash
UV_CACHE_DIR=/tmp/cs336-uv-cache uv run pytest -q tests/test_tokenizer.py
UV_CACHE_DIR=/tmp/cs336-uv-cache uv run pytest -q tests/test_train_bpe.py
UV_CACHE_DIR=/tmp/cs336-uv-cache uv run pytest -q tests/test_model.py
UV_CACHE_DIR=/tmp/cs336-uv-cache uv run pytest -q tests/test_nn_utils.py
UV_CACHE_DIR=/tmp/cs336-uv-cache uv run pytest -q tests/test_optimizer.py
UV_CACHE_DIR=/tmp/cs336-uv-cache uv run pytest -q tests/test_data.py
UV_CACHE_DIR=/tmp/cs336-uv-cache uv run pytest -q tests/test_serialization.py
```

## 数据集

TinyStories 和 OpenWebText 数据放在 `data/` 目录中。数据集、编码后的 token 文件、模型 checkpoint 等大文件不应直接提交到 Git 仓库。

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## 待完成工作

当前还需要完成并验证：

- gradient clipping 的完整测试；
- AdamW；
- cosine learning-rate schedule 和 warmup；
- 随机 batch 采样；
- checkpoint 保存和恢复；
- 完整训练循环；
- TinyStories 训练、验证和文本生成；
- 训练曲线、资源统计和实验报告。

OpenWebText 训练和大规模消融实验计算量较高，可以根据课程要求选择是否完成。

## Git 提交记录

建议按功能阶段提交：

```bash
git add cs336_basics tests/adapters.py
git commit -m "Complete Transformer language model"
```

查看当前提交：

```bash
git log --oneline --decorate -5
```

完成全部实验后，将最终 commit 推送到个人仓库，并把最终 commit ID 写入本 README 和实验报告。
