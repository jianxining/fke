# 文档功能点与关键词提取工具

从产品策划文档中提取结构化的“功能点 -> 关键词组”映射，适用于产品经理、研发、测试和运维人员对版本需求文档进行功能梳理、关键词归档和跨文档能力合并。

支持输入：

- `.docx`
- `.md` / `.markdown`

输出：

- `feature_keywords.csv`
- `feature_keywords.json`
- `sections.json`
- `errors.json`
- `merge_decisions.json`
- `capability_index.json`

## 核心思路

```text
文档
  -> 标题依赖图 + 正文编号
  -> 标题聚类
  -> DeepSeek 判断业务能力边界
  -> 传统 NLP 提取关键词
  -> DeepSeek 合并相似功能点
  -> CSV / JSON 输出
```

项目设计原则：

- 大模型负责业务语义判断。
- 传统 NLP 负责关键词统计。
- 正文编号负责结果可追溯。
- 聚类负责降低大模型调用成本。

## 1. 环境要求

推荐环境：

- Python 3.9+
- macOS / Linux / Windows
- DeepSeek API Key

## 2. 安装依赖

进入项目目录：

```bash
cd vivo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
cd vivo
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. 配置环境变量

在项目根目录创建 `.env`：

```bash
DEEPSEEK_API_KEY=你的DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseekV4.pro
```

注意：不要把真实 API Key 提交到 Git。

## 4. 准备输入文档

创建输入目录：

```bash
mkdir -p input_docs
```

把产品策划文档放进去：

```text
input_docs/
  需求文档A.docx
  需求文档B.md
```

DOCX 文档建议使用标准 Word 样式：

- Heading 1 / 标题 1
- Heading 2 / 标题 2
- Heading 3 / 标题 3
- Normal / 正文

## 5. 运行

```bash
python3 run.py --input ./input_docs --output ./outputs
```

常用参数：

```bash
python3 run.py \
  --input ./input_docs \
  --output ./outputs \
  --keyword-top-k 10 \
  --merge-threshold 0.45
```

参数说明：

- `--input`：输入文档目录。
- `--output`：输出目录。
- `--keyword-top-k`：每个功能点保留的关键词数量，默认 `10`。
- `--merge-threshold`：标题聚类相似度阈值，默认 `0.45`。
- `--cluster-backend`：标题聚类方式，支持 `tfidf` 和 `embedding`，默认 `tfidf`。
- `--embedding-model`：embedding 模型名称，默认 `BAAI/bge-m3`。

## 6. 查看结果

运行完成后会生成：

```text
outputs/
  feature_keywords.csv
  feature_keywords.json
  sections.json
  errors.json
  merge_decisions.json
  capability_index.json
```

推荐优先查看：

```bash
open outputs/feature_keywords.csv
```

或：

```bash
cat outputs/feature_keywords.json
```

CSV 示例：

```csv
功能点,关键词组,来源文档,相关标题,关联正文记号
手机号验证码登录能力,手机号|短信验证码|登录态,demo_requirement.docx,账号体系 > 登录管理 > 手机号登录,B003|B004
```

JSON 示例：

```json
{
  "手机号验证码登录能力": {
    "出现文档": ["demo_requirement.docx"],
    "合并关键词": ["手机号", "短信验证码", "登录态"],
    "相关标题": ["账号体系 > 登录管理 > 手机号登录"],
    "关联正文记号": ["B003", "B004"]
  }
}
```

## 7. 输出文件说明

| 文件 | 说明 |
| --- | --- |
| `feature_keywords.csv` | 面向人工查看的功能点-关键词映射表 |
| `feature_keywords.json` | 程序可复用的结构化结果 |
| `sections.json` | 文档解析后的标题依赖图和正文块 |
| `errors.json` | LLM 调用或结果校验失败记录 |
| `merge_decisions.json` | 功能点语义合并判断记录 |
| `capability_index.json` | 合并后的功能能力索引 |

## 8. 聚类模式说明

默认使用 TF-IDF 做标题聚类：

```bash
python3 run.py --input ./input_docs --output ./outputs
```

优点：

- 快。
- 不下载本地大模型。
- 部署简单。

如果文档很多、标题表述差异很大，可以启用本地 embedding 聚类：

```bash
pip install -r requirements-embedding.txt
python3 run.py \
  --input ./input_docs \
  --output ./outputs \
  --cluster-backend embedding
```

注意：embedding 模式会下载 `BAAI/bge-m3`，模型文件较大，首次运行较慢。

## 9. 常见问题

### 结果为空

先检查：

```bash
cat outputs/errors.json
```

如果看到：

```json
"error": "Connection error."
```

请检查：

- `.env` 中 `DEEPSEEK_API_KEY` 是否正确。
- `DEEPSEEK_BASE_URL` 是否正确。
- 当前网络是否可以访问 DeepSeek API。

### API Key 未配置

如果看到：

```text
DEEPSEEK_API_KEY is required
```

说明 `.env` 不存在，或没有填写真实 API Key。

### DOCX 没有解析出标题

查看：

```bash
cat outputs/sections.json
```

确认 Word 文档是否真的使用了 Heading 样式，而不是手动加粗或放大字体。

### urllib3 OpenSSL 警告

macOS 自带 Python 可能出现：

```text
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+
```

这通常不是业务错误。可以先忽略；如果网络调用异常，可以降级 urllib3：

```bash
pip install "urllib3<2"
```

## 10. 运行测试

开发者可以运行：

```bash
python3 -m pytest
```

当前测试覆盖：

- DOCX 解析。
- Markdown 解析。
- 标题依赖图构建。
- 关键词提取。
- 功能点合并。
- LLM 返回格式兼容。

## 11. 项目目录

```text
feature_keyword_extractor/
  agents/
    feature_agent.py       # 大模型功能点归纳
    merge_agent.py         # 大模型功能点合并判断
    schemas.py             # LLM 输出结构校验

  parsers/
    docx_parser.py         # DOCX 解析
    markdown_parser.py     # Markdown 解析
    base.py                # 标题依赖图构建

  nlp/
    keyword_extractor.py   # 传统 NLP 关键词提取

  cluster.py               # 标题聚类
  merge.py                 # 功能点合并
  pipeline.py              # 主流程
  exporter.py              # CSV / JSON 导出

run.py                     # CLI 入口
requirements.txt           # 基础依赖
requirements-embedding.txt # 可选 embedding 依赖
```

## 12. 开发注意事项

- 不要把真实 API Key 写入源码或提交到 Git。
- 关键词提取禁止调用大模型。
- 功能点归纳和功能点合并必须调用大模型。
- 标题层级只作为依赖关系和上下文，不作为最终功能点划分规则。
- 如果 LLM 返回引用了不存在的正文 ID，结果会进入 `errors.json`，不会静默污染最终结果。
