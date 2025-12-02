

# 基于 LSTM 的期货价格智能预测系统（本地部署版）

**实时行情 + AI 预测 + 一键操作 + 美观大方**  
一个完全本地运行、零云服务依赖的期货分析 Web 系统，使用 PyTorch LSTM 模型预测未来 30 天价格走势。

![演示截图](https://img.shields.io/badge/效果-专业级-brightgreen) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-最新-orange) ![AKShare](https://img.shields.io/badge/AKShare-实时数据-green)

![界面预览](docs/preview.jpg)
![alt text](preview.jpg)
> （实际效果比图片更炫！左侧实时行情卡片，右侧 LSTM 预测曲线）

## 功能亮点

- 实时获取新浪财经主力连续合约行情（AKShare）
- 支持 10+ 热门品种一键预测（黄金、原油、螺纹钢、豆粕等）
- PyTorch LSTM 模型自动训练 & 保存（首次 15~40 秒，后续秒开）
- 训练好的模型永久缓存（`models/品种_lstm.pth`），永不重复训练
- 响应式美观界面（Bootstrap 5 + 渐变卡片 + 悬停动画）
- 实时显示最新价、涨跌幅、成交量、持仓量
- 完全本地运行，无需联网注册、无需付费接口

## 支持品种（主力连续合约，永不过期）

| 品种   | 代码   | 备注           |
|--------|--------|----------------|
| 黄金   | AU0    | 最稳定         |
| 白银   | AG0    |                |
| 沪铜   | CU0    |                |
| 螺纹钢 | RB0    |                |
| 原油   | SC0    |                |
| 沥青   | BU0    |                |
| 豆粕   | M0     |                |
| PTA    | TA0    |                |
| 甲醇   | MA0    |                |
| 沪镍   | NI0    |                |

> 也可手动输入其他合约，如 RB2501、AP2410 等

## 快速开始（3 步搞定）

### 1. 克隆/下载项目
```bash
git clone https://github.com/HanJun27/-LSTM-
cd futures-lstm-predictor

### 2. 安装依赖（推荐使用虚拟环境）
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 3. 一键启动
```bash
uvicorn main.py --reload
```

浏览器打开 → http://127.0.0.1:8000

点击左侧任意品种（如 **黄金 AU0**） → 等待首次训练完成 → 即可看到未来 30 天预测曲线！

## 项目结构
```
futures-lstm-predictor/
├── main.py              # FastAPI 主程序
├── lstm_model.py        # LSTM 模型训练与预测逻辑
├── data_loader.py       # AKShare 数据获取与清洗（已完美适配中英文列名）
├── templates/
│   └── index.html       # 超美观前端页面（卡片 + 实时行情 + Chart.js）
├── models/              # 自动生成，保存训练好的 .pth 模型
├── static/              # 可选静态资源
└── requirements.txt
```

## 常见问题

| 问题                                 | 解决方案                                |
|--------------------------------------|-----------------------------------------|
| 第一次预测要等 15~40 秒？            | 正常！正在训练模型，训练完永久保存      |
| 报错 ConnectionResetError / 10054？  | 新浪限流，等待 1 分钟再试，或先试 AU0   |
| 提示“获取数据失败”？                 | 运行 `pip install akshare --upgrade`    |
| 想加更多品种？                       | 在 `index.html` 里复制一个卡片即可      |

## 技术栈

- **后端**：FastAPI + Uvicorn
- **AI**：PyTorch + LSTM
- **数据源**：AKShare（新浪财经实时数据）
- **前端**：Bootstrap 5 + Chart.js + Jinja2
- **部署**：纯本地运行，支持 Windows / macOS / Linux

## 致谢

- [AKShare](https://github.com/akfamily/akshare) - 强大的开源财经接口
- [FastAPI](https://fastapi.tiangolo.com/) - 现代、快速的 Web 框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

## License

MIT © 2025 HanJun 

