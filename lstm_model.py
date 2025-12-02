import torch
import torch.nn as nn
import numpy as np
from data_loader import prepare_data
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_and_predict(symbol="RB2410", seq_len=60, epochs=50):
    import os
    from data_loader import get_futures_data
    model_path = f"models/{symbol}_lstm.pth"
    
    df = get_futures_data(symbol)
    train_X, train_y, scaler, all_X, real_prices = prepare_data(df, seq_len)
    
    # 转 torch
    train_X = torch.from_numpy(train_X).float()
    train_y = torch.from_numpy(train_y).float().unsqueeze(1)
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练（如果已有模型直接加载）
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    else:
        print(f"正在为 {symbol} 训练模型...")
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(train_X)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")
    
    # 预测未来 30 天
    model.eval()
    with torch.no_grad():
        future_pred = []
        input_seq = all_X[-1:].copy()  # 最后一条序列
        
        for _ in range(30):
            input_tensor = torch.from_numpy(input_seq).float()
            pred = model(input_tensor)
            future_pred.append(pred.numpy()[0, 0])
            # 滑动窗口
            new_seq = np.append(input_seq[0, 1:, :], [[pred.numpy()[0, 0]]], axis=0)
            input_seq = new_seq.reshape(1, seq_len, 1)
    
    future_pred = scaler.inverse_transform(np.array(future_pred).reshape(-1, 1))
    future_pred = future_pred.flatten().tolist()
    
    # 历史真实价格 + 预测价格
    hist_dates = df['date'].dt.strftime('%Y-%m-%d').tolist()[-120:]  # 最近120天
    hist_prices = real_prices[-120:].flatten().tolist()
    
    # 生成未来日期
    last_date = df['date'].iloc[-1]
    future_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(30)]
    
    return {
    "symbol": symbol,
    "hist_dates": hist_dates,
    "hist_prices": hist_prices,
    "future_dates": future_dates,
    "future_prices": future_pred,
    
    # ←←←← 新增：实时行情（用于卡片显示）
    "current_price": float(df['close'].iloc[-1]),        # 最新价
    "change": float(df['close'].iloc[-1] - df['close'].iloc[-2]),  # 涨跌额
    "change_pct": float((df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100),  # 涨跌幅%
    "volume": int(df['volume'].iloc[-1]) if 'volume' in df.columns else 0,       # 成交量
    "open_interest": int(df.get('position', df.get('hold', pd.Series([0]*len(df)))).iloc[-1]),  # 持仓量
}