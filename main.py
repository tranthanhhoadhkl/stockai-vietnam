"""
Hệ thống Web Hỗ Trợ Đầu Tư Chứng Khoán - FastAPI Backend
Hybrid AI System: Random Forest (bank stocks) + Technical Analysis (others)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Hệ thống Hỗ trợ Ra quyết định Đầu tư Chứng khoán",
    description="Nền tảng phân tích tài chính thông minh dựa trên mô hình học máy và hệ chuyên gia quy tắc kỹ thuật",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BANK_TICKERS = ["BID", "VCB", "CTG", "MBB", "TCB", "ACB", "VPB", "HDB", "STB", "EIB"]
FEATURED_TICKERS = ["BID", "VCB", "CTG", "MBB", "VNM", "VIC", "HPG", "MSN", "TCB", "FPT", "VHM", "ACB"]

GLOSSARY = {
    "RSI": {
        "name": "RSI – Relative Strength Index",
        "short": "Chỉ số đo lường tốc độ và biên độ biến động giá",
        "description": "RSI (Relative Strength Index) là chỉ báo dao động dao động từ 0–100. RSI > 70 cho thấy cổ phiếu đang trong vùng mua quá mức (overbought) – có thể sắp giảm. RSI < 30 cho thấy cổ phiếu đang bán quá mức (oversold) – có thể sắp tăng. Vùng 40–60 là trung tính.",
        "formula": "RSI = 100 – (100 / (1 + RS)), trong đó RS = Trung bình tăng / Trung bình giảm",
        "use_case": "Phát hiện điểm đảo chiều xu hướng ngắn hạn",
        "category": "Momentum"
    },
    "MACD": {
        "name": "MACD – Moving Average Convergence Divergence",
        "short": "Chỉ báo xu hướng dựa trên hai đường trung bình động",
        "description": "MACD = EMA(12) - EMA(26). Khi đường MACD cắt Signal Line từ dưới lên → tín hiệu MUA. Khi cắt từ trên xuống → tín hiệu BÁN. Histogram dương = xu hướng tăng, âm = xu hướng giảm.",
        "formula": "MACD = EMA(12) – EMA(26) | Signal = EMA(9) của MACD",
        "use_case": "Xác nhận xu hướng và điểm vào/ra thị trường",
        "category": "Trend"
    },
    "MA": {
        "name": "Moving Average – Đường Trung Bình Động",
        "short": "Làm mượt biến động giá để nhận diện xu hướng",
        "description": "MA20 là trung bình giá 20 ngày, MA50 là 50 ngày. Khi giá vượt MA từ dưới lên → xu hướng tăng. Khi MA20 cắt MA50 từ dưới (Golden Cross) → tín hiệu tăng mạnh. Khi MA20 cắt MA50 từ trên xuống (Death Cross) → tín hiệu giảm.",
        "formula": "MA(n) = Tổng giá đóng cửa n ngày / n",
        "use_case": "Xác định xu hướng dài hạn và ngắn hạn",
        "category": "Trend"
    },
    "Bollinger Bands": {
        "name": "Bollinger Bands – Dải Bollinger",
        "short": "Dải biến động giá dựa trên độ lệch chuẩn",
        "description": "Gồm 3 đường: BB_upper = MA20 + 2σ, BB_middle = MA20, BB_lower = MA20 – 2σ. Giá chạm BB_upper → có thể giảm. Giá chạm BB_lower → có thể tăng. Dải hẹp → sắp có biến động mạnh (Bollinger Squeeze).",
        "formula": "Upper = MA(20) + 2×σ | Lower = MA(20) – 2×σ",
        "use_case": "Đo lường độ biến động và nhận diện breakout",
        "category": "Volatility"
    },
    "Random Forest": {
        "name": "Random Forest – Rừng Ngẫu Nhiên",
        "short": "Mô hình học máy tổng hợp nhiều cây quyết định",
        "description": "Random Forest là thuật toán ensemble kết hợp hàng trăm Decision Tree. Mỗi cây được huấn luyện trên tập dữ liệu ngẫu nhiên khác nhau. Kết quả cuối là bỏ phiếu đa số. Giúp giảm overfitting và tăng độ chính xác so với một cây đơn lẻ.",
        "formula": "Prediction = Mode(Tree₁, Tree₂, ..., Treeₙ)",
        "use_case": "Phân loại tín hiệu MUA/BÁN dựa trên các chỉ báo kỹ thuật",
        "category": "Machine Learning"
    },
    "LSTM": {
        "name": "LSTM – Long Short-Term Memory",
        "short": "Mạng nơ-ron nhớ dài hạn để dự báo chuỗi thời gian",
        "description": "LSTM là loại Recurrent Neural Network (RNN) có khả năng ghi nhớ thông tin dài hạn. Gồm 3 cổng: Forget Gate (quên thông tin cũ), Input Gate (học thông tin mới), Output Gate (xuất kết quả). Phù hợp với dữ liệu chuỗi thời gian như giá cổ phiếu.",
        "formula": "hₜ = oₜ ⊙ tanh(Cₜ), Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ",
        "use_case": "Dự báo giá trong tương lai, nhận diện pattern phức tạp",
        "category": "Deep Learning"
    },
    "Hybrid ": {
        "name": "Hybrid System – Hệ thống Kết hợp",
        "short": "Kết hợp học máy và phân tích kỹ thuật truyền thống",
        "description": "Hệ thống này sử dụng chiến lược hai lớp: Với cổ phiếu ngân hàng (BID, VCB, CTG, MBB...) – áp dụng mô hình Random Forest + LSTM đã được huấn luyện trên dữ liệu lịch sử 5 năm. Với các cổ phiếu khác – sử dụng phân tích kỹ thuật: RSI, MACD, MA, Bollinger Bands.",
        "formula": "if bank_stock → ML Model | else → Technical Analysis",
        "use_case": "Tối ưu hoá tín hiệu theo đặc thù từng nhóm ngành",
        "category": "System Design"
    },
    "EMA": {
        "name": "EMA – Exponential Moving Average (Trung Bình Động Mũ)",
        "short": "Đường trung bình có trọng số cao hơn cho giá gần đây",
        "description": "EMA khác với MA (SMA) ở chỗ nó gán trọng số cao hơn cho các phiên gần đây, phản ứng nhạy hơn với biến động giá mới. EMA12 (ngắn hạn) và EMA26 (dài hạn) là hai thành phần chính của MACD. Khi EMA12 cắt EMA26 từ dưới lên → xu hướng tăng bắt đầu hình thành.",
        "formula": "EMA(t) = Giá(t) × k + EMA(t-1) × (1 - k), trong đó k = 2 / (n + 1)",
        "use_case": "Nền tảng tính MACD; nhận diện xu hướng nhanh hơn SMA",
        "category": "Trend"
    },
    "Sharpe Ratio": {
        "name": "Sharpe Ratio – Tỷ lệ Lợi nhuận Rủi ro",
        "short": "Đo lường lợi nhuận thu được trên mỗi đơn vị rủi ro chịu đựng",
        "description": "Sharpe Ratio là thước đo hiệu quả đầu tư sau khi điều chỉnh rủi ro, do William Sharpe phát minh năm 1966. Sharpe > 1: tốt, > 2: rất tốt, < 0: kém hơn giữ tiền mặt. Trong hệ thống này, tỷ suất phi rủi ro (Rf) = 0 để đơn giản hoá. Dùng để so sánh hiệu quả giữa các cổ phiếu hoặc chiến lược.",
        "formula": "Sharpe = (Rp - Rf) / σp, trong đó Rp = lợi nhuận danh mục, Rf = lãi suất phi rủi ro, σp = độ lệch chuẩn lợi nhuận",
        "use_case": "So sánh hiệu quả đầu tư giữa các cổ phiếu có mức rủi ro khác nhau",
        "category": "Risk"
    },
    "Max Drawdown": {
        "name": "Max Drawdown – Mức Sụt Giảm Tối Đa",
        "short": "Mức thua lỗ lớn nhất tính từ đỉnh cao nhất đến đáy thấp nhất",
        "description": "Max Drawdown (MDD) đo lường kịch bản tệ nhất mà nhà đầu tư có thể gặp nếu mua ở đỉnh và bán ở đáy trong khoảng thời gian nghiên cứu. MDD = -10% nghĩa là giá từng giảm 10% so với đỉnh. Đây là chỉ số rủi ro downside quan trọng hơn độ lệch chuẩn vì nó phản ánh rủi ro thực tế của nhà đầu tư.",
        "formula": "MDD = (Đáy - Đỉnh) / Đỉnh × 100%",
        "use_case": "Đánh giá rủi ro thua lỗ tối đa khi đầu tư vào một cổ phiếu",
        "category": "Risk"
    },
    "Volatility": {
        "name": "Volatility – Độ Biến Động (Annualized)",
        "short": "Mức độ dao động giá hàng năm, thước đo rủi ro tổng thể",
        "description": "Volatility hàng năm được tính bằng cách nhân độ lệch chuẩn lợi nhuận ngày với căn bậc hai của 252 (số phiên giao dịch/năm). Volatility cao (>40%) = rủi ro lớn nhưng cơ hội lợi nhuận cao. Volatility thấp (<15%) = ổn định nhưng tăng trưởng chậm. Volatility 10 ngày (Volatility_10) trong hệ thống này đo độ bất ổn ngắn hạn.",
        "formula": "σ_annual = σ_daily × √252, trong đó σ_daily = độ lệch chuẩn lợi nhuận ngày",
        "use_case": "Đo mức độ rủi ro tổng thể; dùng làm feature trong mô hình Random Forest",
        "category": "Risk"
    },
    "Volume MA": {
        "name": "Volume MA(10) – Trung Bình Khối Lượng 10 Ngày",
        "short": "Đường trung bình khối lượng giao dịch, phát hiện phiên bất thường",
        "description": "Volume MA(10) là trung bình cộng khối lượng giao dịch 10 phiên liên tiếp. Nó giúp lọc nhiễu và xác định liệu khối lượng phiên hiện tại có cao bất thường hay không. Khi Volume > 2× Vol MA(10): phiên bất thường (đột biến) – thường xảy ra khi có tin tức, thông báo kết quả kinh doanh, hoặc tổ chức lớn mua/bán. Volume đi cùng tín hiệu kỹ thuật sẽ xác nhận độ tin cậy.",
        "formula": "Vol_MA_10[i] = (V[i] + V[i-1] + ... + V[i-9]) / 10",
        "use_case": "Xác nhận tín hiệu kỹ thuật; phát hiện phiên giao dịch bất thường",
        "category": "Volume"
    },
    "Momentum": {
        "name": "Momentum – Động Lượng Giá",
        "short": "Đo tốc độ thay đổi giá trong một khoảng thời gian",
        "description": "Momentum_5 trong hệ thống này đo sự chênh lệch giá giữa phiên hiện tại và 5 phiên trước. Momentum dương → giá đang tăng tốc. Momentum âm → giá đang mất đà. Momentum là một trong những hiệu ứng được nghiên cứu nhiều nhất trong tài chính hành vi: cổ phiếu tăng mạnh có xu hướng tiếp tục tăng trong ngắn hạn (hiệu ứng momentum).",
        "formula": "Momentum_5 = Close[t] - Close[t-5]",
        "use_case": "Feature quan trọng trong Random Forest; nhận diện đà tăng/giảm ngắn hạn",
        "category": "Momentum"
    },
    "Backtest": {
        "name": "Backtest – Kiểm Định Chiến Lược Lịch Sử",
        "short": "Kiểm tra hiệu suất chiến lược giao dịch trên dữ liệu quá khứ",
        "description": "Backtest mô phỏng quá trình giao dịch theo tín hiệu từ mô hình trên dữ liệu lịch sử để đánh giá hiệu quả. Trong hệ thống này: 60% dữ liệu đầu dùng để huấn luyện Random Forest, 40% còn lại để backtest. Chiến lược RF được so sánh với Buy & Hold (mua và giữ). Lưu ý: lợi nhuận backtest không đảm bảo lợi nhuận thực tế do overfitting và chi phí giao dịch.",
        "formula": "RF_Return = (RF_Final - Capital) / Capital × 100% | B&H_Return = (Giá_cuối / Giá_đầu - 1) × 100%",
        "use_case": "Đánh giá khả năng sinh lời của mô hình trước khi đưa vào thực tế",
        "category": "System Design"
    },
    "Return 1d": {
        "name": "Return_1d – Lợi Nhuận Ngày",
        "short": "Tỷ suất sinh lời so với phiên giao dịch liền trước",
        "description": "Return_1d (lợi nhuận 1 ngày) là phần trăm thay đổi giá đóng cửa so với phiên hôm trước. Đây là feature cơ bản nhất trong phân tích chuỗi thời gian tài chính. Phân phối của Return_1d gần với phân phối chuẩn (Gaussian) nhưng có fat tail – tức là các sự kiện cực đoan xảy ra thường xuyên hơn lý thuyết thống kê dự đoán.",
        "formula": "Return_1d[t] = (Close[t] - Close[t-1]) / Close[t-1]",
        "use_case": "Feature trong Random Forest; tính Volatility và Sharpe Ratio",
        "category": "Risk"
    },
    "Return": {
        "name": "Return – Tổng Lợi Nhuận Kỳ",
        "short": "Phần trăm sinh lời từ đầu kỳ đến cuối kỳ so với vốn ban đầu",
        "description": "Return (tổng lợi nhuận) đo lường mức sinh lời của cổ phiếu trong một khoảng thời gian nhất định, xác định bằng sự thay đổi giữa giá đóng cửa đầu kỳ và cuối kỳ. Return dương → cổ phiếu tăng giá, nhà đầu tư có lãi. Return âm → cổ phiếu giảm giá, nhà đầu tư lỗ. Trong hệ thống này, Return 1 năm được tính từ giá đầu tiên đến giá cuối cùng trong dữ liệu 365 ngày gần nhất.",
        "formula": "Return = (P_end - P_start) / P_start × 100%",
        "use_case": "So sánh lợi nhuận giữa các cổ phiếu; đánh giá hiệu quả đầu tư so với VN-Index và chiến lược Buy & Hold",
        "category": "Risk"
    }
}



def get_ticker_suffix(symbol: str) -> str:
    """Add .VN suffix for Vietnamese stocks"""
    symbol = symbol.upper().strip()
    if not symbol.endswith(".VN") and not "." in symbol:
        return f"{symbol}.VN"
    return symbol

def is_bank_stock(symbol: str) -> bool:
    return symbol.upper().strip() in BANK_TICKERS

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return_1d"] = df["Close"].pct_change()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Volatility_10"] = df["Return_1d"].rolling(10).std()
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    df["Vol_MA_10"] = df["Volume"].rolling(10).mean()
    df["volume_log"] = np.log1p(df["Volume"])
    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["MA_20"] + 2 * df["BB_std"]
    df["BB_lower"] = df["MA_20"] - 2 * df["BB_std"]
    return df

def fetch_stock_data(symbol: str, days: int = 150) -> pd.DataFrame:
    """Fetch stock data from yfinance"""
    ticker_str = get_ticker_suffix(symbol)
    end = datetime.now()
    start = end - timedelta(days=days + 60)  # extra buffer for indicators
    
    df = yf.download(
        ticker_str, start=start, end=end,
        interval="1d", auto_adjust=False,
        progress=False, threads=False
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu cho mã {symbol}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def technical_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate trading signal from technical analysis"""
    df = feature_engineering(df)
    df = df.dropna().reset_index(drop=True)
    
    if len(df) < 5:
        return {"signal": "GIỮ", "confidence": 50, "reason": "Không đủ dữ liệu"}
    
    latest = df.iloc[-1]
    signals = []
    
    # RSI
    rsi = latest["RSI_14"]
    if rsi < 35:
        signals.append(("MUA", 0.8, f"RSI={rsi:.1f} – vùng quá bán"))
    elif rsi > 65:
        signals.append(("BÁN", 0.8, f"RSI={rsi:.1f} – vùng quá mua"))
    else:
        signals.append(("GIỮ", 0.5, f"RSI={rsi:.1f} – trung tính"))
    
    # MACD
    macd = latest["MACD"]
    macd_signal = latest["MACD_Signal"]
    macd_hist = latest["MACD_Hist"]
    prev_hist = df.iloc[-2]["MACD_Hist"] if len(df) > 2 else 0
    
    if macd > macd_signal and prev_hist < 0 and macd_hist > 0:
        signals.append(("MUA", 0.9, "MACD cắt lên Signal – tín hiệu tăng"))
    elif macd < macd_signal and prev_hist > 0 and macd_hist < 0:
        signals.append(("BÁN", 0.9, "MACD cắt xuống Signal – tín hiệu giảm"))
    elif macd > 0:
        signals.append(("MUA", 0.6, "MACD dương – xu hướng tăng"))
    else:
        signals.append(("BÁN", 0.6, "MACD âm – xu hướng giảm"))
    
    # MA Cross
    ma20 = latest["MA_20"]
    ma50 = latest["MA_50"]
    close = latest["Close"]
    
    if close > ma20 > ma50:
        signals.append(("MUA", 0.75, "Giá > MA20 > MA50 – xu hướng tăng"))
    elif close < ma20 < ma50:
        signals.append(("BÁN", 0.75, "Giá < MA20 < MA50 – xu hướng giảm"))
    else:
        signals.append(("GIỮ", 0.5, "MA chưa rõ xu hướng"))
    
    # Bollinger Bands
    bb_upper = latest["BB_upper"]
    bb_lower = latest["BB_lower"]
    
    if close <= bb_lower:
        signals.append(("MUA", 0.7, "Giá chạm BB Lower – phản hồi tăng"))
    elif close >= bb_upper:
        signals.append(("BÁN", 0.7, "Giá chạm BB Upper – có thể giảm"))
    
    # Tổng hợp điểm tín hiệu
    buy_score = sum(w for s, w, _ in signals if s == "MUA")
    sell_score = sum(w for s, w, _ in signals if s == "BÁN")
    hold_score = sum(w for s, w, _ in signals if s == "GIỮ")
    total = buy_score + sell_score + hold_score or 1
    
    if buy_score > sell_score and buy_score > hold_score:
        final_signal = "MUA"
        confidence = int(buy_score / total * 100)
    elif sell_score > buy_score and sell_score > hold_score:
        final_signal = "BÁN"
        confidence = int(sell_score / total * 100)
    else:
        final_signal = "GIỮ"
        confidence = int(hold_score / total * 100)
    
    confidence = max(45, min(92, confidence))
    reasons = [r for _, _, r in signals if r]
    
    return {
        "signal": final_signal,
        "confidence": confidence,
        "reasons": reasons[:3],
        "rsi": round(float(rsi), 2),
        "macd": round(float(macd), 2),
        "macd_signal": round(float(macd_signal), 2),
        "ma20": round(float(ma20), 2),
        "ma50": round(float(ma50), 2),
        "bb_upper": round(float(bb_upper), 2),
        "bb_lower": round(float(bb_lower), 2),
    }

def _prepare_rf_data(df: pd.DataFrame):

    from sklearn.preprocessing import StandardScaler
    df = feature_engineering(df)
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Return_1d", "MA_20", "MA_50", "Volatility_10",
        "RSI_14", "Vol_MA_10", "volume_log", "Momentum_5",
        "EMA_12", "EMA_26", "MACD", "BB_std", "BB_upper", "BB_lower"
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    y = df["Target"].values
    split = int(len(df) * 0.8)
    # Fit scaler chỉ trên tập Train, sau đó transform Test riêng để tránh Data Leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split])
    y_train = y[:split]
    X_test_s = scaler.transform(X[split:])   # transform tập test độc lập
    y_test = y[split:]
    X_latest_s = scaler.transform(X[-1:])    # điểm dự báo mới nhất
    latest = df.iloc[-1]
    return X_train_s, y_train, X_test_s, y_test, X_latest_s, latest, df


def random_forest_signal(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Random Forest classifier signal for bank stocks (100 trees, depth 8)."""
    from sklearn.ensemble import RandomForestClassifier
    df_orig = df.copy()
    try:
        X_train_s, y_train, _X_test_s, _y_test, X_latest_s, latest, df_fe = _prepare_rf_data(df_orig)
        if len(df_fe) < 100:
            return technical_signal(feature_engineering(df_orig).dropna())
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        proba = rf.predict_proba(X_latest_s)[0]
        buy_prob = float(proba[1]) if len(proba) > 1 else 0.5
        sell_prob = 1 - buy_prob
        # Threshold 0.60 cố định — web dùng real-time data, không có ground truth
        # để tối ưu F1 như trong notebook (notebook dùng Validation Set để tìm threshold)
        if buy_prob > 0.60:
            signal, confidence = "MUA", int(buy_prob * 100)
        elif sell_prob > 0.60:
            signal, confidence = "BÁN", int(sell_prob * 100)
        else:
            signal, confidence = "GIỮ", int(max(buy_prob, sell_prob) * 100)
        confidence = max(50, min(90, confidence))
        rsi  = round(float(latest.get("RSI_14", 50)), 2)
        macd = round(float(latest.get("MACD", 0)), 2)
        # Feature importance top-3
        rf2 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, n_jobs=-1)
        rf2.fit(X_train_s, y_train)
        feat_names = [
            "Open","High","Low","Close","Volume","Return_1d","MA_20","MA_50",
            "Volatility_10","RSI_14","Vol_MA_10","volume_log","Momentum_5",
            "EMA_12","EMA_26","MACD","BB_std","BB_upper","BB_lower"
        ]
        feat_names = feat_names[:X_train_s.shape[1]]
        top_feat = sorted(zip(feat_names, rf2.feature_importances_), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{n}({v*100:.0f}%)" for n, v in top_feat)
        return {
            "signal": signal, "confidence": confidence,
            "buy_probability": round(buy_prob * 100, 1),
            "model": "Random Forest",
            "model_short": "RF",
            "reasons": [
                f"RF xác suất tăng: {buy_prob*100:.1f}% | giảm: {sell_prob*100:.1f}%",
                f"Feature quan trọng nhất: {top_str}",
                f"RSI={rsi} · MACD={macd}",
            ],
            "rsi": rsi, "macd": macd,
            "macd_signal": round(float(latest.get("MACD_Signal", 0)), 2),
            "ma20": round(float(latest.get("MA_20", 0)), 2),
            "ma50": round(float(latest.get("MA_50", 0)), 2),
            "bb_upper": round(float(latest.get("BB_upper", 0)), 2),
            "bb_lower": round(float(latest.get("BB_lower", 0)), 2),
        }
    except Exception:
        return technical_signal(feature_engineering(df.copy()).dropna())


def lstm_signal(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
   
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        return technical_signal(feature_engineering(df.copy()).dropna())

    df_fe = feature_engineering(df.copy()).dropna().reset_index(drop=True)
    if len(df_fe) < 80:
        return technical_signal(df_fe)

    WINDOW = 20
    close = df_fe["Close"].values.reshape(-1, 1)

    # Chia train/test trước khi fit scaler để tránh Data Leakage
    split = int(len(close) * 0.8)
    scaler = MinMaxScaler()
    scaler.fit(close[:split])                            # fit chỉ trên tập Train
    train_scaled = scaler.transform(close[:split]).flatten()   # transform train riêng
    test_scaled  = scaler.transform(close[split:]).flatten()   # transform test riêng
    scaled = np.concatenate([train_scaled, test_scaled])       # ghép lại để tạo chuỗi dữ liệu theo thứ tự thời gian

   
    X_seq, y_seq = [], []
    for i in range(WINDOW, len(scaled)):
        X_seq.append(scaled[i - WINDOW:i])
        y_seq.append(scaled[i])
    X_seq = np.array(X_seq).reshape(-1, WINDOW, 1)
    y_seq = np.array(y_seq)

    split_seq = split - WINDOW
    if split_seq < 10:
        return technical_signal(df_fe)

    X_train, y_train = X_seq[:split_seq], y_seq[:split_seq]
    X_test,  y_test  = X_seq[split_seq:], y_seq[split_seq:]

    # Xây dựng mô hình LSTM 2 lớp với Dropout chống overfitting
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    # Dự báo giá ngày kế tiếp
    last_window = scaled[-WINDOW:].reshape(1, WINDOW, 1)
    next_scaled  = float(model.predict(last_window, verbose=0)[0][0])
    next_price   = float(scaler.inverse_transform([[next_scaled]])[0][0])
    last_price   = float(close[-1][0])
    lstm_trend   = (next_price - last_price) / (last_price + 1e-9)

    # Đánh giá sai số trên tập Test
    pred_test_scaled = model.predict(X_test, verbose=0).flatten()
    actual_test      = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_test        = scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()
    errors_test      = actual_test - pred_test
    mape             = float(np.mean(np.abs(errors_test / (actual_test + 1e-9))) * 100)

    # Chuyển xu hướng dự báo từ LSTM thành tín hiệu mua/bán
    if lstm_trend > 0.004:
        signal, buy_p = "MUA", 0.55 + min(lstm_trend * 30, 0.30)
    elif lstm_trend < -0.004:
        signal, buy_p = "BÁN", 0.55 - min(abs(lstm_trend) * 30, 0.30)
    else:
        signal, buy_p = "GIỮ", 0.50
    sell_p = 1 - buy_p
    confidence = max(50, min(88, int(max(buy_p, sell_p) * 100)))

    latest = df_fe.iloc[-1]
    rsi  = round(float(latest.get("RSI_14", 50)), 2)
    macd = round(float(latest.get("MACD", 0)), 2)

    return {
        "signal": signal, "confidence": confidence,
        "buy_probability": round(buy_p * 100, 1),
        "model": "LSTM",
        "model_short": "LSTM",
        "reasons": [
            f"LSTM dự báo giá kế tiếp: {next_price:,.0f} ({lstm_trend*100:+.2f}%)",
            f"MAPE tập Test: {mape:.2f}% | Cửa sổ: {WINDOW} ngày",
            f"RSI={rsi} · MACD={macd}",
        ],
        "mape": round(mape, 2),
        "lstm_trend_pct": round(lstm_trend * 100, 3),
        "rsi": rsi, "macd": macd,
        "macd_signal": round(float(latest.get("MACD_Signal", 0)), 2),
        "ma20": round(float(latest.get("MA_20", 0)), 2),
        "ma50": round(float(latest.get("MA_50", 0)), 2),
        "bb_upper": round(float(latest.get("BB_upper", 0)), 2),
        "bb_lower": round(float(latest.get("BB_lower", 0)), 2),
    }


def hybrid_signal(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    Hybrid model: weighted vote of RF (60%) + LSTM (40%) for bank stocks.
    Falls back gracefully if either sub-model fails.
    """
    rf   = random_forest_signal(df.copy(), symbol)
    lstm = lstm_signal(df.copy(), symbol)

    # Ánh xạ tín hiệu sang giá trị số: MUA=1, GIỮ=0, BÁN=-1
    sig_map = {"MUA": 1, "GIỮ": 0, "BÁN": -1}
    rf_score   = sig_map.get(rf["signal"], 0)   * (rf["confidence"]   / 100)
    lstm_score = sig_map.get(lstm["signal"], 0) * (lstm["confidence"] / 100)

    # Kết hợp có trọng số: RF 60%, LSTM 40%
    combined = rf_score * 0.60 + lstm_score * 0.40

    if combined > 0.12:
        signal = "MUA"
        confidence = max(50, min(92, int(50 + combined * 60)))
    elif combined < -0.12:
        signal = "BÁN"
        confidence = max(50, min(92, int(50 + abs(combined) * 60)))
    else:
        signal = "GIỮ"
        confidence = max(50, min(85, int(50 + abs(combined) * 40)))

    agree = "RF & LSTM đồng thuận" if rf["signal"] == lstm["signal"] else "RF & LSTM bất đồng – tín hiệu yếu hơn"

    return {
        "signal": signal, "confidence": confidence,
        "model": "Hybrid (RF+LSTM)",
        "model_short": "Hybrid",
        "rf_signal": rf["signal"], "rf_conf": rf["confidence"],
        "lstm_signal": lstm["signal"], "lstm_conf": lstm["confidence"],
        "combined_score": round(combined, 3),
        "reasons": [
            agree,
            f"RF ({rf['signal']} {rf['confidence']}%) × 60% + LSTM ({lstm['signal']} {lstm['confidence']}%) × 40%",
            f"Điểm kết hợp: {combined:+.3f} → {signal}",
        ],
        "rsi":  rf.get("rsi",  50), "macd": rf.get("macd", 0),
        "macd_signal": rf.get("macd_signal", 0),
        "ma20": rf.get("ma20", 0), "ma50": rf.get("ma50", 0),
        "bb_upper": rf.get("bb_upper", 0), "bb_lower": rf.get("bb_lower", 0),
    }

def forecast_price(df: pd.DataFrame, days: int = 7) -> List[Dict]:
    """Simple linear forecast based on recent trend + volatility, with floor/ceiling bands"""
    recent = df.tail(20)
    close = recent["Close"].values
    returns = pd.Series(close).pct_change().dropna()
    
    avg_return = float(returns.mean())
    std_return = float(returns.std())
    last_price = float(close[-1])
    
    forecasts = []
    current = last_price
    base_date = pd.to_datetime(df["Date"].iloc[-1])
    
    for i in range(1, days + 1):
        
        daily_return = avg_return * (0.7 ** i) + np.random.normal(0, std_return * 0.1)
        current = current * (1 + daily_return)
        next_date = base_date + timedelta(days=i)
      
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        # Dải tin cậy 1σ × √i (biến động tích lũy theo thời gian)
        band = last_price * std_return * np.sqrt(i) * 1.0
        price_floor   = round(float(current - band), 2)
        price_ceiling = round(float(current + band), 2)

        forecasts.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "price": round(float(current), 2),
            "change_pct": round(daily_return * 100, 2),
            "price_floor": price_floor,
            "price_ceiling": price_ceiling,
        })
    
    return forecasts

def get_vnindex_benchmark(days: int = 365) -> Dict[str, Any]:
  
    try:
        end = datetime.now()
        start = end - timedelta(days=days + 30)
        df_vn = yf.download(
            "VNINDEX.VN", start=start, end=end,
            interval="1d", auto_adjust=False,
            progress=False, threads=False
        )
        if df_vn.empty:
            return {"return_pct": 0.0, "chart": [], "available": False}

        if isinstance(df_vn.columns, pd.MultiIndex):
            df_vn.columns = df_vn.columns.get_level_values(0)
        df_vn = df_vn.reset_index().sort_values("Date").reset_index(drop=True)

        close = df_vn["Close"].dropna().values
        if len(close) < 2:
            return {"return_pct": 0.0, "chart": [], "available": False}

        ret = (float(close[-1]) - float(close[0])) / float(close[0]) * 100
        chart_60 = [round(float(v), 2) for v in close[-60:]]
        return {"return_pct": round(ret, 2), "chart": chart_60, "available": True,
                "current": round(float(close[-1]), 2)}
    except Exception:
        return {"return_pct": 0.0, "chart": [], "available": False}


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """Tính Beta của cổ phiếu so với thị trường (VN-Index)"""
    df_tmp = pd.DataFrame({"s": stock_returns, "m": market_returns}).dropna()
    if len(df_tmp) < 20:
        return 1.0
    cov = float(df_tmp["s"].cov(df_tmp["m"]))
    var = float(df_tmp["m"].var())
    return round(cov / var, 3) if var > 0 else 1.0

   
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        return {"dates": [], "actual": [], "lstm_pred": [], "error_points": []}

    df = feature_engineering(df).dropna().reset_index(drop=True)
    if len(df) < 80:
        return {"dates": [], "actual": [], "lstm_pred": [], "error_points": []}

    WINDOW = 20
    close = df["Close"].values.reshape(-1, 1)

    # Chia train/test trước khi fit scaler để tránh Data Leakage
    split = int(len(close) * 0.8)
    scaler = MinMaxScaler()
    scaler.fit(close[:split])                                    # fit chỉ trên tập Train
    train_scaled = scaler.transform(close[:split]).flatten()     # transform train riêng
    test_scaled  = scaler.transform(close[split:]).flatten()     # transform test riêng
    scaled = np.concatenate([train_scaled, test_scaled])         

   
    X_seq, y_seq = [], []
    for i in range(WINDOW, len(scaled)):
        X_seq.append(scaled[i - WINDOW:i])
        y_seq.append(scaled[i])
    X_seq = np.array(X_seq).reshape(-1, WINDOW, 1)
    y_seq = np.array(y_seq)

    split_seq = split - WINDOW
    X_train, y_train = X_seq[:split_seq], y_seq[:split_seq]
    X_test,  y_test  = X_seq[split_seq:], y_seq[split_seq:]

    
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    # Dự báo trên toàn bộ sequence để vẽ đồ thị so sánh
    preds_scaled = model.predict(X_seq, verbose=0).flatten()
    pred_prices  = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actual_prices = close[WINDOW:].flatten()
    dates = df["Date"].iloc[WINDOW:].dt.strftime("%Y-%m-%d").tolist()

    # Sử dụng 90 điểm dữ liệu cuối cùng để hiển thị.
    n = min(90, len(dates))
    dates = dates[-n:]
    actual_prices = actual_prices[-n:]
    pred_prices = pred_prices[-n:]

    #Phân tích sai số: tìm các điểm mà |sai số| > 1.5 độ lệch chuẩn.
    errors = actual_prices - pred_prices
    err_std = float(np.std(errors))
    err_mean = float(np.mean(np.abs(errors)))

    # Trích xuất ngữ cảnh của các chỉ báo để tạo giải thích.
    rsi_vals = df["RSI_14"].values[-n:]
    macd_hist_vals = df["MACD_Hist"].values[-n:]
    bb_upper_vals = df["BB_upper"].values[-n:]
    bb_lower_vals = df["BB_lower"].values[-n:]
    vol_ma = df["Vol_MA_10"].values[-n:]
    volume_vals = df["Volume"].values[-n:]

    error_points = []
    for i, (err, act, pred, date) in enumerate(zip(errors, actual_prices, pred_prices, dates)):
        if abs(err) < 1.5 * err_std:
            continue
        direction = "tăng" if err > 0 else "giảm"
        rsi = float(rsi_vals[i]) if i < len(rsi_vals) else 50.0
        macd_h = float(macd_hist_vals[i]) if i < len(macd_hist_vals) else 0.0
        vol_ratio = float(volume_vals[i] / vol_ma[i]) if (i < len(vol_ma) and vol_ma[i] > 0) else 1.0

      
        reasons = []
        if rsi > 75:
            reasons.append(f"RSI={rsi:.0f} – vùng quá mua cực độ nhưng giá vẫn {direction} do lực mua tổ chức")
        elif rsi < 25:
            reasons.append(f"RSI={rsi:.0f} – vùng quá bán cực độ nhưng giá vẫn {direction} do panic sell")
        elif rsi > 65:
            reasons.append(f"RSI={rsi:.0f} – LSTM kỳ vọng điều chỉnh nhưng đà tăng mạnh hơn dự báo")

        if vol_ratio > 2.5:
            reasons.append(f"Khối lượng giao dịch đột biến {vol_ratio:.1f}x trung bình (tin tức bất ngờ)")
        elif vol_ratio < 0.4:
            reasons.append("Thanh khoản cạn kiệt – LSTM không học được pattern biến động thấp")

        if macd_h > 0 and err < 0:
            reasons.append("MACD Histogram dương nhưng giá giảm – phân kỳ ẩn (hidden divergence)")
        elif macd_h < 0 and err > 0:
            reasons.append("MACD Histogram âm nhưng giá tăng – đảo chiều bất ngờ ngoài pattern")

        if not reasons:
          
            if err > 0:
                reasons.append("Giá tăng mạnh hơn dự báo – có thể do tin tức tích cực (chia cổ tức, kết quả kinh doanh vượt kỳ vọng)")
            else:
                reasons.append("Giá giảm sâu hơn dự báo – sự kiện vĩ mô (lãi suất, tỷ giá) tác động ngoài dữ liệu lịch sử")

        explanation = f" LSTM sai {abs(err/act*100):.1f}% ngày {date}: {'; '.join(reasons[:2])}"

        error_points.append({
            "index": i,
            "date": date,
            "actual": round(float(act), 2),
            "predicted": round(float(pred), 2),
            "error_pct": round(float(err / act * 100), 2),
            "explanation": explanation
        })

    
    mape = float(np.mean(np.abs(errors / actual_prices)) * 100)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "dates": dates,
        "actual": [round(float(v), 2) for v in actual_prices],
        "lstm_pred": [round(float(v), 2) for v in pred_prices],
        "error_points": error_points,
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "total_errors": len(error_points),
        "window": WINDOW
    }


def get_market_regime(df: pd.DataFrame) -> str:

    df = feature_engineering(df)
    df = df.dropna()
    if len(df) < 20:
        return "ĐI NGANG"
    latest = df.iloc[-1]
    returns_20 = df["Return_1d"].tail(20).mean()
    
    if returns_20 > 0.003 and latest["Close"] > latest["MA_20"]:
        return "TĂNG"
    elif returns_20 < -0.003 and latest["Close"] < latest["MA_20"]:
        return "GIẢM"
    return "ĐI NGANG"

def run_backtest(df: pd.DataFrame, capital: float, symbol: str) -> Dict[str, Any]:
    """Simple RF-based backtest vs Buy & Hold"""
    df = feature_engineering(df)
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Return_1d", "MA_20", "MA_50", "Volatility_10",
        "RSI_14", "Vol_MA_10", "volume_log", "Momentum_5",
        "EMA_12", "EMA_26", "MACD", "BB_std", "BB_upper", "BB_lower"
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    y = df["Target"].values
    
    split = int(len(df) * 0.6)  
    X_train = X[:split]
    y_train = y[:split]

    # Fit scaler chỉ trên tập Train, transform Test độc lập để tránh Data Leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X[split:])   # transform tập test độc lập
    X_all_s   = np.vstack([X_train_s, X_test_s])

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    rf.fit(X_train_s, y_train)

    preds = rf.predict(X_all_s)
    
    # RF Strategy
    rf_equity = [capital]
    rf_pos = 0
    rf_cash = capital
    
    # Buy & Hold
    bah_equity = [capital]
    initial_price = df["Close"].iloc[split]
    shares_bah = capital / initial_price
    
    dates = []
    for i in range(split, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        date_str = str(row.get("Date", i))[:10]
        dates.append(date_str)
        
        # RF action
        if preds[i] == 1 and rf_pos == 0:  
            rf_pos = rf_cash / row["Close"]
            rf_cash = 0
        elif preds[i] == 0 and rf_pos > 0:  
            rf_cash = rf_pos * row["Close"]
            rf_pos = 0
        
        rf_val = rf_cash + rf_pos * next_row["Close"]
        rf_equity.append(rf_val)
        bah_equity.append(shares_bah * next_row["Close"])
    
    rf_final = rf_equity[-1]
    bah_final = bah_equity[-1]
    
    rf_return = (rf_final - capital) / capital * 100
    bah_return = (bah_final - capital) / capital * 100
    
    return {
        "dates": dates,
        "rf_equity": [round(v, 0) for v in rf_equity[1:]],
        "bah_equity": [round(v, 0) for v in bah_equity[1:]],
        "rf_final": round(rf_final, 0),
        "bah_final": round(bah_final, 0),
        "rf_return_pct": round(rf_return, 2),
        "bah_return_pct": round(bah_return, 2),
        "initial_capital": capital,
        "outperform": rf_return > bah_return
    }

#  API 

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/api/dashboard")
async def get_dashboard():
  
    results = []
    for sym in FEATURED_TICKERS:
        try:
            df = fetch_stock_data(sym, days=60)
            df_fe = feature_engineering(df)
            df_fe = df_fe.dropna()
            
            latest = df_fe.iloc[-1]
            prev = df_fe.iloc[-2] if len(df_fe) > 2 else latest
            
            close = float(latest["Close"])
            prev_close = float(prev["Close"])
            change_pct = (close - prev_close) / prev_close * 100
            
            # Biểu đồ thu nhỏ  trong 30 ngày gần nhất.
            spark = df_fe["Close"].tail(30).tolist()
            
            # Signal
            is_bank = is_bank_stock(sym)
            if is_bank:
                sig_data = hybrid_signal(df.copy(), sym)
            else:
                sig_data = technical_signal(df_fe.copy())
            
            regime = get_market_regime(df.copy())
            
            results.append({
                "symbol": sym,
                "close": round(close, 2),
                "change_pct": round(change_pct, 2),
                "sparkline": [round(float(v), 2) for v in spark],
                "signal": sig_data["signal"],
                "confidence": sig_data["confidence"],
                "is_bank": is_bank,
                "analysis_type": "Model (RF+LSTM)" if is_bank else "Phân tích kỹ thuật",
                "regime": regime,
                "rsi": sig_data.get("rsi", 50),
                "volume": int(df["Volume"].iloc[-1]) if not df.empty else 0,
            })
        except Exception as e:
            continue
    
    # Market-wide regime dựa trên ^VNINDEX – chỉ số đại diện toàn sàn HoSE
    market_regime = "ĐI NGANG"
    try:
        end = datetime.now()
        start = end - timedelta(days=90)
        df_market = yf.download(
            "VNINDEX.VN", start=start, end=end,
            interval="1d", auto_adjust=False,
            progress=False, threads=False
        )
        if not df_market.empty:
            if isinstance(df_market.columns, pd.MultiIndex):
                df_market.columns = df_market.columns.get_level_values(0)
            df_market = df_market.reset_index()
            df_market["Date"] = pd.to_datetime(df_market["Date"])
            df_market = df_market.sort_values("Date").reset_index(drop=True)
            market_regime = get_market_regime(df_market)
    except:
        pass
    
    return {
        "stocks": results,
        "market_regime": market_regime,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stock/{symbol}")
async def get_stock_analysis(symbol: str, timeframe: str = "7"):
   
    symbol = symbol.upper().strip()
    days_map = {"1": 30, "3": 60, "7": 90, "30": 180}
    fetch_days = days_map.get(timeframe, 90)
    
    df = fetch_stock_data(symbol, days=fetch_days + 60)
    df_fe = feature_engineering(df)
    df_fe = df_fe.dropna().reset_index(drop=True)
    
    is_bank = is_bank_stock(symbol)
    analysis_type = "Model (RF+LSTM)" if is_bank else "Phân tích kỹ thuật"

    #  Model signals 
    rf_sig   = None
    lstm_sig = None
    hyb_sig  = None

    if is_bank:
        rf_sig   = random_forest_signal(df.copy(), symbol)
        lstm_sig = lstm_signal(df.copy(), symbol)
        hyb_sig  = hybrid_signal(df.copy(), symbol)
        sig_data = hyb_sig          # primary signal = Hybrid
    else:
        sig_data = technical_signal(df_fe.copy())

    # Dữ liệu biểu đồ trong 90 ngày gần nhất
    chart_df = df_fe.tail(90)
    chart_data = {
        "dates": chart_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "open":  [round(float(v), 2) for v in chart_df["Open"]],
        "close": [round(float(v), 2) for v in chart_df["Close"]],
        "volume": [int(v) for v in chart_df["Volume"]],
        "ma20": [round(float(v), 2) if not np.isnan(v) else None for v in chart_df["MA_20"]],
        "ma50": [round(float(v), 2) if not np.isnan(v) else None for v in chart_df["MA_50"]],
        "bb_upper": [round(float(v), 2) if not np.isnan(v) else None for v in chart_df["BB_upper"]],
        "bb_lower": [round(float(v), 2) if not np.isnan(v) else None for v in chart_df["BB_lower"]],
        "rsi": [round(float(v), 2) if not np.isnan(v) else None for v in chart_df["RSI_14"]],
        "macd": [round(float(v), 2) if not np.isnan(v) else None for v in chart_df["MACD"]],
        "macd_signal": [round(float(v), 2) if not np.isnan(v) else None for v in chart_df["MACD_Signal"]],
        "macd_hist": [round(float(v), 2) if not np.isnan(v) else None for v in chart_df["MACD_Hist"]],
    }
    
    # Dự báo
    forecast = forecast_price(df, days=7)
    latest_close = float(df_fe["Close"].iloc[-1])
    forecast_7d_price = forecast[-1]["price"] if forecast else latest_close
    forecast_change_pct = (forecast_7d_price - latest_close) / latest_close * 100

    # Trần / Sàn hiện tại từ Bollinger Bands (đã có sẵn)
    current_ceiling = float(df_fe["BB_upper"].iloc[-1]) if not np.isnan(df_fe["BB_upper"].iloc[-1]) else None
    current_floor   = float(df_fe["BB_lower"].iloc[-1]) if not np.isnan(df_fe["BB_lower"].iloc[-1]) else None

    # Tham chiếu VN-Index
    vni = get_vnindex_benchmark(days=fetch_days + 60)
    start_price_stock = float(df_fe["Close"].iloc[0])
    stock_return_pct  = (latest_close - start_price_stock) / start_price_stock * 100

    # Beta
    try:
        end_dt  = datetime.now()
        start_dt = end_dt - timedelta(days=fetch_days + 90)
        df_vn = yf.download("VNINDEX.VN", start=start_dt, end=end_dt,
                            interval="1d", auto_adjust=False, progress=False, threads=False)
        if not df_vn.empty:
            if isinstance(df_vn.columns, pd.MultiIndex):
                df_vn.columns = df_vn.columns.get_level_values(0)
            df_vn = df_vn.reset_index().sort_values("Date").reset_index(drop=True)
            df_vn["Ret"] = df_vn["Close"].pct_change()
            df_merged = pd.merge(
                df_fe[["Date", "Return_1d"]].rename(columns={"Return_1d": "stock_ret"}),
                df_vn[["Date", "Ret"]].rename(columns={"Ret": "mkt_ret"}),
                on="Date", how="inner"
            )
            beta = compute_beta(df_merged["stock_ret"], df_merged["mkt_ret"])
        else:
            beta = 1.0
    except Exception:
        beta = 1.0

    benchmark = {
        "vnindex_return_pct": vni["return_pct"],
        "stock_return_pct":   round(stock_return_pct, 2),
        "vs_vnindex_pct":     round(stock_return_pct - vni["return_pct"], 2),
        "outperform":         stock_return_pct > vni["return_pct"],
        "beta":               beta,
        "available":          vni["available"],
    }

    # Regime
    regime = get_market_regime(df.copy())
    
    # Multi-timeframe signals
    mtf = {}
    for tf, d in [("1D", 5), ("3D", 15), ("7D", 30), ("30D", 90)]:
        try:
            df_tf = df_fe.tail(d + 20)
            if is_bank:
                s = hybrid_signal(df.tail(d + 60).copy(), symbol)
            else:
                s = technical_signal(df_tf.copy())
            mtf[tf] = {"signal": s["signal"], "confidence": s["confidence"]}
        except:
            mtf[tf] = {"signal": "GIỮ", "confidence": 50}
    
    return {
        "symbol": symbol,
        "analysis_type": analysis_type,
        "is_bank": is_bank,
        "current_price": latest_close,
        "signal": sig_data["signal"],
        "confidence": sig_data["confidence"],
        "reasons": sig_data.get("reasons", []),
        "indicators": {
            "rsi": sig_data.get("rsi"),
            "macd": sig_data.get("macd"),
            "ma20": sig_data.get("ma20"),
            "ma50": sig_data.get("ma50"),
            "bb_upper": sig_data.get("bb_upper"),
            "bb_lower": sig_data.get("bb_lower"),
        },
        #  Chỉ áp dụng cho nhóm ngân hàng: phân tích chi tiết từng mô hình riêng lẻ
        "models": {
            "rf":     rf_sig,
            "lstm":   lstm_sig,
            "hybrid": hyb_sig,
        } if is_bank else None,
        "forecast": forecast,
        "forecast_change_pct": round(forecast_change_pct, 2),
        "forecast_7d_price": round(forecast_7d_price, 2),
        "current_floor": round(current_floor, 2) if current_floor else None,
        "current_ceiling": round(current_ceiling, 2) if current_ceiling else None,
        "benchmark": benchmark,
        "regime": regime,
        "chart_data": chart_data,
        "multi_timeframe": mtf,
        "timeframe": timeframe,
    }

class BacktestRequest(BaseModel):
    symbol: str
    capital: float
    start_date: Optional[str] = None

@app.post("/api/backtest")
async def run_backtest_api(req: BacktestRequest):
    """Run backtest for a stock"""
    symbol = req.symbol.upper().strip()
    capital = max(1_000_000, req.capital)
    
    df = fetch_stock_data(symbol, days=365 * 3)
    
    if req.start_date:
        try:
            start = pd.to_datetime(req.start_date)
            df = df[df["Date"] >= start].reset_index(drop=True)
        except:
            pass
    
    if len(df) < 100:
        raise HTTPException(status_code=400, detail="Không đủ dữ liệu lịch sử để backtest")
    
    result = run_backtest(df, capital, symbol)
    result["symbol"] = symbol
    return result

@app.get("/api/compare")
async def compare_stocks(symbols: str):
    """Compare multiple stocks"""
    sym_list = [s.strip().upper() for s in symbols.split(",")][:6]
    results = []

    # Lấy VN-Index một lần cho tất cả 365 ngày
    vni = get_vnindex_benchmark(days=365)
    
    for sym in sym_list:
        try:
            df = fetch_stock_data(sym, days=365)
            df_fe = feature_engineering(df).dropna()
            
            is_bank = is_bank_stock(sym)
            sig = hybrid_signal(df.copy(), sym) if is_bank else technical_signal(df_fe.copy())
   
            start_price = float(df_fe["Close"].iloc[0])
            end_price = float(df_fe["Close"].iloc[-1])
            return_pct = (end_price - start_price) / start_price * 100
            
            # Volatility 
            daily_std = float(df_fe["Return_1d"].std())
            annual_vol = daily_std * np.sqrt(252) * 100
            
            # Max drawdown
            roll_max = df_fe["Close"].cummax()
            drawdown = (df_fe["Close"] - roll_max) / roll_max
            max_dd = float(drawdown.min()) * 100
            
            # Sharpe (simplified, assuming 0 risk-free)
            avg_ret = float(df_fe["Return_1d"].mean()) * 252
            sharpe = avg_ret / (daily_std * np.sqrt(252)) if daily_std > 0 else 0

            # Trần / Sàn hiện tại (BB)
            current_ceiling = float(df_fe["BB_upper"].iloc[-1]) if not np.isnan(df_fe["BB_upper"].iloc[-1]) else None
            current_floor   = float(df_fe["BB_lower"].iloc[-1]) if not np.isnan(df_fe["BB_lower"].iloc[-1]) else None

            # Beta so với VN-Index
            try:
                end_dt  = datetime.now()
                start_dt = end_dt - timedelta(days=425)
                df_vn = yf.download("VNINDEX.VN", start=start_dt, end=end_dt,
                                    interval="1d", auto_adjust=False, progress=False, threads=False)
                if not df_vn.empty:
                    if isinstance(df_vn.columns, pd.MultiIndex):
                        df_vn.columns = df_vn.columns.get_level_values(0)
                    df_vn = df_vn.reset_index().sort_values("Date").reset_index(drop=True)
                    df_vn["Ret"] = df_vn["Close"].pct_change()
                    df_merged = pd.merge(
                        df_fe[["Date", "Return_1d"]].rename(columns={"Return_1d": "stock_ret"}),
                        df_vn[["Date", "Ret"]].rename(columns={"Ret": "mkt_ret"}),
                        on="Date", how="inner"
                    )
                    beta = compute_beta(df_merged["stock_ret"], df_merged["mkt_ret"])
                else:
                    beta = 1.0
            except Exception:
                beta = 1.0

            results.append({
                "symbol": sym,
                "signal": sig["signal"],
                "confidence": sig["confidence"],
                "is_bank": is_bank,
                "current_price": round(end_price, 2),
                "return_pct": round(return_pct, 2),
                "annual_volatility": round(annual_vol, 2),
                "max_drawdown": round(max_dd, 2),
                "sharpe_ratio": round(sharpe, 3),
                "current_floor":   round(current_floor, 2) if current_floor else None,
                "current_ceiling": round(current_ceiling, 2) if current_ceiling else None,
                "beta": beta,
                "vs_vnindex_pct": round(return_pct - vni["return_pct"], 2),
                "outperform": return_pct > vni["return_pct"],
                "chart": [round(float(v), 2) for v in df_fe["Close"].tail(60).tolist()],
                "volume_series": [int(v) for v in df_fe["Volume"].tail(60).tolist()],
            })
        except Exception as e:
            continue
    
    return {"stocks": results, "vnindex": vni, "timestamp": datetime.now().isoformat()}

@app.get("/api/lstm/{symbol}")
async def get_lstm_analysis(symbol: str):
    """LSTM prediction vs actual price + error analysis for detail page"""
    symbol = symbol.upper().strip()
    df = fetch_stock_data(symbol, days=200)
    result = lstm_forecast_and_errors(df, symbol)
    result["symbol"] = symbol
    return result


@app.get("/api/glossary")
async def get_glossary():
    return {"terms": GLOSSARY}

@app.get("/api/glossary/{term}")
async def get_term(term: str):
    if term not in GLOSSARY:
        raise HTTPException(status_code=404, detail="Không tìm thấy thuật ngữ")
    return GLOSSARY[term]

import os
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
