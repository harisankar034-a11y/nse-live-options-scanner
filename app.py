import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template

app = Flask(__name__)

IST = ZoneInfo("Asia/Kolkata")

# F&O stocks
FNO_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB", "CANBK",
    "IDFCFIRSTB", "FEDERALBNK", "TATAMOTORS", "M&M", "MARUTI",
    "EICHERMOT", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR",
    "ADANIENT", "ADANIPORTS", "BEL", "HAL", "TRENT", "TITAN",
    "BHARTIARTL", "ITC", "HINDUNILVR", "ASIANPAINT",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "TCS", "INFY",
    "WIPRO", "TECHM", "LT", "DLF", "COFORGE", "HCLTECH",
    "SBILIFE", "HDFCLIFE", "BAJFINANCE", "BAJAJFINSV",
    "SHRIRAMFIN", "JIOFIN", "ONGC", "NTPC", "POWERGRID",
    "COALINDIA"
]

# Cash universe
CASH_SYMBOLS = list(dict.fromkeys(FNO_SYMBOLS + [
    "ABB", "ACC", "ADANIGREEN", "AMBUJACEM", "APOLLOHOSP",
    "BEL", "BIOCON", "BOSCHLTD", "BRITANNIA", "CHOLAFIN",
    "DABUR", "GAIL", "GODREJCP", "GRASIM", "HAVELLS",
    "ICICIGI", "INDIGO", "IOC", "IRCTC", "JINDALSTEL",
    "LICI", "LUPIN", "MARICO", "MAXHEALTH", "NAUKRI",
    "PIDILITIND", "RECLTD", "SAIL", "SIEMENS", "SRF",
    "TATACONSUM", "TATASTEEL", "TORNTPHARM", "ULTRACEMCO",
    "VEDL", "YESBANK", "ZYDUSLIFE"
]))


def nse_ticker(symbol):
    return f"{symbol}.NS"


def safe_float(value, digits=2):
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            str(col[-1] if isinstance(col, tuple) else col)
            for col in df.columns
        ]

    return df


def scan_symbol(symbol):
    try:
        ticker = yf.Ticker(nse_ticker(symbol))

        df = ticker.history(
            period="5d",
            interval="5m",
            auto_adjust=False
        )

        if df is None or df.empty:
            return None

        df = flatten_columns(df)

        required = ["Open", "High", "Low", "Close", "Volume"]

        if not all(col in df.columns for col in required):
            return None

        df = df.dropna(subset=required).copy()

        if len(df) < 30:
            return None

        close = df["Close"]
        volume = df["Volume"]

        df["RSI"] = calculate_rsi(close)
        df["ATR"] = calculate_atr(df)

        df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()

        typical_price = (
            df["High"] + df["Low"] + df["Close"]
        ) / 3

        df["VWAP"] = (
            typical_price * volume
        ).cumsum() / volume.cumsum()

        df["AVG_VOL20"] = volume.rolling(20).mean()

        last = df.iloc[-1]

        price = safe_float(last["Close"])
        rsi = safe_float(last["RSI"])
        atr = safe_float(last["ATR"])
        vwap = safe_float(last["VWAP"])
        ema20 = safe_float(last["EMA20"])
        ema50 = safe_float(last["EMA50"])

        if price is None:
            return None

        previous_close = safe_float(df["Close"].iloc[-2])

        if previous_close:
            momentum = ((price - previous_close) / previous_close) * 100
        else:
            momentum = 0

        momentum = safe_float(momentum)

        current_volume = safe_float(last["Volume"], 0)
        avg_volume = safe_float(last["AVG_VOL20"], 0)

        if avg_volume and avg_volume > 0:
            volume_ratio = current_volume / avg_volume
        else:
            volume_ratio = 0

        volume_ratio = safe_float(volume_ratio)

        unusual_volume = volume_ratio >= 2.0

        buy_score = 0
        sell_score = 0

        # Momentum
        if momentum is not None:
            if momentum > 0.5:
                buy_score += 25
            elif momentum > 0.2:
                buy_score += 12

            if momentum < -0.5:
                sell_score += 25
            elif momentum < -0.2:
                sell_score += 12

        # VWAP
        if vwap:
            if price > vwap:
                buy_score += 20
            else:
                sell_score += 20

        # EMA trend
        if ema20 and ema50:
            if ema20 > ema50:
                buy_score += 20
            else:
                sell_score += 20

        # RSI
        if rsi is not None:
            if 50 <= rsi <= 70:
                buy_score += 15
            elif 30 <= rsi < 50:
                sell_score += 15

        # Volume
        if unusual_volume:
            if momentum and momentum > 0:
                buy_score += 20
            elif momentum and momentum < 0:
                sell_score += 20

        if buy_score >= 70 and buy_score > sell_score:
            signal = "BUY"
            confidence = buy_score
            entry = price
            sl = price - (1.5 * atr if atr else price * 0.01)
            target = price + (3 * atr if atr else price * 0.02)

        elif sell_score >= 70 and sell_score > buy_score:
            signal = "SELL"
            confidence = sell_score
            entry = price
            sl = price + (1.5 * atr if atr else price * 0.01)
            target = price - (3 * atr if atr else price * 0.02)

        else:
            signal = "WAIT"
            confidence = max(buy_score, sell_score)
            entry = price
            sl = None
            target = None

        timestamp = df.index[-1]

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")

        entry_time = timestamp.astimezone(IST).strftime(
            "%d-%m-%Y %H:%M"
        )

        return {
            "symbol": symbol,
            "price": price,
            "momentum": momentum,
            "volume": current_volume,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "unusual_volume": unusual_volume,
            "vwap": vwap,
            "rsi": rsi,
            "ema20": ema20,
            "ema50": ema50,
            "signal": signal,
            "confidence": confidence,
            "entry": safe_float(entry),
            "sl": safe_float(sl),
            "target": safe_float(target),
            "entry_time": entry_time
        }

    except Exception as e:
        print(f"{symbol}: {e}")
        return None


def scan_market(symbols, limit=20):
    results = []

    for symbol in symbols:
        result = scan_symbol(symbol)

        if result:
            results.append(result)

    results.sort(
        key=lambda x: (
            x["confidence"],
            abs(x["momentum"] or 0),
            x["volume_ratio"] or 0
        ),
        reverse=True
    )

    return results[:limit]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "time_ist": datetime.now(IST).strftime("%d-%m-%Y %H:%M:%S")
    })


@app.route("/api/scan")
def api_scan():
    results = scan_market(CASH_SYMBOLS, 20)

    unusual = [
        x for x in results
        if x.get("unusual_volume")
    ]

    return jsonify({
        "success": True,
        "scanned": len(CASH_SYMBOLS),
        "showing": len(results),
        "scan_time_ist": datetime.now(IST).strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "stocks": results,
        "unusual_volume": unusual
    })


@app.route("/api/fno-stocks")
def api_fno_stocks():
    results = scan_market(FNO_SYMBOLS, 30)

    return jsonify({
        "success": True,
        "scan_time_ist": datetime.now(IST).strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "stocks": results
    })


@app.route("/api/options")
def api_options():
    return jsonify({
        "success": False,
        "message": (
            "Dhan option-chain API integration is ready "
            "for the next step. Add Dhan credentials in "
            "Render Environment Variables."
        ),
        "symbols": ["NIFTY", "SENSEX"]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
