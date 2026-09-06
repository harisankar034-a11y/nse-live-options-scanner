import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template

app = Flask(__name__)

IST = ZoneInfo("Asia/Kolkata")


# =========================
# F&O STOCKS
# =========================

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


# =========================
# LARGE CAP
# =========================

LARGE_CAP = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "BHARTIARTL",
    "TCS", "INFY", "SBIN", "LICI", "HINDUNILVR",
    "ITC", "BAJFINANCE", "LT", "MARUTI", "AXISBANK",
    "KOTAKBANK", "SUNPHARMA", "M&M", "HCLTECH",
    "TITAN", "ADANIENT", "ADANIPORTS", "NTPC",
    "POWERGRID", "ONGC", "TATAMOTORS", "ULTRACEMCO",
    "WIPRO", "NESTLEIND", "ASIANPAINT", "COALINDIA",
    "BAJAJFINSV", "HINDZINC", "JSWSTEEL", "TATASTEEL",
    "BEL", "ADANIGREEN", "HDFCLIFE", "SBILIFE"
]


# =========================
# MID CAP
# =========================

MID_CAP = [
    "ABB", "ACC", "AMBUJACEM", "APOLLOHOSP",
    "BANKINDIA", "BIOCON", "BOSCHLTD", "BRITANNIA",
    "CHOLAFIN", "CUMMINSIND", "DABUR", "DEEPAKNTR",
    "DELHIVERY", "GAIL", "GODREJCP", "GRASIM",
    "HAVELLS", "ICICIGI", "IDFCFIRSTB", "INDHOTEL",
    "INDIGO", "IOC", "IRCTC", "JINDALSTEL",
    "LICHSGFIN", "LUPIN", "MARICO", "MAXHEALTH",
    "MUTHOOTFIN", "NAUKRI", "PAGEIND", "PIDILITIND",
    "PERSISTENT", "PFC", "RECLTD", "SAIL",
    "SIEMENS", "SRF", "TATACONSUM", "TORNTPHARM",
    "TRENT", "TVSMOTOR", "UBL", "VEDL", "YESBANK",
    "ZYDUSLIFE"
]


# =========================
# SMALL CAP
# =========================

SMALL_CAP = [
    "AARTIIND", "AFFLE", "ALKEM", "AMARAJABAT",
    "ANANDRATHI", "ANGELONE", "ASTRAL", "BANDHANBNK",
    "BATAINDIA", "BIRLACORPN", "BLS", "CANFINHOME",
    "CDSL", "CENTRALBK", "CESC", "CROMPTON",
    "CYIENT", "EDELWEISS", "EQUITASBNK", "EXIDEIND",
    "FINEORG", "FORTIS", "GNFC", "GRAPHITE",
    "GSPL", "HINDCOPPER", "IDBI", "IEX",
    "IRB", "IRCON", "JBCHEPHARM", "JINDALSAW",
    "KALYANKJIL", "KEI", "KFINTECH", "LAURUSLABS",
    "MANAPPURAM", "MCX", "NATIONALUM", "NBCC",
    "NLCINDIA", "OLECTRA", "PNCINFRA", "RITES",
    "ROUTE", "RVNL", "SONACOMS", "SUZLON",
    "TATACHEM", "TATATECH", "UCOBANK", "UNOMINDA"
]


# =========================
# ALL CASH STOCKS
# =========================

CASH_SYMBOLS = list(dict.fromkeys(
    LARGE_CAP + MID_CAP + SMALL_CAP
))


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

    return 100 - (100 / (1 + rs))


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
    """
    Early-warning scanner:
    EARLY BUY/SELL = setup building before a stronger move.
    BUY/SELL = stronger confirmed move.
    Only current and historical candles are used.
    """
    try:
        ticker = yf.Ticker(nse_ticker(symbol))
        df = ticker.history(period="5d", interval="5m", auto_adjust=False)

        if df is None or df.empty:
            return None

        df = flatten_columns(df)
        required = ["Open", "High", "Low", "Close", "Volume"]

        if not all(col in df.columns for col in required):
            return None

        df = df.dropna(subset=required).copy()

        if len(df) < 60:
            return None

        close = df["Close"]
        volume = df["Volume"]

        df["RSI"] = calculate_rsi(close)
        df["ATR"] = calculate_atr(df)
        df["EMA9"] = close.ewm(span=9, adjust=False).mean()
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()

        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (typical_price * volume).cumsum() / volume.cumsum()

        # Exclude current candle from the volume baseline and breakout levels.
        df["AVG_VOL20"] = volume.shift(1).rolling(20).mean()
        df["BREAKOUT_HIGH"] = df["High"].shift(1).rolling(12).max()
        df["BREAKDOWN_LOW"] = df["Low"].shift(1).rolling(12).min()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        price = safe_float(last["Close"])
        rsi = safe_float(last["RSI"])
        prev_rsi = safe_float(prev["RSI"])
        atr = safe_float(last["ATR"])
        vwap = safe_float(last["VWAP"])
        ema9 = safe_float(last["EMA9"])
        ema20 = safe_float(last["EMA20"])
        ema50 = safe_float(last["EMA50"])

        if price is None:
            return None

        previous_close = safe_float(prev["Close"])
        momentum = ((price - previous_close) / previous_close) * 100 if previous_close else 0
        momentum = safe_float(momentum)

        current_volume = safe_float(last["Volume"], 0)
        avg_volume = safe_float(last["AVG_VOL20"], 0)
        volume_ratio = (current_volume / avg_volume) if avg_volume else 0
        volume_ratio = safe_float(volume_ratio)

        recent_vol = volume.iloc[-3:].mean()
        prior_vol = volume.iloc[-13:-3].mean()
        volume_build_ratio = (recent_vol / prior_vol) if prior_vol else 0
        volume_build_ratio = safe_float(volume_build_ratio)

        unusual_volume = volume_ratio >= 2.0

        breakout_level = safe_float(last["BREAKOUT_HIGH"])
        breakdown_level = safe_float(last["BREAKDOWN_LOW"])

        breakout_distance_atr = None
        breakdown_distance_atr = None
        if atr and atr > 0:
            if breakout_level:
                breakout_distance_atr = safe_float((breakout_level - price) / atr)
            if breakdown_level:
                breakdown_distance_atr = safe_float((price - breakdown_level) / atr)

        ema20_prev = safe_float(df["EMA20"].iloc[-4])
        ema20_slope_pct = ((ema20 - ema20_prev) / ema20_prev * 100) if ema20_prev else 0
        ema20_slope_pct = safe_float(ema20_slope_pct)

        recent_high = safe_float(df["High"].iloc[-12:].max())
        recent_low = safe_float(df["Low"].iloc[-12:].min())
        range_atr = None
        if atr and atr > 0 and recent_high is not None and recent_low is not None:
            range_atr = safe_float((recent_high - recent_low) / atr)

        buy_score = 0
        sell_score = 0
        early_buy_score = 0
        early_sell_score = 0
        buy_reasons = []
        sell_reasons = []

        # Confirmed move scoring
        if momentum is not None:
            if momentum > 0.5:
                buy_score += 25
            elif momentum > 0.2:
                buy_score += 12
            if momentum < -0.5:
                sell_score += 25
            elif momentum < -0.2:
                sell_score += 12

        if vwap:
            if price > vwap:
                buy_score += 20
            else:
                sell_score += 20

        if ema20 and ema50:
            if ema20 > ema50:
                buy_score += 20
            else:
                sell_score += 20

        if rsi is not None:
            if 50 <= rsi <= 70:
                buy_score += 15
            elif 30 <= rsi < 50:
                sell_score += 15

        if unusual_volume:
            if momentum and momentum > 0:
                buy_score += 20
            elif momentum and momentum < 0:
                sell_score += 20

        # Early-warning scoring: does not require 2x volume or a big move.
        if vwap and price >= vwap * 0.998:
            early_buy_score += 15
            buy_reasons.append("above/near VWAP")
        if vwap and price <= vwap * 1.002:
            early_sell_score += 15
            sell_reasons.append("below/near VWAP")

        if ema9 and ema20 and ema50:
            if ema9 >= ema20 and ema20 >= ema50:
                early_buy_score += 20
                buy_reasons.append("EMA trend improving")
            elif ema9 <= ema20 and ema20 <= ema50:
                early_sell_score += 20
                sell_reasons.append("EMA trend weakening")

        if ema20_slope_pct is not None:
            if ema20_slope_pct > 0.03:
                early_buy_score += 15
                buy_reasons.append("EMA20 rising")
            elif ema20_slope_pct < -0.03:
                early_sell_score += 15
                sell_reasons.append("EMA20 falling")

        if rsi is not None and prev_rsi is not None:
            if 45 <= rsi <= 62 and rsi > prev_rsi:
                early_buy_score += 20
                buy_reasons.append("RSI rising")
            elif 38 <= rsi <= 55 and rsi < prev_rsi:
                early_sell_score += 20
                sell_reasons.append("RSI falling")

        if volume_ratio >= 1.15:
            early_buy_score += 10
            early_sell_score += 10

        if volume_build_ratio >= 1.15:
            if momentum is not None and momentum >= 0:
                early_buy_score += 15
                buy_reasons.append("volume building")
            if momentum is not None and momentum <= 0:
                early_sell_score += 15
                sell_reasons.append("volume building")

        if (breakout_distance_atr is not None and 0 <= breakout_distance_atr <= 0.75
                and range_atr is not None and range_atr <= 6):
            early_buy_score += 20
            buy_reasons.append("near breakout")

        if (breakdown_distance_atr is not None and 0 <= breakdown_distance_atr <= 0.75
                and range_atr is not None and range_atr <= 6):
            early_sell_score += 20
            sell_reasons.append("near breakdown")

        if buy_score >= 70 and buy_score > sell_score:
            signal = "BUY"
            confidence = buy_score
            signal_type = "CONFIRMED"
            entry = price
            sl = price - (1.5 * atr if atr else price * 0.01)
            target = price + (3 * atr if atr else price * 0.02)

        elif sell_score >= 70 and sell_score > buy_score:
            signal = "SELL"
            confidence = sell_score
            signal_type = "CONFIRMED"
            entry = price
            sl = price + (1.5 * atr if atr else price * 0.01)
            target = price - (3 * atr if atr else price * 0.02)

        elif early_buy_score >= 60 and early_buy_score > early_sell_score:
            signal = "EARLY BUY"
            confidence = early_buy_score
            signal_type = "EARLY"
            entry = price
            sl = price - (1.0 * atr if atr else price * 0.0075)
            target = price + (2.0 * atr if atr else price * 0.015)

        elif early_sell_score >= 60 and early_sell_score > early_buy_score:
            signal = "EARLY SELL"
            confidence = early_sell_score
            signal_type = "EARLY"
            entry = price
            sl = price + (1.0 * atr if atr else price * 0.0075)
            target = price - (2.0 * atr if atr else price * 0.015)

        else:
            signal = "WAIT"
            confidence = max(buy_score, sell_score, early_buy_score, early_sell_score)
            signal_type = "WATCH"
            entry = price
            sl = None
            target = None

        timestamp = df.index[-1]
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")

        entry_time = timestamp.astimezone(IST).strftime("%d-%m-%Y %H:%M")

        return {
            "symbol": symbol,
            "price": price,
            "momentum": momentum,
            "volume": current_volume,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "volume_build_ratio": volume_build_ratio,
            "unusual_volume": unusual_volume,
            "vwap": vwap,
            "rsi": rsi,
            "prev_rsi": prev_rsi,
            "ema9": ema9,
            "ema20": ema20,
            "ema50": ema50,
            "ema20_slope_pct": ema20_slope_pct,
            "breakout_level": breakout_level,
            "breakdown_level": breakdown_level,
            "breakout_distance_atr": breakout_distance_atr,
            "breakdown_distance_atr": breakdown_distance_atr,
            "range_atr": range_atr,
            "early_buy_score": early_buy_score,
            "early_sell_score": early_sell_score,
            "setup_score": max(early_buy_score, early_sell_score),
            "signal": signal,
            "signal_type": signal_type,
            "confidence": confidence,
            "signal_reason": "; ".join(
                buy_reasons if signal in ("EARLY BUY", "BUY") else sell_reasons
            ),
            "entry": safe_float(entry),
            "sl": safe_float(sl),
            "target": safe_float(target),
            "entry_time": entry_time
        }

    except Exception as e:
        print(f"{symbol}: {e}")
        return None


def add_category(results):

    large = set(LARGE_CAP)
    mid = set(MID_CAP)
    small = set(SMALL_CAP)
    fno = set(FNO_SYMBOLS)

    for item in results:

        symbol = item["symbol"]

        if symbol in large:
            item["category"] = "Large Cap"

        elif symbol in mid:
            item["category"] = "Mid Cap"

        elif symbol in small:
            item["category"] = "Small Cap"

        else:
            item["category"] = "Other"

        item["fno"] = symbol in fno

    return results


def scan_market(symbols, limit=30):

    results = []

    for symbol in symbols:

        result = scan_symbol(symbol)

        if result:
            results.append(result)

    results = add_category(results)

    signal_priority = {
        "BUY": 4,
        "SELL": 4,
        "EARLY BUY": 3,
        "EARLY SELL": 3,
        "WAIT": 1
    }

    results.sort(
        key=lambda x: (
            signal_priority.get(x.get("signal"), 0),
            x.get("setup_score", 0),
            x["confidence"],
            x.get("volume_build_ratio", 0) or 0,
            abs(x["momentum"] or 0)
        ),
        reverse=True
    )

    return results[:limit]


# =========================
# HOME
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# HEALTH
# =========================

@app.route("/api/health")
def health():

    return jsonify({

        "status": "ok",

        "time_ist": datetime.now(
            IST
        ).strftime(
            "%d-%m-%Y %H:%M:%S"
        )
    })


# =========================
# ALL CASH
# =========================

@app.route("/api/scan")
def api_scan():

    results = scan_market(
        CASH_SYMBOLS,
        50
    )

    unusual = [
        x for x in results
        if x.get("unusual_volume")
    ]

    return jsonify({

        "success": True,

        "scanned": len(CASH_SYMBOLS),

        "showing": len(results),

        "scan_time_ist": datetime.now(
            IST
        ).strftime(
            "%d-%m-%Y %H:%M:%S"
        ),

        "stocks": results,

        "unusual_volume": unusual
    })


# =========================
# LARGE CAP
# =========================

@app.route("/api/large-cap")
def api_large_cap():

    results = scan_market(
        LARGE_CAP,
        40
    )

    return jsonify({

        "success": True,

        "category": "Large Cap",

        "scanned": len(LARGE_CAP),

        "stocks": results,

        "scan_time_ist": datetime.now(
            IST
        ).strftime(
            "%d-%m-%Y %H:%M:%S"
        )
    })


# =========================
# MID CAP
# =========================

@app.route("/api/mid-cap")
def api_mid_cap():

    results = scan_market(
        MID_CAP,
        40
    )

    return jsonify({

        "success": True,

        "category": "Mid Cap",

        "scanned": len(MID_CAP),

        "stocks": results,

        "scan_time_ist": datetime.now(
            IST
        ).strftime(
            "%d-%m-%Y %H:%M:%S"
        )
    })


# =========================
# SMALL CAP
# =========================

@app.route("/api/small-cap")
def api_small_cap():

    results = scan_market(
        SMALL_CAP,
        40
    )

    return jsonify({

        "success": True,

        "category": "Small Cap",

        "scanned": len(SMALL_CAP),

        "stocks": results,

        "scan_time_ist": datetime.now(
            IST
        ).strftime(
            "%d-%m-%Y %H:%M:%S"
        )
    })


# =========================
# F&O
# =========================

@app.route("/api/fno-stocks")
def api_fno_stocks():

    results = scan_market(
        FNO_SYMBOLS,
        40
    )

    return jsonify({

        "success": True,

        "category": "F&O",

        "scanned": len(FNO_SYMBOLS),

        "stocks": results,

        "scan_time_ist": datetime.now(
            IST
        ).strftime(
            "%d-%m-%Y %H:%M:%S"
        )
    })


# =========================
# UNUSUAL VOLUME
# =========================

@app.route("/api/unusual-volume")
def api_unusual_volume():

    results = scan_market(
        CASH_SYMBOLS,
        100
    )

    unusual = [
        x for x in results
        if x.get("unusual_volume")
    ]

    unusual.sort(
        key=lambda x: (
            x["volume_ratio"] or 0
        ),
        reverse=True
    )

    return jsonify({

        "success": True,

        "category": "Unusual Volume",

        "stocks": unusual,

        "scan_time_ist": datetime.now(
            IST
        ).strftime(
            "%d-%m-%Y %H:%M:%S"
        )
    })


# =========================
# OPTIONS - DHAN LATER
# =========================

@app.route("/api/options")
def api_options():

    return jsonify({

        "success": False,

        "message": (
            "Dhan option-chain integration "
            "will be added next."
        ),

        "symbols": [
            "NIFTY",
            "SENSEX"
        ]
    })


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
