import requests
import pandas as pd
import numpy as np
import json
import os
import time
import math
from datetime import datetime, timedelta

# ==========================================
# [설정] Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

FILES = {
    "wallet": os.path.join(DATA_DIR, 'wallet.json'),
    "history": os.path.join(DATA_DIR, 'trade_history.json'),
    "analysis": os.path.join(DATA_DIR, 'analysis_result.json'),
    "chart_data": os.path.join(DATA_DIR, 'ohlcv_data.json')
}

MARKET = "KRW-BTC"
FEE = 0.0005
MIN_ORDER_KRW = 6000

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def truncate(number, decimals=0):
    factor = 10.0 ** decimals
    return math.floor(number * factor) / factor

# ==========================================
# [Class 1] Data Fetcher
# ==========================================
class DataFetcher:
    def get_candles(self, unit, target_count=400):
        url = f"https://api.upbit.com/v1/candles/{unit}"
        headers = {"accept": "application/json"}
        all_data = []
        current_to = None
        
        while len(all_data) < target_count:
            req_count = min(200, target_count - len(all_data))
            params = {"market": MARKET, "count": req_count}
            if current_to: params['to'] = current_to
            
            try:
                res = requests.get(url, params=params, headers=headers)
                res.raise_for_status()
                data = res.json()
                if not data: break
                all_data.extend(data)
                current_to = data[-1]['candle_date_time_utc']
                time.sleep(0.04)
            except Exception as e:
                print(f"Error fetching {unit}: {e}")
                break
        
        if not all_data: return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df = df[['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        df = df.sort_values('time').reset_index(drop=True)
        return df

    def get_orderbook(self):
        url = "https://api.upbit.com/v1/orderbook"
        params = {"markets": MARKET}
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            data = res.json()
            return data[0]['orderbook_units']
        except Exception as e:
            print(f"Error fetching orderbook: {e}")
            return None

# ==========================================
# [Class 2] QuantAnalyzer
# ==========================================
class QuantAnalyzer:
    def calculate_indicators(self, df):
        if df.empty: return df
        
        # VWAP
        v_price = df['close'] * df['volume']
        df['VWAP'] = v_price.rolling(20).sum() / df['volume'].rolling(20).sum()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))

        # Bollinger Bands & Squeeze
        df['MA20'] = df['close'].rolling(20).mean()
        df['Std'] = df['close'].rolling(20).std()
        df['BB_Up'] = df['MA20'] + (2 * df['Std'])
        df['BB_Low'] = df['MA20'] - (2 * df['Std'])
        bb_range = (df['BB_Up'] - df['BB_Low']).replace(0, 1)
        df['PctB'] = (df['close'] - df['BB_Low']) / bb_range
        df['BandWidth'] = bb_range / df['MA20']
        df['BandWidth_Mean'] = df['BandWidth'].rolling(50).mean()
        df['Is_Squeeze'] = df['BandWidth'] < (df['BandWidth_Mean'] * 0.8)

        # MACD
        k = df['close'].ewm(span=12, adjust=False).mean()
        d = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = k - d
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # MFI
        typical = (df['high'] + df['low'] + df['close']) / 3
        flow = typical * df['volume']
        p_flow = flow.where(typical > typical.shift(1), 0).rolling(14).sum()
        n_flow = flow.where(typical < typical.shift(1), 0).rolling(14).sum()
        df['MFI'] = 100 - (100 / (1 + (p_flow / n_flow.replace(0, 1))))

        # Stochastic RSI
        min_rsi = df['RSI'].rolling(14).min()
        max_rsi = df['RSI'].rolling(14).max()
        df['Stoch_K'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi).replace(0, 1) * 100

        # ADX
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()
        up = df['high'] - df['high'].shift(1)
        down = df['low'].shift(1) - df['low']
        p_dm = np.where((up > down) & (up > 0), up, 0)
        n_dm = np.where((down > up) & (down > 0), down, 0)
        p_di = 100 * (pd.Series(p_dm).rolling(14).mean() / df['ATR'])
        n_di = 100 * (pd.Series(n_dm).rolling(14).mean() / df['ATR'])
        df['DX'] = 100 * abs(p_di - n_di) / (p_di + n_di).replace(0, 1)
        df['ADX'] = df['DX'].rolling(14).mean()

        # Ichimoku
        h9 = df['high'].rolling(9).max(); l9 = df['low'].rolling(9).min()
        h26 = df['high'].rolling(26).max(); l26 = df['low'].rolling(26).min()
        h52 = df['high'].rolling(52).max(); l52 = df['low'].rolling(52).min()
        df['Tenkan'] = (h9 + l9) / 2
        df['Kijun'] = (h26 + l26) / 2
        df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
        df['SpanB'] = ((h52 + l52) / 2).shift(26)

        # CMF
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 1) * df['volume']
        df['CMF'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()

        # Divergence Slope
        df['Price_Slope'] = df['close'].diff(3)
        df['RSI_Slope'] = df['RSI'].diff(3)
        
        return df

# ==========================================
# [Class 3] Strategy Core (Smart Decision Engine)
# ==========================================
class StrategyCore:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.analyzer = QuantAnalyzer()

    def normalize(self, value, min_v, max_v):
        return max(min((value - min_v) / (max_v - min_v) * 200 - 100, 100), -100)

    def execute_analysis(self):
        # 1. Fetching
        df_3m = self.analyzer.calculate_indicators(self.fetcher.get_candles('minutes/3', 400))
        df_5m = self.analyzer.calculate_indicators(self.fetcher.get_candles('minutes/5', 400))
        df_15m = self.analyzer.calculate_indicators(self.fetcher.get_candles('minutes/15', 400))
        df_60m = self.analyzer.calculate_indicators(self.fetcher.get_candles('minutes/60', 400))
        df_240m = self.analyzer.calculate_indicators(self.fetcher.get_candles('minutes/240', 400))
        df_d = self.analyzer.calculate_indicators(self.fetcher.get_candles('days', 400))

        if df_15m.empty or df_3m.empty: return None

        # 2. Export Chart Data
        chart_export = {
            "d": df_d.tail(200).to_dict(orient='records'),
            "h4": df_240m.tail(200).to_dict(orient='records'),
            "h1": df_60m.tail(200).to_dict(orient='records'),
            "m15": df_15m.tail(200).to_dict(orient='records'),
            "m5": df_5m.tail(200).to_dict(orient='records')
        }
        with open(FILES['chart_data'], 'w', encoding='utf-8') as f:
            json.dump(chart_export, f, cls=NpEncoder)

        # 3. Market Regime Definition
        curr_4h = df_240m.iloc[-2]
        curr_1h = df_60m.iloc[-2]
        curr_15m = df_15m.iloc[-2]
        curr_3m = df_3m.iloc[-2]
        
        regime = "Neutral"
        # 기본 가중치
        weights = {"trend": 0.25, "osc": 0.25, "vol": 0.20, "money": 0.20, "insight": 0.10}
        trade_threshold = 0.05 # 기본 거래 민감도 (5%)

        # (A) Strong Bull: 추세 + 자금 확인
        if curr_4h['ADX'] > 30 and curr_4h['close'] > curr_4h['MA20'] and curr_15m['MFI'] > 40:
            regime = "Strong Bull Trend"
            weights = {"trend": 0.40, "osc": 0.10, "vol": 0.10, "money": 0.30, "insight": 0.10}
            trade_threshold = 0.03 # 강세장에서는 민감하게 반응 (3% 차이만 나도 거래)
        
        # (B) Weak Bull: 추세만 좋음 (자금 이탈 의심)
        elif curr_4h['ADX'] > 30 and curr_4h['close'] > curr_4h['MA20']:
            regime = "Weak Bull (Money Div)"
            weights = {"trend": 0.20, "osc": 0.20, "vol": 0.20, "money": 0.40, "insight": 0.0}
            trade_threshold = 0.05

        # (C) Range/Choppy: 횡보장 (잦은 매매 방지)
        elif curr_1h['Is_Squeeze'] or curr_4h['ADX'] < 20:
            regime = "Squeeze/Range"
            weights = {"trend": 0.10, "osc": 0.40, "vol": 0.40, "money": 0.10, "insight": 0.0}
            trade_threshold = 0.10 # ★ 중요: 횡보장에서는 10% 이상 차이 나야 거래 (수수료 방어)
            
        # (D) Panic: 급락
        elif curr_15m['RSI'] < 25:
            regime = "Panic/Crash"
            weights = {"trend": 0.0, "osc": 0.30, "vol": 0.20, "money": 0.40, "insight": 0.10}
            trade_threshold = 0.05

        # 4. Score Calculation
        sc_vwap = 100 if curr_15m['close'] > curr_15m['VWAP'] else -100
        sc_cloud = 100 if curr_1h['close'] > curr_1h['SpanA'] else -50
        sc_macd = 100 if curr_4h['MACD'] > curr_4h['MACD_Signal'] else -100
        score_trend = (sc_vwap * 0.3) + (sc_cloud * 0.3) + (sc_macd * 0.4)

        sc_rsi = self.normalize(curr_15m['RSI'], 70, 30)
        if regime == "Strong Bull Trend":
             sc_rsi = self.normalize(curr_15m['RSI'], 40, 80)
        else:
             sc_rsi = (50 - curr_15m['RSI']) * 2
        sc_stoch = (50 - curr_15m['Stoch_K']) * 2
        score_osc = (sc_rsi * 0.6) + (sc_stoch * 0.4)

        sc_bb = (0.5 - curr_15m['PctB']) * 200
        score_vol = max(min(sc_bb, 100), -100)

        sc_mfi = (50 - curr_15m['MFI']) * 2
        sc_cmf = self.normalize(curr_15m['CMF'], -0.2, 0.2)
        score_money = (sc_mfi * 0.5) + (sc_cmf * 0.5)

        # Insight Score
        score_insight = 0
        if curr_3m['Price_Slope'] < 0 and curr_3m['RSI_Slope'] > 0: score_insight += 60 # Bull Div
        elif curr_3m['Price_Slope'] > 0 and curr_3m['RSI_Slope'] < 0: score_insight -= 60 # Bear Div
        if df_5m.iloc[-2]['close'] > df_5m.iloc[-2]['MA20']: score_insight += 30 # Micro Trend
        score_insight = max(min(score_insight, 100), -100)

        # 5. Final Score & Penalties
        final_score = (score_trend * weights['trend']) + \
                      (score_osc * weights['osc']) + \
                      (score_vol * weights['vol']) + \
                      (score_money * weights['money']) + \
                      (score_insight * weights['insight'])
        
        # [Penalty 1] Money Flow Veto: 점수 좋아도 돈 빠지면 삭감
        if final_score > 0 and curr_15m['MFI'] < 30:
            final_score *= 0.5
            regime += " (Money Penalty)"

        final_score = round(max(min(final_score, 100), -100), 2)

        # 6. Target Ratio Logic (Dynamic Sizing)
        target_ratio = 0.0
        opinion = "Neutral"
        
        if final_score > 20:
            target_ratio = (final_score - 20) / (100 - 20)
        else:
            target_ratio = 0.0

        # [Logic 1] Regime-based Cap
        if regime.startswith("Strong Bull"):
            if target_ratio > 0: target_ratio = max(target_ratio, 0.3)
        elif "Weak Bull" in regime or "Range" in regime:
            target_ratio = min(target_ratio, 0.5)
        elif "Panic" in regime:
            target_ratio = min(target_ratio, 0.2)

        # [Logic 2] Profit Taking (RSI Overheated)
        # 3분봉 or 15분봉 RSI가 80 이상이면 강제로 비중 축소 (익절 유도)
        if curr_3m['RSI'] > 80 or curr_15m['RSI'] > 80:
            target_ratio = min(target_ratio, 0.2) # 20%만 남기고 다 팔아라
            regime += " (Overheated)"
            opinion = "Profit Take"

        # [Logic 3] Emergency Exit (Flash Crash)
        # 3분봉이 -2% 이상 급락하면 비중 0으로 (손절/회피)
        last_ret_3m = (curr_3m['close'] - curr_3m['open']) / curr_3m['open']
        is_emergency = False
        if last_ret_3m < -0.02:
            target_ratio = 0.0
            is_emergency = True
            regime = "Flash Crash!"
            opinion = "Emergency Sell"

        target_ratio = round(max(min(target_ratio, 0.99), 0.0), 2)

        if not is_emergency and opinion != "Profit Take":
            if final_score >= 80: opinion = "Strong Buy"
            elif final_score >= 50: opinion = "Buy"
            elif final_score >= 20: opinion = "Weak Buy"
            elif final_score >= -20: opinion = "Neutral"
            elif final_score >= -60: opinion = "Sell"
            else: opinion = "Strong Sell"

        # Logging Data
        factors_log = {
            "trend": { "score": round(score_trend), "weight": weights['trend'] },
            "oscillator": { "score": round(score_osc), "weight": weights['osc'] },
            "volatility": { "score": round(score_vol), "weight": weights['vol'] },
            "money_flow": { "score": round(score_money), "weight": weights['money'] },
            "insight": { "score": round(score_insight), "weight": weights['insight'] }
        }
        
        display_reasons = [
            f"Regime: {regime}",
            f"Trend: {round(score_trend)} (Wt: {weights['trend']})",
            f"Oscillator: {round(score_osc)} (Wt: {weights['osc']})",
            f"Vol: {round(score_vol)} (Wt: {weights['vol']})",
            f"Money: {round(score_money)} (Wt: {weights['money']})",
            f"Insight: {round(score_insight)} (Wt: {weights['insight']})"
        ]

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": df_15m.iloc[-1]['close'],
            "score": final_score,
            "score_breakdown": {
                "trend": round(score_trend), "oscillator": round(score_osc),
                "volatility": round(score_vol), "money": round(score_money), "insight": round(score_insight)
            },
            "target_ratio": target_ratio,
            "opinion": opinion,
            "regime": regime,
            "factors": factors_log,
            "reasons": display_reasons,
            "weights": weights,
            "trade_threshold": trade_threshold, # 동적 임계값 전달
            "is_emergency": is_emergency
        }

# ==========================================
# [Class 4] Asset Manager
# ==========================================
class AssetManager:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.load_data()

    def load_data(self):
        try:
            if os.path.exists(FILES['wallet']):
                with open(FILES['wallet'], 'r') as f: self.wallet = json.load(f)
            else: raise FileNotFoundError
        except: self.wallet = {"krw": 1000000, "btc": 0.0, "avg_price": 0.0, "net_equity": 1000000}
        
        try:
            if os.path.exists(FILES['history']):
                with open(FILES['history'], 'r') as f: self.history = json.load(f)
            else: self.history = []
        except: self.history = []

    def save_data(self, analysis):
        with open(FILES['wallet'], 'w', encoding='utf-8') as f:
            json.dump(self.wallet, f, indent=4, cls=NpEncoder)
        with open(FILES['history'], 'w', encoding='utf-8') as f:
            json.dump(self.history[-300:], f, indent=4, cls=NpEncoder)
        with open(FILES['analysis'], 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=4, cls=NpEncoder)

    def calculate_weighted_price(self, side, amount):
        ob_units = self.data_fetcher.get_orderbook()
        if not ob_units: return None, 0

        avg_price = 0
        total_qty = 0
        remain_amt = amount

        if side == 'buy':
            spent_krw = 0
            acquired_btc = 0
            for unit in ob_units:
                price = unit['ask_price']
                size = unit['ask_size']
                cost = price * size
                if remain_amt <= cost:
                    buy_vol = remain_amt / price
                    acquired_btc += buy_vol
                    spent_krw += remain_amt
                    remain_amt = 0
                    break
                else:
                    acquired_btc += size
                    spent_krw += cost
                    remain_amt -= cost
            if acquired_btc > 0:
                avg_price = spent_krw / acquired_btc
                total_qty = truncate(acquired_btc, 8) 
            return avg_price, total_qty

        else:
            gained_krw = 0
            sold_btc = 0
            for unit in ob_units:
                price = unit['bid_price']
                size = unit['bid_size']
                if remain_amt <= size:
                    gained_krw += remain_amt * price
                    sold_btc += remain_amt
                    remain_amt = 0
                    break
                else:
                    gained_krw += size * price
                    sold_btc += size
                    remain_amt -= size
            if sold_btc > 0:
                avg_price = gained_krw / sold_btc
                total_qty = truncate(sold_btc, 8)
            return avg_price, total_qty

    def mark_to_market(self):
        ob = self.data_fetcher.get_orderbook()
        if ob:
            current_bid = ob[0]['bid_price']
            btc_value = self.wallet['btc'] * current_bid
            self.wallet['net_equity'] = self.wallet['krw'] + (btc_value * (1 - FEE))

    def rebalance(self, analysis):
        if not analysis: return
        target_ratio = analysis['target_ratio']
        trade_threshold = analysis['trade_threshold'] # 동적 임계값 사용
        is_emergency = analysis['is_emergency']

        self.mark_to_market()
        total_equity = self.wallet['net_equity']
        
        ob = self.data_fetcher.get_orderbook()
        if not ob: return
        ref_price = ob[0]['bid_price']

        current_btc_val = self.wallet['btc'] * ref_price
        current_ratio = current_btc_val / total_equity if total_equity > 0 else 0
        
        # [중요] No-Trade Zone Logic
        # 긴급 탈출(Emergency)이 아니고, 변화량이 임계값(Threshold) 미만이면 거래 스킵
        if not is_emergency and abs(target_ratio - current_ratio) < trade_threshold:
            print(f" >> [Skip] Change {abs(target_ratio - current_ratio)*100:.2f}% < {trade_threshold*100}% Threshold")
            analysis['current_price'] = ref_price
            self.save_data(analysis)
            return

        target_btc_val = total_equity * target_ratio
        diff_krw = target_btc_val - current_btc_val
        
        action = "HOLD"
        trade_log = None

        # BUY Logic
        if diff_krw > MIN_ORDER_KRW and self.wallet['krw'] >= diff_krw:
            buy_budget = diff_krw
            real_budget = buy_budget / (1 + FEE) 
            avg_price, amount_btc = self.calculate_weighted_price('buy', real_budget)
            
            if amount_btc > 0:
                actual_cost = amount_btc * avg_price
                fee_cost = actual_cost * FEE
                total_spend = actual_cost + fee_cost

                prev_cost = self.wallet['btc'] * self.wallet['avg_price']
                self.wallet['btc'] += amount_btc
                self.wallet['avg_price'] = (prev_cost + actual_cost) / self.wallet['btc']
                self.wallet['krw'] -= total_spend
                
                action = "BUY"
                trade_log = {
                    "time": analysis['timestamp'], "type": "BUY", "price": avg_price, 
                    "amount": total_spend, "volume": amount_btc, 
                    "score": analysis['score'], "regime": analysis['regime'],
                    "reason": str(analysis['reasons'])
                }

        # SELL Logic
        elif diff_krw < -MIN_ORDER_KRW:
            sell_amt_krw = abs(diff_krw)
            est_sell_vol = sell_amt_krw / ref_price
            real_sell_vol = min(est_sell_vol, self.wallet['btc'])
            real_sell_vol = truncate(real_sell_vol, 8)

            if (real_sell_vol * ref_price) >= MIN_ORDER_KRW:
                avg_price, sold_vol = self.calculate_weighted_price('sell', real_sell_vol)
                
                if sold_vol > 0:
                    gross_income = sold_vol * avg_price
                    fee_cost = gross_income * FEE
                    net_income = gross_income - fee_cost

                    self.wallet['btc'] -= sold_vol
                    self.wallet['krw'] += net_income
                    if self.wallet['btc'] < 0.00000001: 
                        self.wallet['btc'] = 0; self.wallet['avg_price'] = 0
                    
                    action = "SELL"
                    trade_log = {
                        "time": analysis['timestamp'], "type": "SELL", "price": avg_price, 
                        "amount": net_income, "volume": sold_vol, 
                        "score": analysis['score'], "regime": analysis['regime'],
                        "reason": str(analysis['reasons'])
                    }

        self.mark_to_market()

        if action != "HOLD" and trade_log:
            trade_log['balance_after'] = self.wallet['net_equity']
            print(f" >> [Trade] {action} {trade_log['volume']} BTC @ {trade_log['price']:,.0f} KRW")
            self.history.append(trade_log)
        else:
            print(f" >> [Hold] {analysis['regime']} | Score:{analysis['score']:.1f} | Ratio:{current_ratio*100:.1f}%->{target_ratio*100:.1f}%")

        analysis['current_price'] = ref_price
        self.save_data(analysis)

if __name__ == "__main__":
    print(f"======== [Alpha-Pro V15.0 Final] {datetime.now()} ========")
    try:
        core = StrategyCore()
        result = core.execute_analysis()
        if result:
            am = AssetManager()
            am.rebalance(result)
        else:
            print(" [Error] Not enough data.")
    except Exception as e:
        print(f" [Critical Error] {e}")
