# Libraries
import pandas as pd
import numpy as np
from datetime import date

#Pandas display Options
pd.set_option('display.max_columns', None)

from tvDatafeed import TvDatafeed, Interval

timeframe_tv = {
    'MN1': Interval.in_monthly,
    'W1': Interval.in_weekly,
    'D1': Interval.in_daily,
    'H4': Interval.in_4_hour,
    'H1': Interval.in_1_hour,
    'M15': Interval.in_15_minute,
    'M5': Interval.in_5_minute,
    'M1': Interval.in_1_minute,
    # None: Interval.in_daily,
}

def get_data(symbol, timeframe, exchange):

    tv = TvDatafeed()
    #   exchange='NASDAQ'
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=timeframe_tv[timeframe],
                        n_bars=6000)  # Interval.in_1_hour
    # create DataFrame out of the obtained data
    df = pd.DataFrame(df).reset_index()
    # # convert time in seconds into the datetime format
    df['time'] = pd.to_datetime(df.index, unit='s')
    #df['time'] = [i + datetime.timedelta(days=1) for i in df['time']]
    #df.index = df.time.values
    df = df.drop(["time",], axis=1)  # "open", "high", "low"
    df = df.rename(columns={"open": "Open",
                          "close": "Close",
                          "high": "High",
                          "low": "Low",
                           "volume": "Volume",
                           "symbol": "Ticker"})
    df.dropna(inplace=True)

    return df

def signal(df, buy_con_type, buy_con, buy_ind1, buy_ind2, sell_con_type, sell_con, sell_ind1, sell_ind2):
#     buy_conditions = 
#     sell_conditions = df['trend_ema_fast'] < df['trend_ema_slow'] and df['momentum_rsi'] > 60.0

    if buy_con_type == 'Indicator':
    
        if buy_con == 'equal to':
             df['signal'] = np.where((df[buy_ind1] == df[buy_ind2]) , 'Buy', 'Neutral')
        elif buy_con == 'greater than' :
             df['signal'] = np.where((df[buy_ind1] > df[buy_ind2]) , 'Buy', 'Neutral')
        elif buy_con == 'less than' :
             df['signal'] = np.where((df[buy_ind1] < df[buy_ind2]) , 'Buy', 'Neutral')
        elif buy_con == 'greater than or equal to' :
             df['signal'] = np.where((df[buy_ind1] >= df[buy_ind2]) , 'Buy', 'Neutral')
        elif buy_con == 'less than or equal to' :
             df['signal'] = np.where((df[buy_ind1] <= df[buy_ind2]) , 'Buy', 'Neutral')  
    else:
        
        if buy_con == 'equal to':
             df['signal'] = np.where((df[buy_ind1] == buy_ind2) , 'Buy', 'Neutral')
        elif buy_con == 'greater than' :
             df['signal'] = np.where((df[buy_ind1] > buy_ind2) , 'Buy', 'Neutral')
        elif buy_con == 'less than' :
             df['signal'] = np.where((df[buy_ind1] < buy_ind2) , 'Buy', 'Neutral')
        elif buy_con == 'greater than or equal to' :
             df['signal'] = np.where((df[buy_ind1] >= buy_ind2) , 'Buy', 'Neutral')
        elif buy_con == 'less than or equal to' :
             df['signal'] = np.where((df[buy_ind1] <= buy_ind2) , 'Buy', 'Neutral')
        
    if sell_con_type == 'Indicator':
        if sell_con == 'equal to':
             df['signal'] = np.where((df[sell_ind1] == df[sell_ind2]) , 'Buy', 'Neutral')
        elif sell_con == 'greater than' :
             df['signal'] = np.where((df[sell_ind1] > df[sell_ind2]) , 'Buy', 'Neutral')
        elif sell_con == 'less than' :
             df['signal'] = np.where((df[sell_ind1] < df[sell_ind2]) , 'Buy', 'Neutral')
        elif sell_con == 'greater than or equal to' :
             df['signal'] = np.where((df[sell_ind1] >= df[sell_ind2]) , 'Buy', 'Neutral')
        elif sell_con == 'less than or equal to' :
             df['signal'] = np.where((df[sell_ind1] <= df[sell_ind2]) , 'Buy', 'Neutral')
                
    else:
        if sell_con == 'equal to':
             df['signal'] = np.where((df[sell_ind1] == sell_ind2) , 'Buy', 'Neutral')
        elif sell_con == 'greater than' :
             df['signal'] = np.where((df[sell_ind1] > sell_ind2) , 'Buy', 'Neutral')
        elif sell_con == 'less than' :
             df['signal'] = np.where((df[sell_ind1] < sell_ind2) , 'Buy', 'Neutral')
        elif sell_con == 'greater than or equal to' :
             df['signal'] = np.where((df[sell_ind1] >= sell_ind2) , 'Buy', 'Neutral')
        elif sell_con == 'less than or equal to' :
             df['signal'] = np.where((df[sell_ind1] <= sell_ind2) , 'Buy', 'Neutral')
                      
            
#         df['signal'] = np.where((df[buy_ind1] > df[buy_ind2]) , 'Buy', 'Neutral') #& (df['momentum_rsi'] > 60.0)
#         df['signal'] = np.where((df[sell_ind1] < df[sell_ind2]), 'Sell', df['signal'])
    
    return df

def calc_metrics(backtest_result):
    returns = backtest_result['returns']
    metric = {
        'Cumulative Returns' : returns.compsum().iloc[-1] *100,
        'CAGR%' : returns.cagr() *100,
        'Win Rate': returns.win_rate()*100,
        'Win Loss Ratio': returns.win_loss_ratio(),
        'Consecutive Wins': returns.consecutive_wins(),
        'Consecutive Losses': returns.consecutive_losses(),
        'Risk Return Ratio': returns.risk_return_ratio(),
        'Sharpe Ratio' : returns.sharpe(),
        'Sortino Ratio' : returns.sortino(),
        'Max Drawdown' : returns.max_drawdown() * 100,
    #     'Longest DD Days' : returns.sharpe(),
    #     'Average DD' : returns.sharpe(),
    #     'Average DD Days' : returns.sharpe(),
        'Recovery Factor' : returns.recovery_factor(),
        'Calmar' : returns.calmar(),
        'Skew' : returns.skew(),
        'Kurtosis' : returns.kurtosis(),
    #     'Expected Daily %' : returns.sharpe(),
    #     'Expected Monthy %' : returns.monthly_returns(),
    #     'Expected Yearly %' : returns.sharpe(),
        'Kelly Criterion' : returns.kelly_criterion() * 100,
        'Daily VaR' : returns.var() *100,
        'Profit Factor' : returns.profit_factor(),
        'Profit Ratio' : returns.profit_ratio(),
    #     'Win Days %' : returns.sharpe(),
    #     'Win Month %' : returns.sharpe(),
    #     'Win Quarter %' : returns.sharpe(),
    #     'Win Year %' : returns.sharpe(),

    }

    metrics = pd.DataFrame.from_dict(metric, orient='index')
    metrics = metrics.reset_index()
    metrics.columns = ['Metrics', 'Values']
    return metrics

# symbol = 'EURUSD'
# timeframe = 'D1'
# # timef = mt5.TIMEFRAME_D1
# EXCHANGE = 'OANDA'
# df = get_data(symbol, timeframe, EXCHANGE)
# # df = get_data(symbol, timef)
# df = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume')
# df



# class Position contain data about trades opened/closed during the backtest 
class Position:
    def __init__(self, open_datetime, open_price, order_type, volume, sl, tp, com):
        self.open_datetime = open_datetime
        self.open_price = open_price
        self.order_type = order_type
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.com = com
        self.close_datetime = None
        self.close_price = None
        self.profit = 0
        self.returns = 0
        self.net_profit = self.profit + self.com
        self.status = 'open'
        self.closedonce = False
        
    def close_position(self, close_datetime, close_price):
        self.close_datetime = close_datetime
        self.close_price = close_price
        self.profit = (self.close_price - self.open_price) * self.volume if self.order_type == 'buy' \
                                                                        else (self.open_price - self.close_price) * self.volume
        
        self.returns = round(((self.close_price - self.open_price) / self.open_price), 5)
#         self.returns = np.log(self.close_price/self.open_price)
        self.status = 'closed'
        
        
    def _asdict(self):

        return {
            'open_datetime': self.open_datetime,
            'open_price': self.open_price,
            'order_type': self.order_type,
            'volume': self.volume,
            'sl': self.sl,
            'tp': self.tp,
#             'tp_half': self.tp_half,
            'close_datetime': self.close_datetime,
            'close_price': self.close_price,
            'returns': self.returns,
            'profit': self.profit,
            'com' : self.com,
            'Net Profit' : self.profit + self.com,
            'status': self.status,
#             'closed_once' : self.closedonce
        }
        

# class Strategy defines trading logic and evaluates the backtest based on opened/closed positions
class Strategy:
    def __init__(self, df, starting_balance, sl_pips, tp_pips):
        self.starting_balance = starting_balance
        self.positions = []
        self.data = df
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        
    # return backtest result
    def get_positions_df(self):
        df = pd.DataFrame([position._asdict() for position in self.positions])
        df['pnl'] = df['Net Profit'].cumsum() + self.starting_balance
        return df
    
    # add Position class to list
    def add_position(self, position):
        self.positions.append(position)
        return True
    
    # close half positions when trade goes 10pips in profit
    def close_tp_sl(self, data):
        for pos in self.positions:
            if pos.status == 'open':

                if (pos.tp <= data.Close and pos.order_type == 'buy'):
                    pos.close_position(data.datetime, pos.tp)
                    
                elif (pos.tp >= data.Close and pos.order_type == 'sell'):
                    pos.close_position(data.datetime, pos.tp)
                                        
                elif (pos.sl >= data.Close and pos.order_type == 'buy'):
                    pos.close_position(data.datetime, pos.sl)
                    
                elif (pos.sl <= data.Close and pos.order_type == 'sell'):
                    pos.close_position(data.datetime, pos.sl)
                    
                elif data['datetime'].date() >= date.today():
                    pos.close_position(data.datetime, data.Close)
                    

    # check for open positions
    def has_open_positions(self):
        for pos in self.positions:
            if pos.status == 'open':
                return True
        return False
    
    # strategy logic how positions should be opened/closed
    def logic(self, data):
        # if no position is open
        if not self.has_open_positions():
            
            # BUY
            if data['signal'] == 'Buy':

                # Position variables
                open_datetime = data['datetime']
                open_price = data['Close']
                order_type = 'buy'
                volume = 100000
                com = - 7.00
                sl = open_price - self.sl_pips/10000
                tp = open_price + self.tp_pips/10000
#                 tp_half = open_price + 0.00100

                self.add_position(Position(open_datetime, open_price, order_type, volume, sl, tp, com))
                
            # SELL
            elif data['signal'] == 'Buy':

                # Position variables
                open_datetime = data['datetime']
                open_price = data['Close']
                order_type = 'sell'
                volume = 100000 #100xau, 1000ypy, 100000REG
                com = - 7.00
                sl = open_price + self.sl_pips/10000 #1.00XAU #0.100JPY, 0.00100REG #10.00
                tp = open_price - self.tp_pips/10000
#                 tp_half = open_price - 0.00100

                self.add_position(Position(open_datetime, open_price, order_type, volume, sl, tp, com))
            
            
# logic
    def run(self):
        # data represents a moment in time while iterating through the backtest
        for i, data in self.data.iterrows():
            # close positions when stop loss or take profit is reached
            self.close_tp_sl(data)
            
            # strategy logic
            self.logic(data)
        
        return self.get_positions_df()


# timeframe = 'D1'
# EXCHANGE = 'OANDA'

# pairs = ["EURCAD", "AUDUSD"] #, "USDCAD", "USDCHF", "AUDCAD", "CADCHF", "NZDUSD", "EURCAD", "AUDCHF", "GBPUSD",
#          #"GBPCAD", "GBPNZD", "AUDNZD", "EURGBP", "EURNZD", "GBPCHF", "EURCHF", "EURAUD", "NZDCAD", "NZDCHF", "GBPAUD"]
# pairs_jpy = ["GBPJPY", "CADJPY", "EURJPY", "AUDJPY", "NZDJPY","USDJPY","CHFJPY"]



# data_list = []
# for inst in pairs: 
#     df_backtest = get_data(inst, timeframe, EXCHANGE)
#     df_backtest = add_all_ta_features(df_backtest, open='Open', high='High', low='Low', close='Close', volume='Volume')
#     df_backtest = signal(df_backtest)
# #     df_backtest.dropna(inplace = True)
# #     df_backtest = df_backtest.reset_index()
# #     df_backtest = df_backtest.rename(columns={"index":"time", "Close":"close"})
#     data_list.append(df_backtest)


# # In[10]:


# strategy_name = "demo"
# starting_balance = 100000
# sl_pips = 10
# tp_pips = 10
# #batch
# results_0 = []
# for symbol, data in zip (pairs, data_list):
#     strategy = Strategy(data, starting_balance, sl_pips, tp_pips)
#     result = strategy.run()
#     result["symbol"] = symbol
#     results_0.append(result)
    


# results = pd.concat(results_0)
# results




