import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px

pd.options.mode.chained_assignment = None


def extract_curr_df(df, num_days_in_year, num_test_years, test_num):
    """Returns the relevant dataframe for current testing year"""

    trainset = np.arange(num_days_in_year)
    testset = np.arange(num_days_in_year, 2 * num_days_in_year)

    trainidx = trainset - num_days_in_year * (num_test_years - test_num + 1)
    testidx = testset - num_days_in_year * (num_test_years - test_num + 1)

    return pd.concat([df.iloc[trainidx], df.iloc[testidx]])


class Pairs_trade_test:

    def __init__(
        self, df, asset1, asset2, price_type, num_days_in_year, num_test_years
    ):

        self.df = df

        self.asset1 = asset1
        self.asset2 = asset2

        self.price_type = price_type

        self.asset1_price_col = "{}_{}".format(price_type.capitalize(), asset1)
        self.asset2_price_col = "{}_{}".format(price_type.capitalize(), asset2)

        self.num_days_in_year = num_days_in_year
        self.num_test_years = num_test_years

        self.trainset = np.arange(self.num_days_in_year)
        self.testset = np.arange(self.num_days_in_year, 2 * self.num_days_in_year)

        # this is for storing the individual rows that are to go into the final result dataframe
        self.result_df_rows = []

    def buy_and_hold_return(self):
        """Returns the buy and hold return for the current testing year"""

        df = self.df
        asset1_price_col = self.asset1_price_col
        asset2_price_col = self.asset2_price_col
        num_days_in_year = self.num_days_in_year

        asset1_ret = (
            df[asset1_price_col].iloc[-1] / df[asset1_price_col].iloc[-num_days_in_year]
            - 1
        )
        asset2_ret = (
            df[asset2_price_col].iloc[-1] / df[asset2_price_col].iloc[-num_days_in_year]
            - 1
        )

        buy_and_hold_return = (asset1_ret + asset2_ret) / 2

        return buy_and_hold_return

    def find_opt_hedge_ratio(self):
        """
        Runs OLS regression on the two assets' prices to find optimal hedge ratio
        Does not return, just stores in self.df
        """

        df = self.df
        asset1_price_col = self.asset1_price_col
        asset2_price_col = self.asset2_price_col

        model = sm.OLS(df[asset1_price_col], df[asset2_price_col])
        res = model.fit()
        hedge_ratio = res.params.loc[asset2_price_col]

        self.hedge_ratio = hedge_ratio

        return hedge_ratio

    def calculate_spread(self):
        """
        Calculates spread between the two assets based on hedge ratio
        Does not return, just stores in self.df
        """

        df = self.df
        asset1_price_col = self.asset1_price_col
        asset2_price_col = self.asset2_price_col

        hedge_ratio = self.find_opt_hedge_ratio()

        df["spread"] = df[asset1_price_col] - hedge_ratio * df[asset2_price_col]

        self.df = df

    def zscore_calculation(self):
        """
        Calculates zscore for spread based on training year spread mean and stdev
        Does not return, just stores in self.df
        """

        self.calculate_spread()
        df = self.df
        trainset = self.trainset

        train_spread = df["spread"].iloc[trainset]
        train_spread_mean = train_spread.mean()
        train_spread_stdev = train_spread.std()

        df["zscore"] = (df["spread"] - train_spread_mean) / train_spread_stdev

        self.df = df

    @staticmethod
    def signal_generation(df_test, entry_std, exit_std, hedge_ratio):
        """
        Optimal positions (trades to be executed) calculated based on zscore
        Returns dataframe with optimal positions lagged by 1 day (implementation delay)
        """

        df_test["short_GLD"] = np.nan
        df_test["short_GDX"] = np.nan
        df_test["long_GLD"] = np.nan
        df_test["long_GDX"] = np.nan
        df_test.loc[
            df_test.index[0], ("short_GLD", "short_GDX", "long_GLD", "long_GDX")
        ] = 0

        # entry and exit conditions
        df_test.loc[df_test["zscore"] >= entry_std, ("short_GLD", "short_GDX")] = [-1, hedge_ratio]
        df_test.loc[df_test["zscore"] <= exit_std, ("short_GLD", "short_GDX")] = [0, 0]
        df_test.loc[df_test["zscore"] <= -entry_std, ("long_GLD", "long_GDX")] = [1, -hedge_ratio]
        df_test.loc[df_test["zscore"] >= -exit_std, ("long_GLD", "long_GDX")] = [0, 0]

        # forward fill so that any time in between entry and exit conditions, we mark position as being currently held
        df_test.loc[:, ["short_GLD", "short_GDX", "long_GLD", "long_GDX"]] = df_test[
            ["short_GLD", "short_GDX", "long_GLD", "long_GDX"]
        ].ffill()

        df_test["pos_GLD"] = df_test["short_GLD"] + df_test["long_GLD"]
        df_test["pos_GDX"] = df_test["short_GDX"] + df_test["long_GDX"]

        # this accounts for the implementation delay of 1 day that is discussed in the Assumptions section of gld_gdx_pairs.ipynb
        df_test["pos_GLD"] = df_test["pos_GLD"].shift(1).fillna(0)
        df_test["pos_GDX"] = df_test["pos_GDX"].shift(1).fillna(0)

        df_test.drop(
            columns=["short_GLD", "short_GDX", "long_GLD", "long_GDX"], inplace=True
        )

        return df_test

    @staticmethod
    def execute_trade_helper(df, hedge_ratio, initial_cash, short_sell_margin):
        """
        Executes trades and evaluates PnL on a day-by-day basis 
        based on the calculated optimal positions and the previous day's positions
        """

        # we need previous day's position to know whether we need to enter/exit a trade or just maintain
        df["prev_pos_GLD"] = df["pos_GLD"].shift(1).fillna(0)
        df["prev_pos_GDX"] = df["pos_GDX"].shift(1).fillna(0)

        df["cash_bop"] = np.nan
        df["cash_eop"] = np.nan

        df["shares_GLD"] = np.nan
        df["shares_GDX"] = np.nan

        df["short_execution_price"] = np.nan

        df["long_liquidity"] = np.nan
        df["short_liquidity"] = np.nan

        df.loc[df.index[0], "cash_bop"] = initial_cash

        for i, row in enumerate(df.itertuples()):

            idx = row.Index
            prev_idx = df.index[max(i - 1, 0)]
            next_idx = df.index[min(i + 1, len(df) - 1)]
            cash_bop = row.cash_bop
            cash_eop = row.cash_bop
            price_GLD = row.Close_GLD
            price_GDX = row.Close_GDX
            pos_GLD = row.pos_GLD
            pos_GDX = row.pos_GDX
            prev_pos_GLD = row.prev_pos_GLD
            prev_pos_GDX = row.prev_pos_GDX

            long_liquidity = 0
            short_liquidity = 0

            # price at which short trade was entered must be recorded for PnL record-keeping
            short_execution_price = np.nan

            if prev_pos_GLD == 0:  # we are not invested
                shares_GLD = 0
                shares_GDX = 0

                if pos_GLD != 0:  # signal tell us to invest
                    if pos_GLD > 0:
                        # this allows us to maintain an optimally hedged portfolio
                        shares = cash_bop / (
                            price_GLD + price_GDX * hedge_ratio * short_sell_margin
                        )
                    elif pos_GLD < 0:
                        shares = cash_bop / (
                            price_GLD * short_sell_margin + price_GDX * hedge_ratio
                        )
                    shares_GLD = shares * pos_GLD
                    shares_GDX = shares * pos_GDX

                    long_price = price_GLD if pos_GLD > 0 else price_GDX
                    short_price = price_GLD if pos_GLD < 0 else price_GDX
                    long_shares = shares_GLD if pos_GLD > 0 else shares_GDX
                    short_shares = -shares_GLD if pos_GLD < 0 else -shares_GDX

                    short_execution_price = short_price

                    # PnL calculation
                    long_liquidity = long_price * long_shares
                    short_liquidity = (
                        short_execution_price * short_shares * short_sell_margin
                    )
                    
                    # assert (long_liquidity + short_liquidity) == cash_bop

                    cash_eop = 0

            else:  # we are invested
                shares_GLD = df.loc[prev_idx, "shares_GLD"]
                shares_GDX = df.loc[prev_idx, "shares_GDX"]

                long_price = price_GLD if prev_pos_GLD > 0 else price_GDX
                short_price = price_GLD if prev_pos_GLD < 0 else price_GDX
                long_shares = shares_GLD if prev_pos_GLD > 0 else shares_GDX
                short_shares = -shares_GLD if prev_pos_GLD < 0 else -shares_GDX

                short_execution_price = df.loc[prev_idx, "short_execution_price"]

                long_liquidity = long_price * long_shares
                short_liquidity = (
                    short_execution_price * short_shares * short_sell_margin
                    + (short_execution_price - short_price) * short_shares
                )

                if pos_GLD == 0:  # we liquidate position
                    cash_eop = cash_bop + long_liquidity + short_liquidity
                    shares_GLD = 0
                    shares_GDX = 0
                    long_liquidity = 0
                    short_liquidity = 0
                    short_execution_price = np.nan

                elif (
                    pos_GLD != prev_pos_GLD
                ):  # liquidate position and enter into opposite trade
                    cash = cash_bop + long_liquidity + short_liquidity

                    if pos_GLD > 0:
                        shares = cash / (
                            price_GLD + price_GDX * hedge_ratio * short_sell_margin
                        )
                    elif pos_GLD < 0:
                        shares = cash / (
                            price_GLD * short_sell_margin + price_GDX * hedge_ratio
                        )

                    shares_GLD = shares * pos_GLD
                    shares_GDX = shares * pos_GDX

                    long_price = price_GLD if pos_GLD > 0 else price_GDX
                    short_price = price_GLD if pos_GLD < 0 else price_GDX
                    long_shares = shares_GLD if pos_GLD > 0 else shares_GDX
                    short_shares = -shares_GLD if pos_GLD < 0 else -shares_GDX

                    short_execution_price = short_price

                    long_liquidity = long_price * long_shares
                    short_liquidity = (
                        short_execution_price * short_shares * short_sell_margin
                    )

                    # assert (long_liquidity + short_liquidity) == cash

                    cash_eop = 0

            # all updates to trade dataframe made here
            df.loc[idx, ("shares_GLD", "shares_GDX")] = [shares_GLD, shares_GDX]
            df.loc[idx, "short_execution_price"] = short_execution_price
            df.loc[idx, "long_liquidity"] = long_liquidity
            df.loc[idx, "short_liquidity"] = short_liquidity
            df.loc[idx, "cash_eop"] = cash_eop
            if i != (len(df) - 1):
                df.loc[next_idx, "cash_bop"] = cash_eop

        df["net_liquidity"] = (
            df["long_liquidity"] + df["short_liquidity"] + df["cash_eop"]
        )
        # daily returns calculated based on net liquidity
        df["daily_ret"] = df["net_liquidity"].pct_change()

        return df

    def execute_trades(
        self, entry_std, exit_std, initial_cash=1000, short_sell_margin=1
    ):
        """
        Leverages signal_generation() and execute_trade_helper() to perform the test
        Returns the test dataframe with all trades executed and PnL calculated
        """

        df = self.df.copy()
        testset = self.testset
        hedge_ratio = self.hedge_ratio
        df_test = df.iloc[testset]
        df_test = Pairs_trade_test.signal_generation(
            df_test, entry_std, exit_std, hedge_ratio
        )
        df_test = Pairs_trade_test.execute_trade_helper(
            df_test, hedge_ratio, initial_cash, short_sell_margin
        )

        return df_test

    def analyze_performance(self, df_test, entry_std, exit_std):
        """
        Calculates cumulative return and days actively in a trade
        Does not return, just stores results of the current parameters for current year in self.result_df_rows
        """

        test_start_date = df_test.index[0]
        test_end_date = df_test.index[-1]

        cumulative_pnl = (
            df_test["net_liquidity"].iloc[-1] / df_test["net_liquidity"].iloc[0] - 1
        )

        positions_used = df_test["pos_GLD"].iloc[:-1]
        days_active = len(positions_used.loc[positions_used != 0])

        df_row_dict = {
            "test_start_date": test_start_date,
            "test_end_date": test_end_date,
            "entry_std": entry_std,
            "exit_std": exit_std,
            "days_in_trade": days_active,
            "return": cumulative_pnl,
        }

        self.result_df_rows.append(df_row_dict)

    @staticmethod
    def process_result_df(result_df, sorted=True):
        """
        Performs groupby on the concatenated result_df with a row for each parameter pair for each year
        to get sharpe ratio and average days actively in trade for each parameter pair
        Returns groupby dataframe
        """

        grouped_df = result_df.groupby(by=["entry_std", "exit_std"]).agg(
            {"return": ["mean", "std"], "days_in_trade": ["mean"]}
        )
        grouped_df.columns = ["mean_return", "std_return", "avg_days_in_trade"]
        grouped_df.reset_index(inplace=True)
        # grouped_df['mean_return'] = grouped_df['mean_return']
        grouped_df["sharpe"] = grouped_df["mean_return"] / grouped_df["std_return"]

        if sorted:
            grouped_df.sort_values("sharpe", ascending=False, inplace=True)

        return grouped_df

    @staticmethod
    def show_top_strategies(
        grouped_df, buy_and_hold_return_list, num_days_in_trade, n=5
    ):
        """
        Returns the top n strategies with their sharpe ratio 
        and compares them to the simple buy and hold strategy in dataframe format

        """

        buy_and_hold_mean_return = np.mean(buy_and_hold_return_list)
        buy_and_hold_std_return = np.std(buy_and_hold_return_list)
        buy_and_hold_sharpe = buy_and_hold_mean_return / buy_and_hold_std_return

        df_row_dict = {
            "entry_std": "Buy and Hold",
            "exit_std": "Buy and Hold",
            "mean_return": buy_and_hold_mean_return,
            "std_return": buy_and_hold_std_return,
            "avg_days_in_trade": num_days_in_trade,
            "sharpe": buy_and_hold_sharpe,
        }

        buy_and_hold_df = pd.DataFrame.from_records([df_row_dict])

        df = grouped_df.sort_values("sharpe", ascending=False)
        top_strategies_df = df.iloc[:n]
        return pd.concat([top_strategies_df, buy_and_hold_df]).reset_index(drop=True)

    def plot_top_strategies_helper(optimal_parameter_list, cum_pnl_dict, n):
        """Helper function to plot the cumulative PnL of the top n strategies and buy and hold strategy"""

        new_cum_pnl_dict = {
            (
                "Entry={}, Exit={}".format(round(key[0], 1), round(key[1], 1))
                if type(key) == tuple
                else key
            ): value
            for key, value in cum_pnl_dict.items()
        }

        df = pd.DataFrame(new_cum_pnl_dict)
        df.reset_index(inplace=True)

        df_melted = df.melt(id_vars=["Date"], var_name="Series", value_name="Value")

        fig = px.line(
            df_melted,
            x="Date",
            y="Value",
            color="Series",
            title="Cumulative PnL of Top {} Strategies".format(n),
            labels={"index": "Index", "Value": "Cumulative PnL", "Series": "Strategy"},
        )
        fig.update_layout(width=1000, height=600)
        fig.show()

    @staticmethod
    def plot_top_strategies(
        grouped_df,
        param_dict,
        df,
        asset1,
        asset2,
        price_type,
        num_days_in_year,
        num_test_years,
        n,
    ):
        """Calculates and Plots the cumulative PnL of the top n strategies and buy and hold strategy"""

        cum_pnl_dict = {}

        grouped_df = grouped_df.sort_values("sharpe", ascending=False)
        optimal_parameter_list = grouped_df[["entry_std", "exit_std"]].iloc[:5].values
        optimal_parameter_list = [
            tuple(inner_list) for inner_list in optimal_parameter_list
        ]

        for params in optimal_parameter_list:
            df_temp = pd.concat(param_dict[params])
            df_temp["daily_ret"] = df_temp["daily_ret"].fillna(0)
            df_temp["cum_pnl"] = (df_temp["daily_ret"] + 1).cumprod()
            cum_pnl_dict[params] = df_temp["cum_pnl"]

        full_df = df.iloc[-num_days_in_year * num_test_years :]
        asset1_price_col = "{}_{}".format(price_type.capitalize(), asset1)
        asset2_price_col = "{}_{}".format(price_type.capitalize(), asset2)

        full_df["ret_{}".format(asset1)] = full_df[asset1_price_col].pct_change()
        full_df["ret_{}".format(asset2)] = full_df[asset2_price_col].pct_change()
        full_df["ret_{}".format(asset1)] = full_df["ret_{}".format(asset1)].fillna(0)
        full_df["ret_{}".format(asset2)] = full_df["ret_{}".format(asset2)].fillna(0)

        full_df["cum_pnl_{}".format(asset1)] = (
            full_df["ret_{}".format(asset1)] + 1
        ).cumprod()
        full_df["cum_pnl_{}".format(asset2)] = (
            full_df["ret_{}".format(asset2)] + 1
        ).cumprod()
        full_df["cum_pnl"] = (
            full_df["cum_pnl_{}".format(asset1)] + full_df["cum_pnl_{}".format(asset2)]
        ) / 2

        cum_pnl_dict["Buy and Hold"] = full_df["cum_pnl"]

        Pairs_trade_test.plot_top_strategies_helper(
            optimal_parameter_list, cum_pnl_dict, n
        )
