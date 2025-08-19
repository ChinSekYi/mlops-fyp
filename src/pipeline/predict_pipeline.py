import os
import sys
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred_result = model.predict(data_scaled)
            return pred_result
        except Exception as e:
            raise CustomException(e, sys)
        )
        self.operating_expenses_short_term_liabilities = (
            operating_expenses_short_term_liabilities
        )
        self.operating_expenses_total_liabilities = operating_expenses_total_liabilities
        self.profit_on_sales_total_assets = profit_on_sales_total_assets
        self.total_sales_total_assets = total_sales_total_assets
        self.current_assets_inventories_long_term_liabilities = (
            current_assets_inventories_long_term_liabilities
        )
        self.constant_capital_total_assets = constant_capital_total_assets
        self.profit_on_sales_sales = profit_on_sales_sales
        self.current_assets_inventory_receivables_short_term_liabilities = (
            current_assets_inventory_receivables_short_term_liabilities
        )
        self.total_liabilities_profit_on_operating_activities_depreciation_12_365 = (
            total_liabilities_profit_on_operating_activities_depreciation_12_365
        )
        self.profit_on_operating_activities_sales = profit_on_operating_activities_sales
        self.rotation_receivables_inventory_turnover_days = (
            rotation_receivables_inventory_turnover_days
        )
        self.receivables_365_sales = receivables_365_sales
        self.net_profit_inventory = net_profit_inventory
        self.current_assets_inventory_short_term_liabilities = (
            current_assets_inventory_short_term_liabilities
        )
        self.inventory_365_cost_of_products_sold = inventory_365_cost_of_products_sold
        self.EBITDA_profit_on_operating_activities_depreciation_total_assets = (
            EBITDA_profit_on_operating_activities_depreciation_total_assets
        )
        self.EBITDA_profit_on_operating_activities_depreciation_sales = (
            EBITDA_profit_on_operating_activities_depreciation_sales
        )
        self.current_assets_total_liabilities = current_assets_total_liabilities
        self.short_term_liabilities_total_assets = short_term_liabilities_total_assets
        self.short_term_liabilities_365_cost_of_products_sold = (
            short_term_liabilities_365_cost_of_products_sold
        )
        self.equity_fixed_assets = equity_fixed_assets
        self.constant_capital_fixed_assets = constant_capital_fixed_assets
        self.working_capital = working_capital
        self.sales_cost_of_products_sold_sales = sales_cost_of_products_sold_sales
        self.current_assets_inventory_short_term_liabilities_sales_gross_profit_depreciation = current_assets_inventory_short_term_liabilities_sales_gross_profit_depreciation
        self.total_costs_total_sales = total_costs_total_sales
        self.long_term_liabilities_equity = long_term_liabilities_equity
        self.sales_inventory = sales_inventory
        self.sales_receivables = sales_receivables
        self.short_term_liabilities_365_sales = short_term_liabilities_365_sales
        self.sales_short_term_liabilities = sales_short_term_liabilities
        self.sales_fixed_assets = sales_fixed_assets

    def get_data_as_dataframe(self):
        """
        Converts the attributes of CustomData into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame where each row represents a financial metric attribute.
        """
        try:
            custom_data_input = {
                "net profit / total assets": [self.net_profit_total_assets],
                "total liabilities / total assets": [
                    self.total_liabilities_total_assets
                ],
                "working capital / total assets": [self.working_capital_total_assets],
                "current assets / short-term liabilities": [
                    self.current_assets_short_term_liabilities
                ],
                "[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365": [
                    self.cash_short_term_securities_receivables_short_term_liabilities_operating_expenses_depreciation_365
                ],
                "retained earnings / total assets": [
                    self.retained_earnings_total_assets
                ],
                "EBIT / total assets": [self.EBIT_total_assets],
                "book value of equity / total liabilities": [
                    self.book_value_of_equity_total_liabilities
                ],
                "sales / total assets": [self.sales_total_assets],
                "equity / total assets": [self.equity_total_assets],
                "(gross profit + extraordinary items + financial expenses) / total assets": [
                    self.gross_profit_extraordinary_items_financial_expenses_total_assets
                ],
                "gross profit / short-term liabilities": [
                    self.gross_profit_short_term_liabilities
                ],
                "(gross profit + depreciation) / sales": [
                    self.gross_profit_depreciation_sales
                ],
                "(gross profit + interest) / total assets": [
                    self.gross_profit_interest_total_assets
                ],
                "(total liabilities * 365) / (gross profit + depreciation)": [
                    self.total_liabilities_365_gross_profit_depreciation
                ],
                "(gross profit + depreciation) / total liabilities": [
                    self.gross_profit_depreciation_total_liabilities
                ],
                "total assets / total liabilities": [
                    self.total_assets_total_liabilities
                ],
                "gross profit / total assets": [self.gross_profit_total_assets],
                "gross profit / sales": [self.gross_profit_sales],
                "(inventory * 365) / sales": [self.inventory_365_sales],
                "sales (n) / sales (n-1)": [self.sales_n_sales_n_1],
                "profit on operating activities / total assets": [
                    self.profit_on_operating_activities_total_assets
                ],
                "net profit / sales": [self.net_profit_sales],
                "gross profit (in 3 years) / total assets": [
                    self.gross_profit_3_years_total_assets
                ],
                "(equity - share capital) / total assets": [
                    self.equity_share_capital_total_assets
                ],
                "(net profit + depreciation) / total liabilities": [
                    self.net_profit_depreciation_total_liabilities
                ],
                "profit on operating activities / financial expenses": [
                    self.profit_on_operating_activities_financial_expenses
                ],
                "working capital / fixed assets": [self.working_capital_fixed_assets],
                "logarithm of total assets": [self.logarithm_total_assets],
                "(total liabilities - cash) / sales": [
                    self.total_liabilities_cash_sales
                ],
                "(gross profit + interest) / sales": [self.gross_profit_interest_sales],
                "(current liabilities * 365) / cost of products sold": [
                    self.current_liabilities_365_cost_of_products_sold
                ],
                "operating expenses / short-term liabilities": [
                    self.operating_expenses_short_term_liabilities
                ],
                "operating expenses / total liabilities": [
                    self.operating_expenses_total_liabilities
                ],
                "profit on sales / total assets": [self.profit_on_sales_total_assets],
                "total sales / total assets": [self.total_sales_total_assets],
                "(current assets - inventories) / long-term liabilities": [
                    self.current_assets_inventories_long_term_liabilities
                ],
                "constant capital / total assets": [self.constant_capital_total_assets],
                "profit on sales / sales": [self.profit_on_sales_sales],
                "(current assets - inventory - receivables) / short-term liabilities": [
                    self.current_assets_inventory_receivables_short_term_liabilities
                ],
                "total liabilities / ((profit on operating activities + depreciation) * (12/365))": [
                    self.total_liabilities_profit_on_operating_activities_depreciation_12_365
                ],
                "profit on operating activities / sales": [
                    self.profit_on_operating_activities_sales
                ],
                "rotation receivables + inventory turnover in days": [
                    self.rotation_receivables_inventory_turnover_days
                ],
                "(receivables * 365) / sales": [self.receivables_365_sales],
                "net profit / inventory": [self.net_profit_inventory],
                "(current assets - inventory) / short-term liabilities": [
                    self.current_assets_inventory_short_term_liabilities
                ],
                "(inventory * 365) / cost of products sold": [
                    self.inventory_365_cost_of_products_sold
                ],
                "EBITDA (profit on operating activities - depreciation) / total assets": [
                    self.EBITDA_profit_on_operating_activities_depreciation_total_assets
                ],
                "EBITDA (profit on operating activities - depreciation) / sales": [
                    self.EBITDA_profit_on_operating_activities_depreciation_sales
                ],
                "current assets / total liabilities": [
                    self.current_assets_total_liabilities
                ],
                "short-term liabilities / total assets": [
                    self.short_term_liabilities_total_assets
                ],
                "(short-term liabilities * 365) / (cost of products sold)": [
                    self.short_term_liabilities_365_cost_of_products_sold
                ],
                "equity / fixed assets": [self.equity_fixed_assets],
                "constant capital / fixed assets": [self.constant_capital_fixed_assets],
                "working capital": [self.working_capital],
                "(sales - cost of products sold) / sales": [
                    self.sales_cost_of_products_sold_sales
                ],
                "(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)": [
                    self.current_assets_inventory_short_term_liabilities_sales_gross_profit_depreciation
                ],
                "total costs /total sales": [self.total_costs_total_sales],
                "long-term liabilities / equity": [self.long_term_liabilities_equity],
                "sales / inventory": [self.sales_inventory],
                "sales / receivables": [self.sales_receivables],
                "(short-term liabilities *365) / sales": [
                    self.short_term_liabilities_365_sales
                ],
                "sales / short-term liabilities": [self.sales_short_term_liabilities],
                "sales / fixed assets": [self.sales_fixed_assets],
            }

            return pd.DataFrame(custom_data_input)

        except Exception as e:
            raise CustomException(e, sys) from e
