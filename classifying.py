"""
金融大数据应用案例 - 量化选股 (Python版本)
作者：1
日期：2025.12.1
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
import xlrd
import openpyxl

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ================ 添加智能读取函数 ================
def smart_read_excel(file_path, expected_columns=None):
    """
    智能读取Excel文件，自动处理表头错行问题

    Parameters:
    -----------
    file_path : str
        Excel文件路径
    expected_columns : list
        预期的列名列表，用于验证

    Returns:
    --------
    pandas.DataFrame
        读取并清理后的数据
    """
    import pandas as pd

    # 尝试读取前20行来理解文件结构
    temp_df = pd.read_excel(file_path, header=None, nrows=20)

    # 寻找可能的表头行
    header_row = None
    for i in range(min(10, len(temp_df))):
        row_values = temp_df.iloc[i].astype(str).tolist()

        # 检查这一行是否包含预期的列名关键词
        keywords = ['代码', '日期', '开盘', '最高', '最低', '收盘', '成交']
        if expected_columns:
            # 从预期列名中提取关键词
            expected_keywords = []
            for col in expected_columns:
                if isinstance(col, str):
                    # 提取中文关键词
                    if any(ch in col for ch in ['代码', '日期', '价', '量']):
                        expected_keywords.append(col)
            if expected_keywords:
                keywords = expected_keywords

        row_text = ' '.join(row_values).lower()
        # 检查是否包含任何关键词
        if any(any(keyword in row_text for keyword in keyword_list) for keyword_list in [keywords]):
            header_row = i
            print(f"找到表头行: {i} (内容: {row_values})")
            break

    if header_row is not None:
        # 使用找到的表头行
        df = pd.read_excel(file_path, header=header_row)

        # 清理列名
        df.columns = [str(col).strip() for col in df.columns]

        # 如果有多余的行，删除
        df = df.dropna(how='all')

        return df
    else:
        # 如果没找到，使用默认方式
        print("警告：未找到明确的表头行，使用默认读取方式")
        return pd.read_excel(file_path)
# ================ 函数定义结束 ================


class DataProcessor:
    """数据获取与预处理类"""

    def __init__(self, data_path='./processed_data/'):  # 改为processed_data更符合项目结构
        self.data_path = data_path
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def smart_read_cta_excel(self, file_path):
        """
        智能读取国泰安数据文件
        处理可能存在的表头错行问题
        """
        try:
            # 定义预期的列名（英文和中文）
            expected_english_cols = ['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dnshrtrd']
            expected_chinese_cols = ['股票代码', '日期', '开盘价', '最高价', '最低价', '收盘价', '成交量']

            # 先尝试用智能读取函数
            df = smart_read_excel(file_path, expected_chinese_cols)

            # 检查列名
            current_cols = [str(col).strip() for col in df.columns]

            # 尝试匹配英文列名
            english_match = all(col in current_cols for col in expected_english_cols)
            # 尝试匹配中文列名
            chinese_match = all(col in current_cols for col in expected_chinese_cols)

            if english_match:
                print(f"  使用英文列名: {current_cols}")
                # 已经是英文列名，直接使用
                return df
            elif chinese_match:
                print(f"  使用中文列名: {current_cols}")
                # 已经是中文列名，直接使用
                return df
            else:
                # 列名不匹配，尝试手动处理
                print(f"  列名不匹配，尝试其他方式...")
                print(f"  当前列名: {current_cols}")

                # 尝试读取前5行查看数据结构
                temp_df = pd.read_excel(file_path, header=None, nrows=5)
                print(f"  文件前5行预览:")
                print(temp_df)

                # 尝试不同的读取方式
                for skip_rows in [0, 1, 2, 3, 4]:
                    try:
                        test_df = pd.read_excel(file_path, skiprows=skip_rows)
                        test_cols = [str(col).strip() for col in test_df.columns]

                        # 检查是否包含关键列
                        has_stkcd = any('stkcd' in col.lower() or '代码' in col for col in test_cols)
                        has_trddt = any('trddt' in col.lower() or '日期' in col for col in test_cols)

                        if has_stkcd and has_trddt:
                            print(f"  使用skiprows={skip_rows}，列名: {test_cols}")
                            return test_df
                    except:
                        continue

                # 如果上述方法都失败，使用默认方式并手动设置列名
                print("  所有自动方式失败，使用默认读取并手动设置列名")
                df = pd.read_excel(file_path, header=None)

                # 假设数据有7列，设置默认列名
                if len(df.columns) >= 7:
                    df = df.iloc[:, :7]  # 只取前7列
                    df.columns = ['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dnshrtrd']
                    print("  已手动设置英文列名")
                    return df
                else:
                    raise ValueError(f"文件列数不足: {len(df.columns)}")

        except Exception as e:
            print(f"智能读取失败: {e}")
            # 回退到原始方法
            return pd.read_excel(file_path)

    def process_cta_data(self, input_files):
        """
        处理国泰安数据库导出的数据
        输入格式：如TRD_Dalyr.xlsx文件
        内容格式：Stkcd, Trddt, Opnprc, Hiprc, Loprc, Clsprc, Dnshrtrd
        """
        all_stocks_data = {}

        # 处理每个输入文件
        for file_path in input_files:
            print(f"处理文件: {file_path}")

            try:
                # 使用智能读取方法
                df = self.smart_read_cta_excel(file_path)

                print(f"  读取成功，数据形状: {df.shape}")
                print(f"  列名: {list(df.columns)}")

                # 检查数据
                if len(df) == 0:
                    print(f"  警告: 文件 {file_path} 没有数据")
                    continue

                # 重命名列，确保统一列名
                column_mapping = {
                    'Stkcd': '股票代码',
                    'Trddt': '日期',
                    'Opnprc': '开盘价',
                    'Hiprc': '最高价',
                    'Loprc': '最低价',
                    'Clsprc': '收盘价',
                    'Dnshrtrd': '成交量',
                    '股票代码': '股票代码',
                    '日期': '日期',
                    '开盘价': '开盘价',
                    '最高价': '最高价',
                    '最低价': '最低价',
                    '收盘价': '收盘价',
                    '成交量': '成交量'
                }

                # 应用列名映射
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

                # 确保有必要的列
                required_cols = ['股票代码', '日期', '开盘价', '最高价', '最低价', '收盘价', '成交量']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"  文件 {file_path} 缺少列: {missing_cols}")
                    print(f"  当前列: {list(df.columns)}")
                    # 尝试继续处理，可能有些文件格式不同
                    continue

                # 清理数据：删除包含非数字的行
                rows_before = len(df)
                for col in ['开盘价', '最高价', '最低价', '收盘价', '成交量']:
                    if col in df.columns:
                        # 转换为数值，无法转换的设为NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # 删除包含NaN的行
                df = df.dropna(subset=['开盘价', '最高价', '最低价', '收盘价', '成交量'])
                rows_after = len(df)
                print(f"  数据清理: {rows_before} → {rows_after} 行")

                # 按股票代码分组处理
                stock_codes = df['股票代码'].unique()
                print(f"  包含股票数量: {len(stock_codes)}")

                for stock_code in stock_codes:
                    group = df[df['股票代码'] == stock_code].copy()

                    # 确保股票代码是字符串
                    stock_code_str = str(int(stock_code)) if isinstance(stock_code, (int, float)) else str(stock_code)

                    # 处理日期格式
                    try:
                        group['日期'] = pd.to_datetime(group['日期'], errors='coerce')
                        # 删除日期无效的行
                        group = group.dropna(subset=['日期'])
                    except Exception as e:
                        print(f"  日期转换失败: {e}")
                        continue

                    if len(group) == 0:
                        continue

                    # 按日期降序排序（从近到远）
                    group = group.sort_values('日期', ascending=False).reset_index(drop=True)

                    # 转换为数字日期（类似MATLAB的datenum）
                    # 从1900-01-01开始的天数（Excel日期系统）
                    group['日期数字'] = (group['日期'] - pd.Timestamp('1900-01-01')).dt.days + 2

                    # 选择需要的列并重新排序
                    processed_df = pd.DataFrame({
                        '股票代码': stock_code_str,
                        '日期数字': group['日期数字'],
                        '开盘价': group['开盘价'],
                        '最高价': group['最高价'],
                        '最低价': group['最低价'],
                        '收盘价': group['收盘价'],
                        '成交量': group['成交量']
                    })

                    # 保存到字典
                    if stock_code_str not in all_stocks_data:
                        all_stocks_data[stock_code_str] = processed_df
                    else:
                        all_stocks_data[stock_code_str] = pd.concat(
                            [all_stocks_data[stock_code_str], processed_df],
                            ignore_index=True
                        ).drop_duplicates().sort_values('日期数字', ascending=False)

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 保存每支股票的数据到单独文件
        print(f"\n总共处理了 {len(all_stocks_data)} 支股票")
        saved_count = 0

        for stock_code, stock_df in all_stocks_data.items():
            if len(stock_df) >= 100:  # 至少100个交易日数据
                output_file = os.path.join(self.data_path, f"{stock_code}.xlsx")
                stock_df.to_excel(output_file, index=False)
                saved_count += 1

                if saved_count % 20 == 0:
                    print(f"已保存 {saved_count} 支股票数据")

        print(f"成功保存 {saved_count} 支股票数据到 {self.data_path}")
        return all_stocks_data

    def load_stock_data(self, stock_code):
        """加载单支股票数据"""
        file_path = os.path.join(self.data_path, f"{stock_code}.xlsx")
        if os.path.exists(file_path):
            return pd.read_excel(file_path)
        return None


class TechnicalIndicators:
    """计算技术指标类"""

    def __init__(self):
        self.indicators_list = [
            's_x1', 's_x2', 's_x3', 's_x4', 's_x5',  # 涨幅指标
            's_x6', 's_x7',  # ADR, RSI
            's_x8', 's_x9', 's_x10',  # K线值
            's_x11', 's_x12',  # 乘离率
            's_x13', 's_x14', 's_x15',  # RSV
            's_x16', 's_x17', 's_x18', 's_x19', 's_x20',  # OBV量比
            's_y'  # 分类标签
        ]

    def calculate_indicators(self, df):
        """计算20个技术指标和分类标签"""
        if df is None or len(df) < 100:
            return None

        # 创建副本，避免修改原始数据
        result_df = df.copy()

        # 1-5: 不同周期的涨幅
        for n, col in zip([1, 2, 5, 10, 30], ['s_x1', 's_x2', 's_x3', 's_x4', 's_x5']):
            result_df[col] = ((result_df['收盘价'] - result_df['收盘价'].shift(n)) /
                              result_df['收盘价'].shift(n) * 100)

        # 6: 10日涨跌比率ADR
        # 先计算每日涨跌
        price_change = result_df['收盘价'].diff()
        up_days = (price_change > 0).rolling(window=10, min_periods=1).sum()
        down_days = (price_change < 0).rolling(window=10, min_periods=1).sum()
        result_df['s_x6'] = up_days / down_days.replace(0, 0.001)

        # 7: 10日相对强弱指标RSI
        gains = price_change.where(price_change > 0, 0)
        losses = -price_change.where(price_change < 0, 0)

        avg_gain = gains.rolling(window=10, min_periods=1).mean()
        avg_loss = losses.rolling(window=10, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, 0.001)
        result_df['s_x7'] = 100 - (100 / (1 + rs))

        # 8: 当日K线值
        result_df['s_x8'] = (result_df['收盘价'] - result_df['开盘价']) / (
                result_df['最高价'] - result_df['最低价']).replace(0, 0.001)

        # 9: 3日K线值
        result_df['s_x9'] = (result_df['收盘价'] - result_df['开盘价'].shift(2)) / (
                result_df['最高价'].rolling(3).max() -
                result_df['最低价'].rolling(3).min()).replace(0, 0.001)

        # 10: 6日K线值
        result_df['s_x10'] = (result_df['收盘价'] - result_df['开盘价'].shift(5)) / (
                result_df['最高价'].rolling(6).max() -
                result_df['最低价'].rolling(6).min()).replace(0, 0.001)

        # 11: 6日乘离率
        result_df['s_x11'] = ((result_df['收盘价'] - result_df['收盘价'].rolling(6).mean()) /
                              result_df['收盘价'].rolling(6).mean() * 100)

        # 12: 10日乘离率
        result_df['s_x12'] = ((result_df['收盘价'] - result_df['收盘价'].rolling(10).mean()) /
                              result_df['收盘价'].rolling(10).mean() * 100)

        # 13-15: RSV指标 (9日, 30日, 90日)
        for n, col in zip([9, 30, 90], ['s_x13', 's_x14', 's_x15']):
            lowest = result_df['最低价'].rolling(n).min()
            highest = result_df['最高价'].rolling(n).max()
            result_df[col] = ((result_df['收盘价'] - lowest) /
                              (highest - lowest).replace(0, 0.001) * 100)

        # 16-20: OBV量比指标
        # 先计算OBV
        obv = np.zeros(len(result_df))
        volume = result_df['成交量'].values
        close = result_df['收盘价'].values

        obv[0] = volume[0]
        for i in range(1, len(result_df)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        result_df['OBV'] = obv

        # 计算不同周期的OBV量比
        for n, col in zip([1, 5, 10, 30, 60], ['s_x16', 's_x17', 's_x18', 's_x19', 's_x20']):
            if n == 1:
                # 当日OBV量比 = 当日OBV / 5日OBV均值
                result_df[col] = result_df['OBV'] / result_df['OBV'].rolling(5).mean().replace(0, 0.001)
            else:
                # n日OBV量比 = n日OBV均值 / (n*2)日OBV均值
                result_df[col] = (result_df['OBV'].rolling(n).mean() /
                                  result_df['OBV'].rolling(n * 2).mean().replace(0, 0.001))

        # 计算分类标签 s_y
        # 根据未来1日和3日涨幅确定标签
        future_1d_return = result_df['收盘价'].pct_change(-1) * 100
        future_3d_return = result_df['收盘价'].pct_change(-3) * 100

        # 创建分类标签：1表示涨，-1表示跌
        result_df['s_y'] = 0
        result_df.loc[(future_1d_return > 0) & (future_3d_return > 0), 's_y'] = 1
        result_df.loc[(future_1d_return < 0) & (future_3d_return < 0), 's_y'] = -1

        # 删除中间计算列
        result_df = result_df.drop(['OBV'], axis=1, errors='ignore')

        # 删除NaN值
        result_df = result_df.dropna(subset=self.indicators_list)

        return result_df

    def create_samples(self, all_stocks_data, n_calculations=18):
        """
        创建训练样本和预测样本
        每支股票计算n_calculations次，最近一次作为预测样本，其余作为训练样本
        """
        good_samples = []  # s_y = 1
        bad_samples = []  # s_y = -1
        forecast_samples = []  # 预测样本

        for stock_code, df in all_stocks_data.items():
            if df is not None and len(df) >= n_calculations + 20:  # 确保有足够数据
                # 获取最近n_calculations天的数据
                recent_data = df.head(n_calculations).copy()

                # 最近一天作为预测样本
                forecast_sample = recent_data.iloc[0:1].copy()
                forecast_sample['股票代码'] = stock_code
                forecast_samples.append(forecast_sample)

                # 其余作为训练样本
                for i in range(1, n_calculations):
                    if i < len(recent_data):
                        train_sample = recent_data.iloc[i:i + 1].copy()
                        train_sample['股票代码'] = stock_code

                        s_y_value = train_sample['s_y'].iloc[0]
                        if s_y_value == 1:
                            good_samples.append(train_sample)
                        elif s_y_value == -1:
                            bad_samples.append(train_sample)

        # 合并样本
        train_good_df = pd.concat(good_samples, ignore_index=True) if good_samples else pd.DataFrame()
        train_bad_df = pd.concat(bad_samples, ignore_index=True) if bad_samples else pd.DataFrame()
        forecast_df = pd.concat(forecast_samples, ignore_index=True) if forecast_samples else pd.DataFrame()

        return train_good_df, train_bad_df, forecast_df


class DataStandardizer:
    """数据标准化类"""

    def __init__(self):
        self.scaler_params = {}
        self.feature_columns = [f's_x{i}' for i in range(1, 21)]

    def min_max_scale(self, df, feature_columns=None):
        """最大最小标准化"""
        if feature_columns is None:
            feature_columns = self.feature_columns

        scaled_df = df.copy()

        for col in feature_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()

                if max_val > min_val:
                    scaled_df[col] = (df[col] - min_val) / (max_val - min_val)
                    self.scaler_params[col] = {'min': min_val, 'max': max_val}
                else:
                    scaled_df[col] = 0

        return scaled_df

    def scale_forecast_data(self, forecast_df):
        """使用训练数据参数标准化预测数据"""
        scaled_forecast = forecast_df.copy()

        for col in self.scaler_params:
            if col in forecast_df.columns:
                params = self.scaler_params[col]
                min_val = params['min']
                max_val = params['max']

                if max_val > min_val:
                    scaled_forecast[col] = (forecast_df[col] - min_val) / (max_val - min_val)
                else:
                    scaled_forecast[col] = 0

        return scaled_forecast


class FeatureSelector:
    """变量筛选类"""

    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.selected_features = []
        self.feature_columns = [f's_x{i}' for i in range(1, 21)]

    def select_features(self, scaled_train_df):
        """基于相关系数筛选特征"""
        if 's_y' not in scaled_train_df.columns:
            print("警告: 未找到分类标签列 s_y")
            return self.feature_columns

        # 计算相关系数矩阵
        corr_matrix = scaled_train_df[self.feature_columns + ['s_y']].corr()

        # 获取与s_y的相关系数绝对值
        corr_with_target = corr_matrix['s_y'].abs()

        # 筛选相关系数大于阈值的特征
        selected = corr_with_target[corr_with_target > self.threshold].index.tolist()

        # 移除s_y本身
        if 's_y' in selected:
            selected.remove('s_y')

        self.selected_features = selected
        print(f"筛选后特征数量: {len(selected)}")
        print(f"选中的特征: {selected}")

        return selected

    def filter_data(self, df):
        """过滤数据，只保留选中的特征"""
        if not self.selected_features:
            # 如果没有选中的特征，返回所有特征
            available_features = [f for f in self.feature_columns if f in df.columns]
            return df[available_features]

        # 确保选中的特征在数据中
        available_features = [f for f in self.selected_features if f in df.columns]
        return df[available_features]


class StockPredictor:
    """股票预测模型类"""

    def __init__(self):
        # 初始化分类器
        self.models = {
            'decision_tree': DecisionTreeClassifier(
                max_depth=5,
                random_state=42,
                min_samples_split=10,
                min_samples_leaf=5
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(50, 25),
                max_iter=1000,
                random_state=42,
                alpha=0.01,
                learning_rate_init=0.001
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0,
                solver='lbfgs'
            ),
            'svm': SVC(
                probability=True,
                random_state=42,
                C=1.0,
                kernel='rbf',
                gamma='scale'
            )
        }

        self.results = {}

    def train_and_evaluate(self, X_train, y_train, X_test=None, y_test=None):
        """训练和评估模型"""
        results = {}

        for model_name, model in self.models.items():
            print(f"\n训练 {model_name}...")

            # 训练模型
            model.fit(X_train, y_train)

            # 训练集预测
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)

            # 测试集预测（如果有）
            test_accuracy = None
            if X_test is not None and y_test is not None:
                y_test_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_test_pred)

            # 保存结果
            results[model_name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'y_train_true': y_train,
                'y_train_pred': y_train_pred,
                'y_test_true': y_test if y_test is not None else None,
                'y_test_pred': y_test_pred if X_test is not None else None
            }

            print(f"训练集准确率: {train_accuracy:.4f}")
            if test_accuracy is not None:
                print(f"测试集准确率: {test_accuracy:.4f}")

        self.results = results
        return results

    def predict_forecast(self, X_forecast):
        """预测样本预测"""
        forecast_results = {}

        for model_name, result in self.results.items():
            model = result['model']

            if hasattr(model, 'predict_proba'):
                # 获取预测概率
                forecast_prob = model.predict_proba(X_forecast)
                # 对于二分类问题，取正类的概率
                if forecast_prob.shape[1] == 2:
                    forecast_pred = forecast_prob[:, 1]
                else:
                    forecast_pred = forecast_prob
            else:
                # 直接预测类别
                forecast_pred = model.predict(X_forecast)

            forecast_results[model_name] = forecast_pred

        return forecast_results

    def evaluate_models(self):
        """评估所有模型"""
        evaluation_results = []

        for model_name, result in self.results.items():
            # 基础评估指标
            eval_dict = {
                '模型': model_name,
                '训练集准确率': result['train_accuracy']
            }

            if result['test_accuracy'] is not None:
                eval_dict['测试集准确率'] = result['test_accuracy']

            # 计算AUC
            if hasattr(result['model'], 'predict_proba'):
                y_score = result['model'].predict_proba(result['y_train_true'].values.reshape(-1, 1))[:, 1]
                fpr, tpr, _ = roc_curve(result['y_train_true'], y_score)
                auc_score = auc(fpr, tpr)
                eval_dict['AUC值'] = auc_score

            # 分类报告
            print(f"\n{model_name} 分类报告:")
            print("=" * 40)
            print(classification_report(
                result['y_train_true'],
                result['y_train_pred'],
                target_names=['跌', '涨']
            ))

            evaluation_results.append(eval_dict)

        # 创建评估结果DataFrame
        eval_df = pd.DataFrame(evaluation_results)

        print("\n" + "=" * 60)
        print("模型评估汇总:")
        print("=" * 60)
        print(eval_df.to_string(index=False))

        return eval_df

    def plot_roc_curves(self, save_path='./results/roc_curves.png'):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))

        for model_name, result in self.results.items():
            if hasattr(result['model'], 'predict_proba'):
                y_score = result['model'].predict_proba(result['y_train_true'].values.reshape(-1, 1))[:, 1]
                fpr, tpr, _ = roc_curve(result['y_train_true'], y_score)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, lw=2,
                         label=f'{model_name} (AUC = {roc_auc:.3f})')

        # 绘制对角线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)')
        plt.ylabel('真阳性率 (True Positive Rate)')
        plt.title('ROC曲线比较')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"ROC曲线已保存到: {save_path}")

    def save_forecast_results(self, forecast_results, stock_codes, output_dir='./results/'):
        """保存预测结果到Excel文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for model_name, predictions in forecast_results.items():
            # 创建预测结果DataFrame
            forecast_df = pd.DataFrame({
                '股票代码': stock_codes,
                '预测值': predictions,
                '预测标签': ['涨' if p > 0.5 else '跌' for p in predictions] if predictions.ndim == 1 else [
                    '涨' if p[1] > 0.5 else '跌' for p in predictions],
                '预测日期': datetime.now().strftime('%Y-%m-%d')
            })

            # 排序：预测值高的在前
            forecast_df = forecast_df.sort_values('预测值', ascending=False)

            # 保存到Excel
            output_file = os.path.join(output_dir, f'forecast_result_{model_name}.xlsx')
            forecast_df.to_excel(output_file, index=False)
            print(f"已保存 {model_name} 预测结果到: {output_file}")


class EnsemblePredictor:
    """组合选举法综合分类器"""

    def __init__(self, base_predictor):
        self.base_predictor = base_predictor
        self.ensemble_model = None

    def create_voting_classifier(self, voting='soft'):
        """创建投票分类器"""
        # 收集所有基础模型
        estimators = []
        for model_name, result in self.base_predictor.results.items():
            estimators.append((model_name, result['model']))

        # 创建投票分类器
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting=voting
        )

        return self.ensemble_model

    def ensemble_predict(self, X_train, y_train, X_forecast):
        """集成预测"""
        if self.ensemble_model is None:
            self.create_voting_classifier()

        # 训练集成模型
        self.ensemble_model.fit(X_train, y_train)

        # 预测
        if hasattr(self.ensemble_model, 'predict_proba'):
            ensemble_pred = self.ensemble_model.predict_proba(X_forecast)
        else:
            ensemble_pred = self.ensemble_model.predict(X_forecast)

        return ensemble_pred

    def evaluate_ensemble(self, X_train, y_train):
        """评估集成模型"""
        if self.ensemble_model is None:
            print("集成模型未创建")
            return None

        # 训练集成模型
        self.ensemble_model.fit(X_train, y_train)

        # 预测
        y_pred = self.ensemble_model.predict(X_train)

        # 计算准确率
        accuracy = accuracy_score(y_train, y_pred)

        # 计算AUC（如果有概率预测）
        auc_score = None
        if hasattr(self.ensemble_model, 'predict_proba'):
            y_score = self.ensemble_model.predict_proba(X_train)[:, 1]
            fpr, tpr, _ = roc_curve(y_train, y_score)
            auc_score = auc(fpr, tpr)

        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'model': self.ensemble_model
        }


def main():
    """主函数"""
    print("=" * 70)
    print("金融大数据应用案例 - 量化选股 (Python版本)")
    print("=" * 70)

    # 创建结果目录
    results_dir = './results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 步骤1: 数据获取与预处理
    print("\n步骤1: 数据获取与预处理")
    print("-" * 40)

    processor = DataProcessor('./processed_data/')

    # 查找国泰安数据文件 (先在当前目录查找，因为您的文件在项目根目录)
    cta_files = glob.glob('TRD_Dalyr*.xlsx')
    if not cta_files:
        # 再查找raw_data目录
        cta_files = glob.glob('./raw_data/TRD_Dalyr*.xlsx')
    if not cta_files:
        print("未找到国泰安数据文件，正在生成模拟数据用于演示...")
        generate_sample_data(200, './raw_data/')
        cta_files = glob.glob('./raw_data/TRD_Dalyr*.xlsx')


    print(f"找到 {len(cta_files)} 个国泰安数据文件")

    # 处理数据
    all_stocks_data = processor.process_cta_data(cta_files)

    if not all_stocks_data:
        print("数据预处理失败，请检查数据文件")
        return

    # 步骤2: 计算指标
    print("\n步骤2: 计算技术指标")
    print("-" * 40)

    indicator_calculator = TechnicalIndicators()

    # 计算每支股票的指标
    stocks_with_indicators = {}
    for stock_code in list(all_stocks_data.keys())[:200]:  # 取前200支股票
        stock_df = processor.load_stock_data(stock_code)
        if stock_df is not None:
            df_with_indicators = indicator_calculator.calculate_indicators(stock_df)
            if df_with_indicators is not None and len(df_with_indicators) > 50:
                stocks_with_indicators[stock_code] = df_with_indicators

    print(f"成功计算了 {len(stocks_with_indicators)} 支股票的技术指标")

    # 创建训练样本和预测样本
    train_good, train_bad, forecast = indicator_calculator.create_samples(
        stocks_with_indicators, n_calculations=18
    )

    print(f"好样本数量: {len(train_good)}")
    print(f"坏样本数量: {len(train_bad)}")
    print(f"预测样本数量: {len(forecast)}")

    # 合并训练样本
    train_data = pd.concat([train_good, train_bad], ignore_index=True)

    # 保存原始样本
    train_data.to_excel('./results/原始训练样本.xlsx', index=False)
    forecast.to_excel('./results/原始预测样本.xlsx', index=False)

    print("原始样本已保存到 ./results/ 目录")

    # 步骤3: 数据标准化
    print("\n步骤3: 数据标准化")
    print("-" * 40)

    standardizer = DataStandardizer()

    # 标准化训练数据
    scaled_train = standardizer.min_max_scale(train_data)
    # 使用训练数据参数标准化预测数据
    scaled_forecast = standardizer.scale_forecast_data(forecast)

    # 保存标准化后的数据
    scaled_train.to_excel('./results/标准化训练样本.xlsx', index=False)
    scaled_forecast.to_excel('./results/标准化预测样本.xlsx', index=False)

    print(f"训练样本标准化完成: {len(scaled_train)} 条记录")
    print(f"预测样本标准化完成: {len(scaled_forecast)} 条记录")

    # 步骤4: 变量筛选
    print("\n步骤4: 变量筛选")
    print("-" * 40)

    feature_selector = FeatureSelector(threshold=0.2)
    selected_features = feature_selector.select_features(scaled_train)

    # 筛选数据
    X_train = feature_selector.filter_data(scaled_train)
    y_train = scaled_train['s_y']

    X_forecast = feature_selector.filter_data(scaled_forecast)

    print(f"筛选后特征数量: {len(selected_features)}")
    print(f"训练数据形状: {X_train.shape}")
    print(f"预测数据形状: {X_forecast.shape}")

    # 保存筛选后的数据
    filtered_train = pd.concat([X_train, y_train], axis=1)
    filtered_train.to_excel('./results/筛选后训练样本.xlsx', index=False)

    filtered_forecast = X_forecast.copy()
    if '股票代码' in forecast.columns:
        filtered_forecast['股票代码'] = forecast['股票代码'].values
    filtered_forecast.to_excel('./results/筛选后预测样本.xlsx', index=False)

    # 步骤5: 模型的建立与评估
    print("\n步骤5: 模型的建立与评估")
    print("-" * 40)

    # 划分训练集和测试集
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )

    print(f"训练集大小: {X_train_split.shape}")
    print(f"测试集大小: {X_test_split.shape}")

    # 创建预测器
    predictor = StockPredictor()

    # 训练和评估模型
    results = predictor.train_and_evaluate(
        X_train_split, y_train_split,
        X_test_split, y_test_split
    )

    # 评估模型
    eval_df = predictor.evaluate_models()

    # 绘制ROC曲线
    predictor.plot_roc_curves('./results/roc_curves.png')

    # 预测样本
    forecast_predictions = predictor.predict_forecast(X_forecast)

    # 获取预测样本的股票代码
    forecast_stock_codes = forecast['股票代码'].values if '股票代码' in forecast.columns else [f'Stock_{i}' for i in
                                                                                               range(len(forecast))]

    # 保存预测结果
    predictor.save_forecast_results(forecast_predictions, forecast_stock_codes, './results/')

    # 探索题1: 额外评估方法
    print("\n" + "=" * 70)
    print("探索题1: 额外评估方法")
    print("=" * 70)

    for model_name, result in results.items():
        print(f"\n{model_name} 额外评估:")
        print("-" * 30)

        # 混淆矩阵
        cm = confusion_matrix(result['y_train_true'], result['y_train_pred'])
        print("混淆矩阵:")
        print(cm)

        # 计算精确率、召回率、F1分数
        report = classification_report(
            result['y_train_true'],
            result['y_train_pred'],
            target_names=['跌', '涨'],
            output_dict=True
        )

        print("\n分类报告:")
        print(f"精确率 (涨): {report['涨']['precision']:.4f}")
        print(f"召回率 (涨): {report['涨']['recall']:.4f}")
        print(f"F1分数 (涨): {report['涨']['f1-score']:.4f}")

    # 探索题2: 组合选举法
    print("\n" + "=" * 70)
    print("探索题2: 组合选举法")
    print("=" * 70)

    ensemble_predictor = EnsemblePredictor(predictor)
    ensemble_result = ensemble_predictor.evaluate_ensemble(X_train_split, y_train_split)

    if ensemble_result:
        print(f"集成模型准确率: {ensemble_result['accuracy']:.4f}")
        if ensemble_result['auc']:
            print(f"集成模型AUC值: {ensemble_result['auc']:.4f}")

    # 集成预测
    ensemble_forecast = ensemble_predictor.ensemble_predict(X_train_split, y_train_split, X_forecast)

    # 保存集成预测结果
    ensemble_forecast_df = pd.DataFrame({
        '股票代码': forecast_stock_codes,
        '集成预测概率': ensemble_forecast[:, 1] if ensemble_forecast.ndim == 2 else ensemble_forecast,
        '集成预测标签': ['涨' if p[1] > 0.5 else '跌' for p in ensemble_forecast] if ensemble_forecast.ndim == 2 else [
            '涨' if p > 0.5 else '跌' for p in ensemble_forecast],
        '预测日期': datetime.now().strftime('%Y-%m-%d')
    })

    # 排序：预测概率高的在前
    ensemble_forecast_df = ensemble_forecast_df.sort_values('集成预测概率', ascending=False)
    ensemble_forecast_df.to_excel('./results/ensemble_forecast_result.xlsx', index=False)
    print("集成预测结果已保存到: ./results/ensemble_forecast_result.xlsx")

    # 最终选股推荐
    print("\n" + "=" * 70)
    print("最终选股推荐")
    print("=" * 70)

    # 根据集成预测结果推荐股票
    recommended_stocks = ensemble_forecast_df[ensemble_forecast_df['集成预测标签'] == '涨'].copy()

    print(f"推荐买入的股票数量: {len(recommended_stocks)}")
    print("\n推荐股票列表 (前20支):")
    print(recommended_stocks[['股票代码', '集成预测概率']].head(20).to_string(index=False))

    # 保存最终选股结果
    recommended_stocks.to_excel('./results/final_stock_recommendations.xlsx', index=False)

    print("\n" + "=" * 70)
    print("量化选股程序执行完成!")
    print("所有结果已保存到 ./results/ 目录")
    print("=" * 70)

    return {
        'num_stocks': len(stocks_with_indicators),
        'train_samples': len(train_data),
        'forecast_samples': len(forecast),
        'selected_features': selected_features,
        'model_evaluation': eval_df,
        'recommended_stocks': recommended_stocks
    }


def generate_sample_data(num_stocks=200, output_dir='./raw_data/'):
    """生成模拟的国泰安格式数据用于测试"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在生成 {num_stocks} 支股票的模拟数据...")

    # 创建示例数据文件
    sample_data = []

    for i in range(num_stocks):
        stock_code = 600000 + i  # 从600000开始

        # 生成一年（约250个交易日）的数据
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=j) for j in range(250)]

        # 生成价格序列（随机游走）
        base_price = np.random.uniform(5, 50)
        returns = np.random.normal(0.0005, 0.02, 250)
        prices = base_price * np.exp(np.cumsum(returns))

        # 生成OHLCV数据
        for j, date in enumerate(dates):
            open_price = prices[j] * np.random.uniform(0.98, 1.02)
            close_price = prices[j] * np.random.uniform(0.98, 1.02)
            high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.05)
            low_price = min(open_price, close_price) * np.random.uniform(0.95, 1.0)
            volume = np.random.randint(1000000, 10000000)

            sample_data.append({
                'Stkcd': stock_code,
                'Trddt': date.strftime('%Y-%m-%d'),
                'Opnprc': round(open_price, 2),
                'Hiprc': round(high_price, 2),
                'Loprc': round(low_price, 2),
                'Clsprc': round(close_price, 2),
                'Dnshrtrd': volume
            })

        if (i + 1) % 20 == 0:
            print(f"已生成 {i + 1} 支股票数据")

    # 转换为DataFrame
    df = pd.DataFrame(sample_data)

    # 分成多个文件保存（模拟多个TRD_Dalyr文件）
    file_size = len(df) // 6
    for i in range(6):
        start_idx = i * file_size
        end_idx = (i + 1) * file_size if i < 5 else len(df)

        file_df = df.iloc[start_idx:end_idx]
        file_path = os.path.join(output_dir, f'TRD_Dalyr{i}.xlsx')
        file_df.to_excel(file_path, index=False)

    print(f"模拟数据已生成到 {output_dir} 目录")
    print("文件列表:")
    for file in glob.glob(os.path.join(output_dir, 'TRD_Dalyr*.xlsx')):
        print(f"  {os.path.basename(file)}")


if __name__ == "__main__":
    # 检查是否有原始数据，如果没有则生成模拟数据
    cta_files = glob.glob('./raw_data/TRD_Dalyr*.xlsx')

    if not cta_files:
        print("未找到原始数据文件，生成模拟数据用于演示...")
        generate_sample_data(200, './raw_data/')

    # 运行主程序
    results = main()

    # 生成总结报告
    print("\n生成总结报告...")
    report_file = './results/量化选股分析报告.txt'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("量化选股分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析股票数量: {results['num_stocks']}\n")
        f.write(f"训练样本数量: {results['train_samples']}\n")
        f.write(f"预测样本数量: {results['forecast_samples']}\n")
        f.write(f"筛选特征数量: {len(results['selected_features'])}\n")
        f.write(f"推荐股票数量: {len(results['recommended_stocks'])}\n\n")

        f.write("选中的特征:\n")
        for feature in results['selected_features']:
            f.write(f"  {feature}\n")
        f.write("\n")

        f.write("模型评估结果:\n")
        f.write(results['model_evaluation'].to_string(index=False))
        f.write("\n\n")

        f.write("推荐股票列表 (前10支):\n")
        f.write(results['recommended_stocks'][['股票代码', '集成预测概率']].head(10).to_string(index=False))

    print(f"总结报告已保存到: {report_file}")