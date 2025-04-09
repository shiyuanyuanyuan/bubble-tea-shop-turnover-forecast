import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Function to construct path relative to script directory
def data_path(filename):
    return os.path.join(script_dir, filename)

# 读取销售数据
sales_data_dec = pd.read_csv(data_path('sales_data_processed.csv'))
sales_data_feb = pd.read_csv(data_path('sales_data_processed_202502.csv'))  # 2月数据作为测试集

# 读取事件和假期数据
event_holiday = pd.read_csv(data_path('event_holiday_data.csv'))

# 读取天气数据
weather_data = pd.read_csv(data_path('combined_weather_data.csv'))

# 读取促销数据
promo_data = pd.read_csv(data_path('promotion_data.csv'))

# 统一处理日期和小时格式
def preprocess_datetime_hour(df, date_col='date', hour_col='hour'):
    df[date_col] = pd.to_datetime(df[date_col])
    df[hour_col] = df[hour_col].astype(str).str.zfill(2)
    return df

# 处理销售数据
sales_data_dec = preprocess_datetime_hour(sales_data_dec)
sales_data_feb = preprocess_datetime_hour(sales_data_feb)
sales_data_dec['weekday'] = sales_data_dec['date'].dt.dayofweek
sales_data_feb['weekday'] = sales_data_feb['date'].dt.dayofweek

# 处理事件假期数据
event_holiday = preprocess_datetime_hour(event_holiday)

# 处理天气数据
weather_data = preprocess_datetime_hour(weather_data)
# 选择所需的天气特征
weather_features = ['date', 'hour', 'tempreture', 'Clear', 'Clouds', 'Fog', 'Rain']
weather_data = weather_data[weather_features]

# 处理促销数据
promo_data = preprocess_datetime_hour(promo_data)
# 选择并重命名促销特征以避免冲突
promo_data = promo_data[['date', 'hour', 'isPromotion']]

# 计算促销影响 (这个函数现在可能不再需要，因为我们有直接的isPromotion标志，但保留以防万一或用于比较)
def calculate_promotion_impact(df):
    # 计算每个小时段的平均单价
    df['avg_price_per_order'] = df['amount'] / df['bill'].replace(0, np.nan)

    # 计算每个小时段的正常单价（使用中位数作为基准）
    normal_price = df.groupby('hour')['avg_price_per_order'].median()
    df['normal_price'] = df['hour'].map(normal_price)

    # 计算促销影响（实际单价与正常单价的比率）
    df['promotion_impact'] = df['avg_price_per_order'] / df['normal_price']

    # 处理无穷大和NaN值
    df['promotion_impact'] = df['promotion_impact'].replace([np.inf, -np.inf], np.nan)
    df['promotion_impact'] = df['promotion_impact'].fillna(1.0)  # 没有订单的时段假设无促销影响

    return df

# 添加促销影响特征 (可以选择性保留或移除)
sales_data_dec = calculate_promotion_impact(sales_data_dec)
sales_data_feb = calculate_promotion_impact(sales_data_feb)

# 合并所有数据
def merge_all_data(sales_df, event_df, weather_df, promo_df):
    # 确保合并键类型一致 (date: datetime, hour: string)
    # sales_df, event_df, weather_df, promo_df 已经通过 preprocess_datetime_hour 处理

    # 1. 合并销售和事件/假期
    merged = pd.merge(
        sales_df,
        event_df[['date', 'hour', 'isEvent', 'isHoliday']],
        on=['date', 'hour'],
        how='left'
    )
    merged['isEvent'] = merged['isEvent'].fillna(0)
    merged['isHoliday'] = merged['isHoliday'].fillna(0)

    # 2. 合并天气数据
    merged = pd.merge(
        merged,
        weather_df,
        on=['date', 'hour'],
        how='left'
    )
    # 填充可能的天气数据缺失值（例如用0填充或更复杂的方法如插值/前向填充）
    weather_cols_to_fill = ['tempreture', 'Clear', 'Clouds', 'Fog', 'Rain']
    merged[weather_cols_to_fill] = merged[weather_cols_to_fill].fillna(0) # 简单填充0，可能需要调整

    # 3. 合并促销数据
    merged = pd.merge(
        merged,
        promo_df,
        on=['date', 'hour'],
        how='left'
    )
    merged['isPromotion'] = merged['isPromotion'].fillna(0) # 假设无记录=无促销

    return merged

# 合并训练数据和测试数据
train_data = merge_all_data(sales_data_dec, event_holiday, weather_data, promo_data)
test_data = merge_all_data(sales_data_feb, event_holiday, weather_data, promo_data)

# 创建特征
def create_features(df):
    # 创建小时的独热编码
    hour_dummies = pd.get_dummies(df['hour'], prefix='hour')

    # 创建星期几的独热编码
    weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday')

    # --- 对 isPromotion 进行独热编码 (修正版) ---
    # 1. 定义所有可能的促销类别 (作为字符串)
    all_promo_categories = ['0.0', '0.2', '0.5'] # Add any other possible values if they exist

    # 2. 将 isPromotion 转换为字符串类别
    df['isPromotion_cat'] = df['isPromotion'].astype(str)

    # 3. 将其转换为 Categorical 类型，并指定所有可能的类别
    df['isPromotion_cat'] = pd.Categorical(
        df['isPromotion_cat'],
        categories=all_promo_categories,
        ordered=False
    )

    # 4. 现在进行独热编码，它将始终为所有定义的类别创建列
    promotion_dummies = pd.get_dummies(df['isPromotion_cat'], prefix='promo')
    # print(f"Promotion Dummies for this slice: {promotion_dummies.columns.tolist()}")
    # ------------------------------------------

    # 选择要包含的基础和新特征
    # !! 移除了原始的 'isPromotion' !!
    feature_columns = [
        'isEvent', 'isHoliday', 'bill', 'promotion_impact', # 原有特征 (promotion_impact可选)
        # 'isPromotion', # <--- Removed original numerical feature
        'tempreture', 'Clear', 'Clouds', 'Fog', 'Rain' # 新增天气特征
    ]

    # 检查特征列是否存在，可能因为合并失败而缺失
    existing_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"警告: 特征列 {missing_features} 在DataFrame中缺失，将不会被包含。")

    # 组合所有特征 (小时，星期，基础特征，以及新的促销独热编码)
    features = pd.concat([
        hour_dummies,
        weekday_dummies,
        df[existing_features],
        promotion_dummies # <--- Added promotion dummies
    ], axis=1)

    # --- 新增：确保训练和测试集特征列一致 ---
    # (虽然 Categorical 方法应该能保证，但作为安全措施或替代方法可以取消注释)
    # global train_columns # 如果在函数外定义了 train_columns
    # if 'train_columns' in globals():
    #     features = features.reindex(columns=train_columns, fill_value=0)
    # ------------------------------------------

    return features

# 准备训练数据
X_train = create_features(train_data)
y_train = train_data['amount']

# --- 新增：存储训练列以供对齐 --- (配合上面 reindex 的安全措施)
# train_columns = X_train.columns.tolist()
# print(f"Stored training columns: {train_columns}")
# ----------------------------------

# 准备测试数据
# 确保测试数据有 'amount' 列，如果没有 (例如纯粹预测未来)，需要调整
if 'amount' in test_data.columns:
    X_test = create_features(test_data)
    y_test = test_data['amount']
else:
    print("警告: 测试数据中缺少 'amount' 列，无法进行评估。")
    X_test = create_features(test_data)
    y_test = None # 或者根据需要处理

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型 (仅当y_test存在时)
if y_test is not None:
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"模型评估结果：")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # 分析特征重要性
    feature_names = X_train.columns
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    coefficients = coefficients.sort_values('Coefficient', key=abs, ascending=False) # 按系数绝对值排序

    print("\n最重要的特征 (按影响大小排序)：")
    print(coefficients.head(15)) # 显示更多特征

    # 可视化实际值vs预测值
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际销售额')
    plt.ylabel('预测销售额')
    plt.title('实际销售额 vs 预测销售额 (含天气和促销数据)')
    plt.savefig(data_path('prediction_results_v2.png')) # 保存为新文件名
    plt.close()
    print("\n图表 prediction_results_v2.png 已保存")

# 保存预测结果
results_df = pd.DataFrame({
    'Date': test_data['date'],
    'Hour': test_data['hour'],
    'Actual': y_test if y_test is not None else np.nan, # 处理 y_test 可能不存在的情况
    'Predicted': y_pred
})
if y_test is not None:
    results_df['Difference'] = results_df['Actual'] - results_df['Predicted']
results_df['Promotion_Impact'] = test_data['promotion_impact'] # 原促销影响因子
results_df['isPromotion'] = test_data['isPromotion'] # 新促销标志
# 可以选择性加入天气特征到结果文件
# results = pd.concat([results, test_data[weather_features]], axis=1)

results_df.to_csv(data_path('prediction_results_v2.csv'), index=False) # 保存为新文件名
print("\n预测结果已保存到 prediction_results_v2.csv") 