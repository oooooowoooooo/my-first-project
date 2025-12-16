# -*- coding: utf-8 -*-
"""
波士顿房价预测模型

"""

# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
plt.style.use('ggplot')  # 使用ggplot样式

# 创建输出目录用于保存图表
import os

output_dir = 'visualization'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def load_data(file_path):
    """
    加载数据集
    :param file_path: 数据文件路径
    :return: 加载后的DataFrame
    """
    try:
        # 尝试使用空格分隔符读取（因为波士顿数据集通常是空格分隔）
        df = pd.read_csv(file_path, sep='\s+', skipinitialspace=True)
        print(f"✓ 数据加载成功，共{df.shape[0]}行，{df.shape[1]}列")

        # 检查列数是否正确（波士顿数据集应该有14列）
        if df.shape[1] == 14:
            # 添加列名
            column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
            df.columns = column_names
            print(f"✓ 已添加列名: {list(df.columns)}")
        else:
            print(f"⚠ 数据列数为{df.shape[1]}，可能不是标准的波士顿数据集格式")

        return df
    except Exception as e:
        print(f"✗ 数据加载失败: {str(e)}")
        return None


def preprocess_data(df):
    """
    预处理数据
    :param df: 原始数据DataFrame
    :return: 特征矩阵X、目标变量y、标准化器
    """
    try:
        # 检查是否包含目标变量列
        if 'MEDV' not in df.columns:
            print("✗ 数据中不包含目标变量列MEDV")
            return None, None, None

        # 分离特征和目标变量
        X = df.drop('MEDV', axis=1)
        y = df['MEDV']
        print(f"✓ 已分离特征和目标变量")

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"✓ 数据标准化完成")

        return X_scaled, y, scaler
    except Exception as e:
        print(f"✗ 数据预处理失败: {str(e)}")
        return None, None, None


def train_model(X_train, y_train):
    """
    训练模型
    :param X_train: 训练特征
    :param y_train: 训练目标变量
    :return: 训练好的模型
    """
    try:
        # 使用Gradient Boosting模型（性能较好）
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        print(f"✓ 模型训练完成")
        return model
    except Exception as e:
        print(f"✗ 模型训练失败: {str(e)}")
        return None


def visualize_data(df):
    """
    数据可视化分析
    :param df: 原始数据集
    """
    print(f"\n=== 数据可视化分析 ===")

    # 1. 目标变量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['MEDV'], bins=30, kde=True, alpha=0.7)
    plt.title('房价分布直方图')
    plt.xlabel('房价（千美元）')
    plt.ylabel('频数')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_distribution.png'), dpi=300)
    plt.close()
    print(f"✓ 房价分布直方图已保存")

    # 2. 主要特征与目标变量的关系
    key_features = ['RM', 'LSTAT', 'PTRATIO', 'DIS']
    plt.figure(figsize=(12, 10))

    for i, feature in enumerate(key_features, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x=df[feature], y=df['MEDV'], alpha=0.6)
        sns.regplot(x=df[feature], y=df['MEDV'], scatter=False, color='red')
        plt.title(f'{feature}与房价的关系')
        plt.xlabel(feature)
        plt.ylabel('房价（千美元）')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_vs_price.png'), dpi=300)
    plt.close()
    print(f"✓ 特征与房价关系图已保存")

    # 3. 相关性热力图
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    print(f"✓ 特征相关性热力图已保存")


def visualize_model_performance(model, X_test, y_test, y_pred):
    """
    模型性能可视化
    :param model: 训练好的模型
    :param X_test: 测试特征
    :param y_test: 测试目标变量
    :param y_pred: 预测值
    """
    print(f"\n=== 模型性能可视化 ===")

    # 1. 预测值与实际值散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('预测值与实际值散点图')
    plt.xlabel('实际房价（千美元）')
    plt.ylabel('预测房价（千美元）')
    plt.xlim(y_test.min() - 1, y_test.max() + 1)
    plt.ylim(y_test.min() - 1, y_test.max() + 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'), dpi=300)
    plt.close()
    print(f"✓ 预测值与实际值散点图已保存")

    # 2. 残差分布图
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=20, kde=True, alpha=0.7)
    plt.title('模型残差分布图')
    plt.xlabel('残差（千美元）')
    plt.ylabel('频数')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_distribution.png'), dpi=300)
    plt.close()
    print(f"✓ 残差分布图已保存")

    # 3. 残差与预测值关系图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.title('残差与预测值关系图')
    plt.xlabel('预测房价（千美元）')
    plt.ylabel('残差（千美元）')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_vs_prediction.png'), dpi=300)
    plt.close()
    print(f"✓ 残差与预测值关系图已保存")


def visualize_feature_importance(model, feature_names):
    """
    特征重要性可视化
    :param model: 训练好的模型
    :param feature_names: 特征名称列表
    """
    print(f"\n=== 特征重要性可视化 ===")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title('特征重要性排序')
        plt.xlabel('重要性分数')
        plt.ylabel('特征名称')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
        plt.close()
        print(f"✓ 特征重要性图已保存")

        # 打印前5个重要特征
        print(f"\n前5个最重要的特征：")
        for _, row in feature_importance.head().iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    else:
        print(f"当前模型不支持特征重要性分析")


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    :param model: 训练好的模型
    :param X_test: 测试特征
    :param y_test: 测试目标变量
    :return: 评估指标（MSE, RMSE, R2）
    """
    try:
        # 进行预测
        y_pred = model.predict(X_test)

        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"\n=== 模型评估结果 ===")
        print(f"MSE (均方误差): {mse:.4f}")
        print(f"RMSE (均方根误差): {rmse:.4f}")
        print(f"MAE (平均绝对误差): {mae:.4f}")
        print(f"R2 (决定系数): {r2:.4f}")

        return mse, rmse, r2, mae, y_pred
    except Exception as e:
        print(f"✗ 模型评估失败: {str(e)}")
        return None, None, None, None, None


def predict_house_price(model, scaler, new_data):
    """
    预测房价
    :param model: 训练好的模型
    :param scaler: 数据标准化器
    :param new_data: 新的房屋特征数据
    :return: 预测的房价
    """
    try:
        # 标准化新数据
        new_data_scaled = scaler.transform(new_data)

        # 进行预测
        prediction = model.predict(new_data_scaled)

        return prediction[0]
    except Exception as e:
        print(f"✗ 房价预测失败: {str(e)}")
        return None


def main():
    """
    主函数，执行完整的预测流程
    """
    print("=" * 50)
    print("波士顿房价预测模型")
    print("=" * 50)

    # 1. 设置数据文件路径
    # 注意：请根据实际情况修改数据文件路径
    file_path = r'C:\Users\hp\Desktop\杂七杂八的东西\波士顿房价-数据集\train.csv'

    # 2. 加载数据
    df = load_data(file_path)
    if df is None:
        return

    # 3. 数据可视化分析
    visualize_data(df)

    # 4. 预处理数据
    X_scaled, y, scaler = preprocess_data(df)
    if X_scaled is None:
        return

    # 5. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"✓ 已划分训练集和测试集")
    print(f"  训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 6. 训练模型
    model = train_model(X_train, y_train)
    if model is None:
        return

    # 7. 评估模型
    mse, rmse, r2, mae, y_pred = evaluate_model(model, X_test, y_test)

    # 8. 模型性能可视化
    visualize_model_performance(model, X_test, y_test, y_pred)

    # 9. 特征重要性分析
    visualize_feature_importance(model, df.columns[:-1])  # 排除目标变量MEDV

    # 10. 示例预测
    print(f"\n=== 示例预测 ===")

    # 创建一个示例房屋数据（特征值来源于波士顿数据集的第一条记录）
    example_house = np.array([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2,
                               4.0900, 1, 296, 15.3, 396.90, 4.98]])

    # 将示例数据转换为DataFrame，保持列名一致
    example_df = pd.DataFrame(example_house, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                                                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

    # 进行预测
    predicted_price = predict_house_price(model, scaler, example_df)

    if predicted_price is not None:
        print(f"示例房屋的预测房价为: ${predicted_price:.2f}千美元")
        print("（实际价格为24.0千美元）")

    print(f"\n=== 流程结束 ===")
    print("感谢使用波士顿房价预测模型！")
    print(f"可视化图表已保存到 {os.path.abspath(output_dir)} 目录")
    print("=" * 50)


# 执行主函数
if __name__ == "__main__":
    main()