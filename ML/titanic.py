"""
泰坦尼克号生存预测 - 随机森林算法
包含三次调参过程和结果记录
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
import warnings

# 只忽略“废弃警告”，其他警告仍显示
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data():
    """加载和预处理数据"""
    print("=" * 50)
    print("数据加载与预处理")
    print("=" * 50)

    # 加载数据
    train_path = r"C:\Users\hp\Desktop\混乱\泰坦尼克号-数据集\train.csv"
    test_path = r"C:\Users\hp\Desktop\混乱\泰坦尼克号-数据集\test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"训练集大小: {train_df.shape}")
    print(f"测试集大小: {test_df.shape}")

    # 保存测试集乘客ID
    test_ids = test_df['PassengerId']

    # 合并数据集以便统一处理
    combined = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)

    # 特征工程
    # 1. 从姓名中提取称谓
    combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # 将称谓归类
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    combined['Title'] = combined['Title'].map(title_mapping)

    # 2. 创建家庭规模特征
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1

    # 3. 创建是否独自出行特征
    combined['IsAlone'] = 0
    combined.loc[combined['FamilySize'] == 1, 'IsAlone'] = 1

    # 4. 处理缺失值
    # 年龄用中位数填充
    age_median = combined['Age'].median()
    combined['Age'].fillna(age_median, inplace=True)

    # 票价用中位数填充
    fare_median = combined['Fare'].median()
    combined['Fare'].fillna(fare_median, inplace=True)

    # 登船港口用众数填充
    embarked_mode = combined['Embarked'].mode()[0]
    combined['Embarked'].fillna(embarked_mode, inplace=True)

    # 5. 将年龄分组
    combined['AgeGroup'] = pd.cut(combined['Age'], bins=[0, 12, 18, 60, 100],
                                  labels=['Child', 'Teen', 'Adult', 'Senior'])

    # 6. 将票价分组
    combined['FareGroup'] = pd.qcut(combined['Fare'], 4, labels=['Low', 'Medium', 'High', 'VeryHigh'])

    # 7. 类别特征编码
    # 性别: male=0, female=1
    combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1})

    # 登船港口: S=0, C=1, Q=2
    combined['Embarked'] = combined['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # 称谓编码
    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    combined['Title'] = combined['Title'].map(title_mapping)

    # 年龄组编码
    age_mapping = {'Child': 0, 'Teen': 1, 'Adult': 2, 'Senior': 3}
    combined['AgeGroup'] = combined['AgeGroup'].map(age_mapping)

    # 票价组编码
    fare_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'VeryHigh': 3}
    combined['FareGroup'] = combined['FareGroup'].map(fare_mapping)

    # 选择特征
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'Title', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup']

    # 分割回训练集和测试集
    train_processed = combined[:len(train_df)]
    test_processed = combined[len(train_df):]

    X = train_processed[features]
    y = train_df['Survived']
    X_test = test_processed[features]

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test_scaled.shape}")

    return X_train, X_val, y_train, y_val, X_test_scaled, test_ids, scaler, features


def evaluate_model(model, X_train, X_val, y_train, y_val, model_name="随机森林"):
    """评估模型性能"""
    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    # 打印评估结果
    print(f"\n{model_name} 模型评估结果:")
    print(f"准确率(Accuracy): {accuracy:.4f}")
    print(f"精确率(Precision): {precision:.4f}")
    print(f"召回率(Recall): {recall:.4f}")
    print(f"F1分数(F1-Score): {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"5折交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['未幸存', '幸存'],
                yticklabels=['未幸存', '幸存'])
    plt.title(f'{model_name} 混淆矩阵')
    plt.ylabel('真实值')
    plt.xlabel('预测值')
    plt.savefig(f'random_forest_confusion_matrix.png')
    plt.close()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }


def random_forest_tuning_round_1(X_train, X_val, y_train, y_val):
    """随机森林调参第一轮：调整n_estimators"""
    print("\n" + "=" * 50)
    print("随机森林调参第一轮：调整n_estimators")
    print("=" * 50)

    # 测试不同的n_estimators值
    estimators_range = [10, 20, 50, 100, 150, 200, 300, 500]
    cv_scores = []

    for n in estimators_range:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
        print(f"n_estimators={n}: 5折交叉验证准确率: {scores.mean():.4f} (±{scores.std():.4f})")

    # 绘制不同n_estimators值的交叉验证准确率
    plt.figure(figsize=(10, 6))
    plt.plot(estimators_range, cv_scores, marker='o')
    plt.xlabel('n_estimators')
    plt.ylabel('5折交叉验证准确率')
    plt.title('随机森林: 不同n_estimators值的交叉验证准确率')
    plt.grid(True)
    plt.savefig('random_forest_tuning_round1_n_estimators.png')
    plt.close()

    # 找到最佳n_estimators值
    optimal_estimators = estimators_range[np.argmax(cv_scores)]
    print(f"最佳n_estimators值: {optimal_estimators}")
    print(f"最佳交叉验证准确率: {max(cv_scores):.4f}")

    # 使用最佳n_estimators值训练模型并评估
    best_rf = RandomForestClassifier(n_estimators=optimal_estimators, random_state=42)
    results = evaluate_model(best_rf, X_train, X_val, y_train, y_val, "随机森林(第一轮调参)")

    return optimal_estimators, results


def random_forest_tuning_round_2(X_train, X_val, y_train, y_val, optimal_estimators):
    """随机森林调参第二轮：调整max_depth和min_samples_split"""
    print("\n" + "=" * 50)
    print("随机森林调参第二轮：调整max_depth和min_samples_split")
    print("=" * 50)

    # 定义参数网格
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10, 15]
    }

    # 使用网格搜索
    rf = RandomForestClassifier(n_estimators=optimal_estimators, random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # 获取最佳参数
    best_params = grid_search.best_params_
    print(f"最佳参数组合: {best_params}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

    # 使用最佳参数训练模型并评估
    best_rf = RandomForestClassifier(
        n_estimators=optimal_estimators,
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
    )
    results = evaluate_model(best_rf, X_train, X_val, y_train, y_val, "随机森林(第二轮调参)")

    return best_params, results


def random_forest_tuning_round_3(X_train, X_val, y_train, y_val, optimal_estimators, best_params):
    """随机森林调参第三轮：调整max_features和min_samples_leaf"""
    print("\n" + "=" * 50)
    print("随机森林调参第三轮：调整max_features和min_samples_leaf")
    print("=" * 50)

    # 定义参数网格
    param_grid = {
        'max_features': ['auto', 'sqrt', 'log2', 0.3, 0.5, 0.7],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    # 使用网格搜索
    rf = RandomForestClassifier(
        n_estimators=optimal_estimators,
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
    )
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # 获取最佳参数
    best_params_3 = grid_search.best_params_
    print(f"最佳参数组合: {best_params_3}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

    # 使用最佳参数训练模型并评估
    best_rf = RandomForestClassifier(
        n_estimators=optimal_estimators,
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        max_features=best_params_3['max_features'],
        min_samples_leaf=best_params_3['min_samples_leaf'],
        random_state=42
    )
    results = evaluate_model(best_rf, X_train, X_val, y_train, y_val, "随机森林(第三轮调参)")

    # 特征重要性
    feature_importance = best_rf.feature_importances_
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                     'Title', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup']

    # 创建特征重要性数据框
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('随机森林特征重要性')
    plt.tight_layout()
    plt.savefig('random_forest_feature_importance.png')
    plt.close()

    print("\n特征重要性排序:")
    for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Importance']), 1):
        print(f"{i}. {feature}: {importance:.4f}")

    return best_params_3, results, importance_df


def generate_predictions(model, X_test, test_ids, filename='random_forest_submission.csv'):
    """生成测试集预测结果"""
    print("\n" + "=" * 50)
    print("生成测试集预测结果")
    print("=" * 50)

    # 预测测试集
    test_predictions = model.predict(X_test)

    # 创建提交文件
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': test_predictions
    })

    # 保存预测结果
    submission.to_csv(filename, index=False)
    print(f"预测结果已保存为 '{filename}'")

    # 预测统计
    survived_count = sum(test_predictions)
    not_survived_count = len(test_predictions) - survived_count
    print(f"预测幸存人数: {survived_count} ({survived_count / len(test_predictions) * 100:.2f}%)")
    print(f"预测未幸存人数: {not_survived_count} ({not_survived_count / len(test_predictions) * 100:.2f}%)")

    return submission


def save_results_to_file(results, filename='random_forest_results.txt'):
    """将调参结果保存到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("随机森林模型调参结果记录\n")
        f.write("=" * 50 + "\n\n")

        f.write("第一轮调参：调整n_estimators\n")
        f.write(f"最佳n_estimators值: {results['round1']['n_estimators']}\n")
        f.write(f"最佳交叉验证准确率: {results['round1']['cv_mean']:.4f} (±{results['round1']['cv_std']:.4f})\n")
        f.write(f"验证集准确率: {results['round1']['accuracy']:.4f}\n")
        f.write(f"验证集精确率: {results['round1']['precision']:.4f}\n")
        f.write(f"验证集召回率: {results['round1']['recall']:.4f}\n")
        f.write(f"验证集F1分数: {results['round1']['f1']:.4f}\n")
        f.write(f"验证集AUC-ROC: {results['round1']['auc']:.4f}\n\n")

        f.write("第二轮调参：调整max_depth和min_samples_split\n")
        f.write(f"最佳参数组合: {results['round2']['params']}\n")
        f.write(f"验证集准确率: {results['round2']['accuracy']:.4f}\n")
        f.write(f"验证集精确率: {results['round2']['precision']:.4f}\n")
        f.write(f"验证集召回率: {results['round2']['recall']:.4f}\n")
        f.write(f"验证集F1分数: {results['round2']['f1']:.4f}\n")
        f.write(f"验证集AUC-ROC: {results['round2']['auc']:.4f}\n\n")

        f.write("第三轮调参：调整max_features和min_samples_leaf\n")
        f.write(f"最佳参数组合: {results['round3']['params']}\n")
        f.write(f"验证集准确率: {results['round3']['accuracy']:.4f}\n")
        f.write(f"验证集精确率: {results['round3']['precision']:.4f}\n")
        f.write(f"验证集召回率: {results['round3']['recall']:.4f}\n")
        f.write(f"验证集F1分数: {results['round3']['f1']:.4f}\n")
        f.write(f"验证集AUC-ROC: {results['round3']['auc']:.4f}\n\n")

        f.write("特征重要性排序:\n")
        for i, (feature, importance) in enumerate(zip(results['round3']['importance_df']['Feature'],
                                                      results['round3']['importance_df']['Importance']), 1):
            f.write(f"{i}. {feature}: {importance:.4f}\n")

        f.write("\n最终模型性能总结\n")
        f.write(f"最佳参数组合: n_estimators={results['round1']['n_estimators']}, ")
        f.write(f"max_depth={results['round2']['params']['max_depth']}, ")
        f.write(f"min_samples_split={results['round2']['params']['min_samples_split']}, ")
        f.write(f"max_features={results['round3']['params']['max_features']}, ")
        f.write(f"min_samples_leaf={results['round3']['params']['min_samples_leaf']}\n")
        f.write(f"最终验证集准确率: {results['round3']['accuracy']:.4f}\n")
        f.write(f"最终验证集精确率: {results['round3']['precision']:.4f}\n")
        f.write(f"最终验证集召回率: {results['round3']['recall']:.4f}\n")
        f.write(f"最终验证集F1分数: {results['round3']['f1']:.4f}\n")
        f.write(f"最终验证集AUC-ROC: {results['round3']['auc']:.4f}\n")

    print(f"调参结果已保存到 '{filename}'")


def main():
    """主函数"""
    # 加载和预处理数据
    X_train, X_val, y_train, y_val, X_test, test_ids, scaler, features = load_and_preprocess_data()

    # 存储每轮调参的结果
    results = {}

    # 第一轮调参：调整n_estimators
    optimal_estimators, round1_results = random_forest_tuning_round_1(X_train, X_val, y_train, y_val)
    results['round1'] = {
        'n_estimators': optimal_estimators,
        'accuracy': round1_results['accuracy'],
        'precision': round1_results['precision'],
        'recall': round1_results['recall'],
        'f1': round1_results['f1'],
        'auc': round1_results['auc'],
        'cv_mean': round1_results['cv_mean'],
        'cv_std': round1_results['cv_std']
    }

    # 第二轮调参：调整max_depth和min_samples_split
    best_params_2, round2_results = random_forest_tuning_round_2(X_train, X_val, y_train, y_val, optimal_estimators)
    results['round2'] = {
        'params': best_params_2,
        'accuracy': round2_results['accuracy'],
        'precision': round2_results['precision'],
        'recall': round2_results['recall'],
        'f1': round2_results['f1'],
        'auc': round2_results['auc']
    }

    # 第三轮调参：调整max_features和min_samples_leaf
    best_params_3, round3_results, importance_df = random_forest_tuning_round_3(
        X_train, X_val, y_train, y_val, optimal_estimators, best_params_2)
    results['round3'] = {
        'params': best_params_3,
        'accuracy': round3_results['accuracy'],
        'precision': round3_results['precision'],
        'recall': round3_results['recall'],
        'f1': round3_results['f1'],
        'auc': round3_results['auc'],
        'importance_df': importance_df
    }

    # 使用最佳参数训练最终模型
    print("\n" + "=" * 50)
    print("使用最佳参数训练最终随机森林模型")
    print("=" * 50)
    final_rf = RandomForestClassifier(
        n_estimators=optimal_estimators,
        max_depth=best_params_2['max_depth'],
        min_samples_split=best_params_2['min_samples_split'],
        max_features=best_params_3['max_features'],
        min_samples_leaf=best_params_3['min_samples_leaf'],
        random_state=42
    )
    final_rf.fit(X_train, y_train)

    # 生成测试集预测
    submission = generate_predictions(final_rf, X_test, test_ids, 'random_forest_submission.csv')

    # 保存调参结果
    save_results_to_file(results, 'random_forest_results.txt')

    print("\n" + "=" * 50)
    print("随机森林模型调参与预测完成")
    print("=" * 50)
    print(f"最终模型参数: n_estimators={optimal_estimators}, max_depth={best_params_2['max_depth']}, ")
    print(f"min_samples_split={best_params_2['min_samples_split']}, max_features={best_params_3['max_features']}, ")
    print(f"min_samples_leaf={best_params_3['min_samples_leaf']}")
    print(f"最终验证集准确率: {round3_results['accuracy']:.4f}")
    print(f"最终验证集AUC-ROC: {round3_results['auc']:.4f}")


if __name__ == "__main__":
    main()