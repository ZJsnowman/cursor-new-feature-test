#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电商机器学习演示脚本
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 数据生成
def generate_ecommerce_data():
    """生成模拟电商数据"""
    np.random.seed(42)
    
    # 用户数据
    n_users = 1000
    users = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'age': np.random.normal(35, 12, n_users).astype(int).clip(18, 70),
        'gender': np.random.choice(['M', 'F'], n_users),
        'city_tier': np.random.choice([1, 2, 3], n_users),
        'income_level': np.random.choice(['low', 'medium', 'high'], n_users)
    })
    
    # 商品数据
    n_products = 200
    categories = ['电子产品', '服装', '家居', '食品', '图书']
    products = pd.DataFrame({
        'product_id': range(1, n_products + 1),
        'category': np.random.choice(categories, n_products),
        'price': np.random.lognormal(4, 1, n_products),
        'rating': np.random.normal(4.0, 0.8, n_products).clip(1, 5)
    })
    
    # 交易数据
    n_transactions = 5000
    transactions = []
    for _ in range(n_transactions):
        user_id = np.random.choice(users['user_id'])
        product_id = np.random.choice(products['product_id'])
        quantity = np.random.poisson(2) + 1
        
        product_price = products[products['product_id'] == product_id]['price'].iloc[0]
        total_amount = product_price * quantity
        
        transactions.append({
            'user_id': user_id,
            'product_id': product_id,
            'quantity': quantity,
            'total_amount': total_amount,
            'date': datetime.now() - timedelta(days=np.random.randint(1, 365))
        })
    
    transactions_df = pd.DataFrame(transactions)
    
    return users, products, transactions_df

# 推荐系统
class SimpleRecommender:
    """简单推荐系统"""
    
    def __init__(self):
        self.user_item_matrix = None
        
    def fit(self, transactions_df):
        """训练推荐模型"""
        self.user_item_matrix = transactions_df.pivot_table(
            index='user_id', 
            columns='product_id', 
            values='quantity', 
            fill_value=0
        )
        
    def recommend(self, user_id, n_recommendations=5):
        """推荐商品"""
        if user_id not in self.user_item_matrix.index:
            return []
            
        user_purchases = self.user_item_matrix.loc[user_id]
        similar_users = self.user_item_matrix.corrwith(user_purchases, axis=1).dropna()
        similar_users = similar_users.sort_values(ascending=False)[1:6]
        
        recommendations = []
        for similar_user in similar_users.index:
            user_items = self.user_item_matrix.loc[similar_user]
            for product_id in user_items[user_items > 0].index:
                if user_purchases[product_id] == 0 and product_id not in recommendations:
                    recommendations.append(product_id)
                    if len(recommendations) >= n_recommendations:
                        break
            if len(recommendations) >= n_recommendations:
                break
                
        return recommendations

# 销量预测
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SalesPredictor:
    """销量预测模型"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def prepare_features(self, products_df, transactions_df):
        """准备特征"""
        # 计算商品销量统计
        sales_stats = transactions_df.groupby('product_id').agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        })
        
        # 合并商品特征
        features_df = products_df.merge(sales_stats, on='product_id', how='left')
        features_df = features_df.fillna(0)
        
        # 编码分类特征
        category_dummies = pd.get_dummies(features_df['category'])
        features_df = pd.concat([features_df, category_dummies], axis=1)
        
        # 选择特征列
        feature_cols = ['price', 'rating'] + list(category_dummies.columns)
        X = features_df[feature_cols]
        y = features_df['quantity']
        
        return X, y
        
    def train(self, products_df, transactions_df):
        """训练模型"""
        X, y = self.prepare_features(products_df, transactions_df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"销量预测模型 MSE: {mse:.2f}")
        
        return mse

def main():
    """主函数"""
    print("🛒 电商机器学习演示")
    print("=" * 50)
    
    # 生成数据
    print("📊 生成模拟数据...")
    users, products, transactions = generate_ecommerce_data()
    
    print(f"用户数: {len(users)}")
    print(f"商品数: {len(products)}")
    print(f"交易数: {len(transactions)}")
    
    # 数据分析
    print("\n📈 数据分析...")
    
    # 最受欢迎的商品类别
    category_sales = transactions.merge(products, on='product_id')['category'].value_counts()
    print("\n最受欢迎的商品类别:")
    for category, count in category_sales.head().items():
        print(f"  {category}: {count}次购买")
    
    # 用户消费分析
    user_spending = transactions.groupby('user_id')['total_amount'].sum()
    print(f"\n用户平均消费: ¥{user_spending.mean():.2f}")
    print(f"用户最高消费: ¥{user_spending.max():.2f}")
    
    # 推荐系统演示
    print("\n🎯 推荐系统演示...")
    recommender = SimpleRecommender()
    recommender.fit(transactions)
    
    # 为前5个用户生成推荐
    for user_id in range(1, 6):
        recommendations = recommender.recommend(user_id)
        print(f"用户 {user_id} 推荐商品: {recommendations}")
    
    # 销量预测演示
    print("\n📊 销量预测演示...")
    predictor = SalesPredictor()
    mse = predictor.train(products, transactions)
    
    print("\n✅ 演示完成!")

if __name__ == "__main__":
    main()