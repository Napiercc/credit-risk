import numpy as np
import pandas as pd


class WOEEncoder:
    def __init__(self, bins=5, eps=1e-6):
        self.bins = bins
        self.eps = eps
        self.woe_maps = {}

    def fit(self, X, y):


        for col in X.columns:
            # 连续特征分箱
            if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                try:
                    bins = pd.qcut(X[col], self.bins, duplicates='drop')
                except:
                    bins = pd.qcut(X[col], self.bins - 1, duplicates='drop')

                grouped = pd.crosstab(bins, y)
            else:
                # 分类特征
                grouped = pd.crosstab(X[col], y)

            # 计算WOE
            grouped['good'] = grouped[0]
            grouped['bad'] = grouped[1]
            grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
            grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()

            # 平滑处理
            grouped['good_dist'] = grouped['good_dist'].replace(0, self.eps)
            grouped['bad_dist'] = grouped['bad_dist'].replace(0, self.eps)

            grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])

            # 保存WOE映射
            self.woe_maps[col] = grouped['woe'].to_dict()

        return self

    def transform(self, X):
        X_woe = X.copy()
        for col in X.columns:
            if col in self.woe_maps:
                # 连续特征：分箱后映射 WOE，转换为数值型
                if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    bins = pd.qcut(X[col], self.bins, duplicates='drop')
                    # 映射后转换为 float 类型（避免 Categorical）
                    X_woe[col] = bins.map(self.woe_maps[col]).astype(float)
                    # 分类特征：映射 WOE，确保为数值型
                else:
                    X_woe[col] = X[col].map(self.woe_maps[col]).astype(float)  # 显式转换类型

                # 处理缺失值（此时数据类型为数值型，可直接填充）
                X_woe[col] = X_woe[col].fillna(0)  # now safe!

        return X_woe