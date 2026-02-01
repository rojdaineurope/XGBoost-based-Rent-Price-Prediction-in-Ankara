import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
import joblib

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('dataset.csv')

ohe_cols = ['Tipi', 'Kullanım Durumu']

# Encoding

binanın_yaşı = ['0 (Yeni)', '1', '2', '3', '4', '5-10', '11-15', '16-20', '21 Ve Üzeri']
bulunduğu_kat = ['Kot 4 (-4).Kat','Kot 3 (-3).Kat','Kot 2 (-2).Kat','Kot 1 (-1).Kat','Bodrum Kat','Düz Giriş (Zemin)','Bahçe Katı','Müstakil','Yüksek Giriş','1.Kat','2.Kat','3.Kat','4.Kat','5.Kat','6.Kat','7.Kat','8.Kat','9.Kat','10.Kat','11.Kat','12.Kat','13.Kat','14.Kat','15.Kat','16.Kat','17.Kat','21.Kat','23.Kat','25.Kat','29.Kat','30.Kat','35.Kat', 'Çatı Katı', 'Çatı Dubleks' ]
site_içerisinde = ['Hayır', 'Evet']
ısıtma_tipi = ['Sobalı','Doğalgaz Sobalı','Klimalı','Kombi Fueloil','Merkezi Fueloil','Merkezi Doğalgaz','Merkezi (Pay Ölçer)','Kombi Doğalgaz','Yerden Isıtma']
banyo_sayısı = ['Yok','1','2','3','4','5','6+']
oda_sayısı = ['Stüdyo','1 Oda','1+1','2+0','1.5+1','2+1','3+0','2.5+1','3+1','2+2', '3.5+1','4+1','3+2','5+0','4.5+1','5+1','4+2','6+1','6+2','7+1','9+ Oda']
balkon_durumu = ['Yok', 'Var']


ordinal_cols = [
    "Binanın Yaşı", "Bulunduğu Kat", "Site İçerisinde",
    "Isıtma Tipi", "Banyo Sayısı", "Oda Sayısı", "Balkon Durumu"
]

ordinal_categories = [binanın_yaşı, bulunduğu_kat, site_içerisinde, ısıtma_tipi, banyo_sayısı, oda_sayısı, balkon_durumu]

nominal_cols = ['Ilce', 'Mahalle']


preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_cols),

        ('ordinal_manual', OrdinalEncoder(categories=ordinal_categories,
                                          handle_unknown='use_encoded_value',
                                          unknown_value=-1), ordinal_cols),

        ('ordinal_auto', OrdinalEncoder(handle_unknown='use_encoded_value',
                                        unknown_value=-1), nominal_cols)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

X = df.drop(["Fiyat"], axis=1)
y = df["Fiyat"]

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(n_estimators=1015,
        max_depth=7,
        learning_rate=0.005096682410107076,
        subsample=0.7862251490324859,
        colsample_bytree=0.5402146573293187,
        min_child_weight=1,
        reg_alpha=4.123074190413875,
        reg_lambda=2.8296705006332252,
        n_jobs=-1,
        random_state=42))
])

full_pipeline.fit(X, y)

joblib.dump(full_pipeline, 'model_pipeline.pkl')