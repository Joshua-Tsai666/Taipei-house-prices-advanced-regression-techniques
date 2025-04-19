import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 讀取資料
df = pd.read_csv("Taipei_house.csv")

# 2. 分出特徵（X）和目標（y）
X = df.drop(columns=["總價", "交易日期"])  # 把總價跟日期移除，剩下的拿來預測
y = df["總價"]

# 3. 指定哪些是文字欄位（需要轉換成數字）
text_columns = ["行政區", "車位類別"]

# 4. 建立前處理流程（把文字轉成數字）
preprocess = ColumnTransformer(
    transformers=[
        ("text", OneHotEncoder(handle_unknown="ignore"), text_columns)
    ],
    remainder="passthrough"  # 其他數字欄位保持不變
)

# 5. 建立整體流程（前處理 + 預測模型）
model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])

# 6. 切分訓練資料和測試資料（80%訓練、20%測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 訓練模型
model.fit(X_train, y_train)

# 8. 預測房價
y_pred = model.predict(X_test)

# 9. 顯示模型準確度
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"平均誤差：{mae:.0f} 萬元")
print(f"R²（準確度）：{r2:.2%}")
