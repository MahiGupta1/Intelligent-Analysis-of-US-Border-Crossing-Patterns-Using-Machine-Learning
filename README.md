# Intelligent-Analysis-of-US-Border-Crossing-Patterns-Using-Machine-Learning
This project analyzes US border crossing data using machine learning to classify border types, identify patterns, and detect anomalies. Models like KNN, Random Forest, K-Means, and Isolation Forest are compared using accuracy and visual analysis.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------- Linear Regression ----------
X_lr = yearly[["year"]]
y_lr = yearly["total_crossings"]

lr = LinearRegression()
lr.fit(X_lr, y_lr)
y_lr_pred = lr.predict(X_lr)

lr_r2 = r2_score(y_lr, y_lr_pred)

# ---------- Logistic Regression ----------
df["traffic_level"] = (df["value"] > df["value"].median()).astype(int)

X_log = df[["year"]]
y_log = df["traffic_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

log = LogisticRegression()
log.fit(X_train, y_train)
y_log_pred = log.predict(X_test)

log_acc = accuracy_score(y_test, y_log_pred)

# ---------- K-Means Clustering ----------
cluster_data = yearly[["year", "total_crossings"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(scaled_data)

# ---------- PCA ----------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# ---------- Model Comparison ----------
comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Logistic Regression",
        "K-Means Clustering",
        "PCA"
    ],
    "Evaluation Metric": [
        "R2 Score",
        "Accuracy",
        "Clusters Formed",
        "Dimensions Reduced"
    ],
    "Result": [
        round(lr_r2, 4),
        round(log_acc, 4),
        len(set(clusters)),
        pca_data.shape[1]
    ]
})

print(comparison)





import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score                                                             df.columns = df.columns.str.lower().str.replace(" ", "_")

value_col = df.select_dtypes(include="number").columns[0]
date_col = [c for c in df.columns if "date" in c][0]

df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
df[date_col] = pd.to_datetime(df[date_col], format="mixed")

df = df.dropna(subset=[value_col, date_col, "state", "measure"])

df = df.loc[:, :].copy()

df.loc[:, "value"] = df[value_col].values
df.loc[:, "year"] = df[date_col].dt.year.values#Analyze yearly border crossing trends
yearly = (
    df.groupby("year", as_index=False)
      .agg(total_crossings=("value", "sum"))
)

plt.figure()
plt.plot(yearly["year"], yearly["total_crossings"], marker="o")
plt.title("Yearly Border Crossing Trend")
plt.xlabel("Year")
plt.ylabel("Total Crossings")
plt.show()#Compare border crossings across states
state_data = (
    df.groupby("state", as_index=False)
      .agg(total=("value", "sum"))
      .sort_values(by="total", ascending=False)
)

plt.figure()
plt.bar(state_data["state"], state_data["total"])
plt.xticks(rotation=90)
plt.title("State-wise Border Traffic")
plt.show()#Compare border crossings by transport type (measure)
measure_data = (
    df.groupby("measure", as_index=False)
      .agg(total=("value", "sum"))
)

plt.figure()
plt.bar(measure_data["measure"], measure_data["total"])
plt.xticks(rotation=45)
plt.title("Transport Type Comparison")
plt.show()#Predict traffic trends using Linear Regression
X = yearly[["year"]]
y = yearly["total_crossings"]

lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

plt.figure()
plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, label="Predicted")
plt.legend()
plt.title("Linear Regression Trend Prediction")
plt.show()

print("R2 Score:", r2_score(y, y_pred)) #Classify traffic levels using Logistic Regression
df["traffic_level"] = (df["value"] > df["value"].median()).astype(int)

Xc = df[["year"]]
yc = df["traffic_level"]

clf = LogisticRegression()
clf.fit(Xc, yc)
pred = clf.predict(Xc)

print("Classification Accuracy:", accuracy_score(yc, pred))

plt.figure()
df["traffic_level"].value_counts().plot(kind="bar")
plt.title("High vs Low Traffic")
plt.show()  # Evaluate models using R² score and accuracy
cluster_data = yearly[["year", "total_crossings"]]

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=0)
yearly["cluster"] = kmeans.fit_predict(scaled)

plt.figure()
plt.scatter(yearly["year"], yearly["total_crossings"], c=yearly["cluster"])
plt.title("K-Means Traffic Clusters")
plt.show()  #Group similar traffic patterns using K-Means clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cluster_data = yearly[["year", "total_crossings"]]

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=0)
yearly = yearly.copy()
yearly["cluster"] = kmeans.fit_predict(scaled)


plt.figure()
plt.scatter(
    yearly["year"],
    yearly["total_crossings"],
    c=yearly["cluster"]
)
plt.xlabel("Year")
plt.ylabel("Total Crossings")
plt.title("K-Means Clustering of Border Crossing Trends")
plt.show()  #Reduce dimensions and visualize patterns using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled)

plt.figure()
plt.scatter(pca_data[:,0], pca_data[:,1], c=yearly["cluster"])
plt.title("PCA Visualization")
plt.show()   
