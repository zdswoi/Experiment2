import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, cross_val_score
from imblearn.over_sampling import SMOTE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.ticker import MultipleLocator
import math

# 导入深度学习相关库
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 设置随机种子，保证结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 设置matplotlib后端
import matplotlib
matplotlib.use('TkAgg')

# 全局字体设置
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 5
plt.rcParams['legend.fontsize'] = 26

# 加载Excel数据
file_path = f"E:/研一上/5 抗生素检测/实验2/荧光/260126/6-拟合-选择性-抗干扰.xlsx"
sheet = pd.read_excel(file_path, sheet_name="6-虾-未知")

# 提取数据
data = sheet.iloc[:, 1:].values.T  # 样本为行，特征为列
new_data = sheet.iloc[:, 1:2].values.T  # 待预测的新数据

# 提取并处理标签
class_labels = sheet.columns[1:]
labels = [math.trunc(float(label)) for label in class_labels]
labels = np.array(labels)
# 标签归一化（深度学习需要从0开始的连续标签）
labels = labels - np.min(labels)  # 确保标签从0开始，避免索引错误
class_names = ["1","2","3"]
# class_names = ["FTD","NFZ","FZD","CTC", "TC", "DOX", "OTC","LEV","NOR","ENR"]
# class_names = ["FTD","NIT","FZD","CTC", "TC", "DOX", "OTC"]
# class_names = ["Blank","OTC","FTD","LEV", "His", "Cys", "KAN","SD-Na","SMZ-Na","Na+","K+","SO42-","HCO3-","Cl-"]
unique_labels = np.unique(labels)

# 处理新数据标签
class_labels_prediction = sheet.columns[1:2]
labels_prediction = [math.trunc(float(label)) for label in class_labels_prediction]
labels_prediction = np.array(labels_prediction) - np.min(labels)  # 同步归一化
prediction = np.unique(labels_prediction)
prediction = class_names[prediction[0]]

# 生成sample_labels（用于HCA树状图）
class_counts = {cls: 0 for cls in class_names}
label_to_name = {unique_labels[i]: class_names[i] for i in range(len(unique_labels))}
sample_true_names = [label_to_name[label] for label in labels]
sample_labels = []
for name in sample_true_names:
    class_counts[name] += 1
    sample_labels.append(f"{name}{class_counts[name]}")

# 数据归一化（LDA将使用此原始标准化数据）
scaler = StandardScaler()
data_norm = scaler.fit_transform(data)
new_data_norm = scaler.transform(new_data)

# PCA降维（仅用于PCA绘图、KMeans、HCA、GNB、深度学习模型）
pca = PCA(n_components=2)
score = pca.fit_transform(data_norm)
new_data_pca = pca.transform(new_data_norm)
latent = pca.explained_variance_ratio_ * 100

# 修复后的置信椭圆函数（解决无效值问题）
def confidence_ellipse(x, y, ax, n_std=3, edgecolor='k',facecolor='k', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    # 处理单样本情况（无法计算协方差）
    if len(x) <= 1:
        return ax
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # 防止sqrt出现负数（浮点精度问题）
    pearson = np.clip(pearson, -0.999999, 0.999999)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      edgecolor=edgecolor, facecolor=facecolor,** kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# 颜色映射
custom_colors = [
    (0.651,0.808,0.890), (0.122,0.471,0.706), (0.698,0.875,0.541),
    (0.2,0.627,0.173), (0.984, 0.604, 0.6), (0.890,0.102,0.109),
    (0.992,0.749,0.435), (1,0.498,0), (0.792,0.698,0.839),
    (0.416,0.239,0.604), (0.804,0.831,0.769), (0.694,0.349,0.157),
    (0.3373,0.4392,0.6039),(0.3843,0.7059,0.6471)
]
# custom_colors = [
#       (0.122,0.471,0.706), (0.698,0.875,0.541),
#       (0.2,0.627,0.173), (0.984, 0.604, 0.6), (0.890,0.102,0.109),
#       (0.992,0.749,0.435), (1,0.498,0), (0.792,0.698,0.839),
#       (0.416,0.239,0.604), (0.804,0.831,0.769), (0.694,0.349,0.157)
# ]
custom_cmap = plt.cm.colors.ListedColormap(custom_colors)
color_map = {label: custom_cmap(i) for i, label in enumerate(unique_labels)}

# 1. 绘制PCA散点图
plt.figure(figsize=(9, 7.2))
ax = plt.gca()
class_centers = {}
for i, class_id in enumerate(unique_labels):
    mask = (labels == class_id)
    x = score[mask, 0]
    y = score[mask, 1]
    class_centers[class_id] = (np.mean(x), np.mean(y))
    plt.scatter(x, y, color=color_map[class_id], s=150, alpha=1.0,
                label=f'{class_names[i]} (n={len(x)})', edgecolor='k')
    if len(x) > 1:
        confidence_ellipse(x, y, ax, n_std=3,
                           edgecolor=color_map[class_id],
                           facecolor=color_map[class_id],
                           linewidth=1, linestyle='-', alpha=0.65)

plt.title('PCA classification plot', fontsize=30, fontweight='bold')
plt.xlabel(f'PC1 ({latent[0]:.1f}%)', fontsize=30, fontweight='bold')
plt.ylabel(f'PC2 ({latent[1]:.1f}%)', fontsize=30, fontweight='bold')
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=28, width=5, length=8, pad=8)
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
ax.tick_params(axis='both', which='minor', width=5, length=5)
plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# LDA降维（输入为原始标准化数据data_norm）
lda = LinearDiscriminantAnalysis(n_components=2)
lda_score = lda.fit_transform(data_norm, labels)

# 绘制LDA散点图
plt.figure(figsize=(9, 7.2))
ax = plt.gca()
lda_class_centers = {}
for i, class_id in enumerate(unique_labels):
    mask = (labels == class_id)
    x = lda_score[mask, 0]
    y = lda_score[mask, 1]
    lda_class_centers[class_id] = (np.mean(x), np.mean(y))
    plt.scatter(x, y, color=color_map[class_id], s=150, alpha=1.0,
                label=f'{class_names[i]} (n={len(x)})', edgecolor='k')
    if len(x) > 1:
        confidence_ellipse(x, y, ax, n_std=4,
                           edgecolor=color_map[class_id],
                           facecolor=color_map[class_id],
                           linewidth=1, linestyle='-', alpha=0.65)

# 新数据的LDA降维
new_data_lda = lda.transform(new_data_norm)

# LDA图表美化
plt.title('LDA classification plot', fontsize=30, fontweight='bold')
plt.xlabel(f'LD1 ({lda.explained_variance_ratio_[0]*100:.1f}%)', fontsize=30, fontweight='bold')
plt.ylabel(f'LD2 ({lda.explained_variance_ratio_[1]*100:.1f}%)', fontsize=30, fontweight='bold')
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=28, width=5, length=8, pad=8)
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_minor_locator(MultipleLocator(25))
ax.tick_params(axis='both', which='minor', width=5, length=5)
plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# LDA预测新数据类别
lda_pred = lda.predict(new_data_norm)
lda_pred_mapped = [class_names[int(p)] for p in lda_pred]
print(f"\nLDA Prediction for new data: {lda_pred_mapped}")

# 2. K-Means聚类（修复索引越界问题）
n_clusters = len(unique_labels)  # 动态匹配类别数，避免硬编码
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(score)
centers = kmeans.cluster_centers_
dist_matrix = cdist(centers, list(class_centers.values()))
cluster_to_class = np.argmin(dist_matrix, axis=1)
kmeans_labels_mapped = np.array([cluster_to_class[label] for label in kmeans.labels_])

plt.figure(figsize=(9, 7.2))
ax = plt.gca()
added_labels = set()
# 修复：循环范围改为实际聚类数（n_clusters）
for i in range(n_clusters):
    mask = (kmeans.labels_ == i)
    if not np.any(mask):  # 跳过空聚类
        continue
    x = score[mask, 0]
    y = score[mask, 1]
    true_class_idx = cluster_to_class[i]
    class_name = class_names[true_class_idx]
    label = class_name if class_name not in added_labels else ''
    plt.scatter(x, y, color=color_map[unique_labels[true_class_idx]],
                s=150, alpha=1.0, edgecolor='k', label=label)
    if class_name not in added_labels:
        added_labels.add(class_name)
    if len(x) > 1:
        confidence_ellipse(x, y, ax, n_std=3.0,
                           edgecolor=color_map[unique_labels[true_class_idx]],
                           facecolor=color_map[unique_labels[true_class_idx]],
                           linestyle='-', linewidth=1, alpha=0.65)

kmeans_pred = kmeans.predict(new_data_pca)
kmeans_pred_mapped = [class_names[cluster_to_class[p]] for p in kmeans_pred]
plt.xlabel(f'PC1 ({latent[0]:.1f}%)', fontsize=30, fontweight='bold')
plt.ylabel(f'PC2 ({latent[1]:.1f}%)', fontsize=30, fontweight='bold')
plt.title('K-Means classification plot', fontsize=30, fontweight='bold', pad=10)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=28, width=5, length=8, pad=8)
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
ax.tick_params(axis='both', which='minor', width=5, length=5)
plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 4. 分层聚类HCA
linked = linkage(score, 'average')
num_clusters = 4
cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')
hca_centers = []
for i in range(1, num_clusters + 1):
    mask = (cluster_labels == i)
    hca_centers.append((np.mean(score[mask, 0]), np.mean(score[mask, 1])))
dist_matrix_hca = cdist(hca_centers, list(class_centers.values()))
hca_to_class = np.argmin(dist_matrix_hca, axis=1)
hca_labels_mapped = np.array([hca_to_class[label - 1] for label in cluster_labels])

# 新数据HCA预测
combined_data = np.vstack([score, new_data_pca])
linked_combined = linkage(combined_data, 'average')
cluster_labels_combined = fcluster(linked_combined, num_clusters, criterion='maxclust')
hca_pred = cluster_labels_combined[-len(new_data_pca):]
hca_pred_mapped = [class_names[hca_to_class[p - 1]] for p in hca_pred]

# 绘制HCA树状图
plt.figure(figsize=(12, 6))
simple_sample_labels = [label_to_name[label] for label in labels]
dendrogram(linked, orientation='top', distance_sort='descending',
           show_leaf_counts=True, color_threshold=0.2, labels=simple_sample_labels)
ax = plt.gca()
for collection in ax.collections:
    collection.set_linewidth(4)
current_ylim = ax.get_ylim()
y_expand = 0.05
ax.set_ylim(current_ylim[0]-y_expand, current_ylim[1])
plt.axhline(y=0.2, color='r', linestyle='--', linewidth=4)
plt.ylabel('Distance', fontsize=30, fontweight='bold')
plt.title('HCA Dendrogram', fontsize=30, fontweight='bold',pad=10)
ax.minorticks_on()
ax.tick_params(axis='y', which='major', labelsize=28, width=5, length=8, pad=8)
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.tick_params(axis='y', which='minor', width=5, length=5)
ax.xaxis.set_minor_locator(plt.NullLocator())
plt.xticks(rotation=90, fontsize=20, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 划分数据集（用于GNB和深度学习模型训练）
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, test_idx in sss.split(score, labels):
    X_train, X_test = score[train_idx], score[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

# SMOTE处理数据不平衡
unique, counts = np.unique(y_train, return_counts=True)
min_samples = np.min(counts)
n_neighbors = min(5, min_samples - 1)
if min_samples > 1:
    smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE processed training set samples: {X_train_resampled.shape[0]} (original: {X_train.shape[0]})")
else:
    X_train_resampled, y_train_resampled = X_train, y_train
    print(f"Insufficient sample size, SMOTE not used (minimum class samples: {min_samples})")

# 动态调整交叉验证折数
unique_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
min_samples_resampled = np.min(counts_resampled)
cv_folds = min(5, min_samples_resampled)
print(f"Cross-validation folds adjusted based on sample size: {cv_folds}")

# 高斯朴素贝叶斯分类
gnb_param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
gnb_grid = GridSearchCV(GaussianNB(), gnb_param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
gnb_grid.fit(X_train_resampled, y_train_resampled)
print(f"GNB Optimal parameters: {gnb_grid.best_params_}")
print(f"GNB Cross-validation accuracy: {gnb_grid.best_score_:.4f}")
gnb = gnb_grid.best_estimator_

# GNB预测新数据和全部数据
gnb_pred_new = gnb.predict(new_data_pca)
gnb_pred_mapped = [class_names[int(p)] for p in gnb_pred_new]
gnb_pred_all = gnb.predict(score)
gnb_acc_all = accuracy_score(labels, gnb_pred_all)
print(f"GNB Full dataset accuracy: {gnb_acc_all:.2f}")

# 深度学习模型（MLP）
# 1. 准备深度学习模型的输入数据（标签转为独热编码）
y_train_onehot = to_categorical(y_train_resampled, num_classes=len(unique_labels))
y_test_onehot = to_categorical(y_test, num_classes=len(unique_labels))

# 2. 构建简单的全连接神经网络模型
def build_mlp_model(input_dim, num_classes):
    model = Sequential([
        # 输入层
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),  # 防止过拟合
        # 隐藏层
        Dense(32, activation='relu'),
        Dropout(0.2),
        # 输出层（多分类用softmax）
        Dense(num_classes, activation='softmax')
    ])
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. 初始化并训练模型
mlp_model = build_mlp_model(input_dim=X_train_resampled.shape[1], num_classes=len(unique_labels))
# 训练模型（小批量、早停防止过拟合）
history = mlp_model.fit(
    X_train_resampled, y_train_onehot,
    validation_data=(X_test, y_test_onehot),
    batch_size=8,
    epochs=50,
    verbose=1,  # 1=显示训练进度，0=静默
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# 4. 深度学习模型预测
# 预测全部数据
mlp_pred_all_prob = mlp_model.predict(score, verbose=0)
mlp_pred_all = np.argmax(mlp_pred_all_prob, axis=1)  # 从概率转为类别标签
mlp_acc_all = accuracy_score(labels, mlp_pred_all)
print(f"MLP Full dataset accuracy: {mlp_acc_all:.2f}")

# 预测新数据
mlp_pred_new_prob = mlp_model.predict(new_data_pca, verbose=0)
mlp_pred_new = np.argmax(mlp_pred_new_prob, axis=1)
mlp_pred_mapped = [class_names[int(p)] for p in mlp_pred_new]

# 绘制混淆矩阵（GNB + 深度学习）
plt.figure(figsize=(12, 5))

# 1. GNB混淆矩阵（全部数据）
plt.subplot(121)
gnb_cm = confusion_matrix(labels, gnb_pred_all)
gnb_disp = ConfusionMatrixDisplay(confusion_matrix=gnb_cm, display_labels=class_names)
gnb_disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
plt.title(f'GNB Confusion Matrix', fontsize=28, fontweight='bold')
plt.xlabel('Prediction', fontsize=26, fontweight='bold')
plt.ylabel('Target', fontsize=26, fontweight='bold')
plt.gca().tick_params(axis='both', labelsize=28, width=4, length=0, pad=9)
plt.gca().set_xticklabels(class_names, rotation=90, ha='center', fontsize=22, fontweight='bold')
plt.gca().set_yticklabels(class_names, fontsize=22, fontweight='bold')
for text in gnb_disp.text_.flatten():
    text.set_fontsize(22)
    text.set_weight('bold')

# 2. 深度学习（MLP）混淆矩阵（全部数据）
plt.subplot(122)
mlp_cm = confusion_matrix(labels, mlp_pred_all)
mlp_disp = ConfusionMatrixDisplay(confusion_matrix=mlp_cm, display_labels=class_names)
mlp_disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
plt.title(f'MLP Confusion Matrix', fontsize=28, fontweight='bold')
plt.xlabel('Prediction', fontsize=26, fontweight='bold')
plt.ylabel('Target', fontsize=26, fontweight='bold')
plt.gca().tick_params(axis='both', labelsize=28, width=4, length=0, pad=9)
plt.gca().set_xticklabels(class_names, rotation=90, ha='center', fontsize=22, fontweight='bold')
plt.gca().set_yticklabels(class_names, fontsize=22, fontweight='bold')
for text in mlp_disp.text_.flatten():
    text.set_fontsize(22)
    text.set_weight('bold')

plt.tight_layout()

# 最终预测结果输出
print("\n=== Final Prediction Results ===")
print(f"Actual Category: ['{prediction}']")
print(f"K-Means Prediction: {kmeans_pred_mapped}")
print(f"HCA Prediction: {hca_pred_mapped}")
print(f"Gaussian Naive Bayes Prediction: {gnb_pred_mapped}")
print(f"Deep Learning (MLP) Prediction: {mlp_pred_mapped}")
print(f"LDA Prediction (original normalized data): {lda_pred_mapped}")

# ====================== 新增：导出 PC1 和 LD1 到 Excel ======================
# import pandas as pd
#
# # 1. 整理 PCA 降维结果（只取 PC1）
# pca_result_df = pd.DataFrame({
#     "Sample": [f"Sample{i+1}" for i in range(len(score))],
#     "True_Label": labels,
#     "Class_Name": sample_true_names,
#     "PC1": score[:, 0]  # 只保留第一主成分 PC1
# })
#
# # 2. 整理 LDA 降维结果（只取 LD1）
# lda_result_df = pd.DataFrame({
#     "Sample": [f"Sample{i+1}" for i in range(len(lda_score))],
#     "True_Label": labels,
#     "Class_Name": sample_true_names,
#     "LD1": lda_score[:, 0]  # 只保留第一线性判别分量 LD1
# })

# # 3. 合并到同一个 Excel 的不同 sheet
# with pd.ExcelWriter(f"E:/研一上/5 抗生素检测/实验2/荧光/260126/PCA_LD1_LEV.xlsx") as writer:
#     pca_result_df.to_excel(writer, sheet_name="PCA_PC1_Results", index=False)
#     lda_result_df.to_excel(writer, sheet_name="LDA_LD1_Results", index=False)
#
# print("✅ PC1 和 LD1 已成功保存到 Excel 文件：PCA_LD1_Dimension_Reduction_Results.xlsx")
# ==========================================================================

# 显示所有图表
plt.show()