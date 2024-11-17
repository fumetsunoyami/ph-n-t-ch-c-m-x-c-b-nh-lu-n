import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv('demo3.csv', encoding="utf-8")
data.columns = data.columns.str.strip()

keyword_df = pd.read_csv('keyword.csv', encoding="utf-8")
keyword_df.columns = keyword_df.columns.str.strip()

keywords = {
    'positive': set(keyword_df['Tích cực'].dropna().values),
    'neutral': set(keyword_df['Trung tính'].dropna().values),
    'negative': set(keyword_df['Tiêu Cực'].dropna().values),
    'toxic': set(keyword_df['Toxic'].dropna().values)
}
data['comment'] = data['comment'].fillna('')

def calculate_scores(comment, keywords):
    scores = {label: 0 for label in keywords.keys()}
    for label, words in keywords.items():
        for word in words:
            if word in comment:
                scores[label] += 1
    return scores

def label_comment(comment, keywords, priority_order):
    scores = calculate_scores(comment, keywords)
    max_score = max(scores.values())
    max_labels = [label for label, score in scores.items() if score == max_score]

    if max_score == 0 or len(max_labels) > 1:
        return 'unknown'

    for label in priority_order:
        if label in max_labels:
            return label
    return 'unknown'

priority_order = ['toxic', 'negative', 'neutral', 'positive']

if 'label' not in data.columns or data['label'].isnull().any():
    data['label'] = data['comment'].apply(lambda x: label_comment(x, keywords, priority_order))
filtered_comments = data[data['label'] != 'unknown']
filtered_comments.to_csv('du_lieu_gan_nhan.csv', index=False)

# Tiền xử lý và chuyển đổi văn bản thành dạng số
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(filtered_comments['comment'])
y = filtered_comments['label']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

# Huấn luyện mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Đánh giá mô hình
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Hiển thị biểu đồ cột các nhãn cảm xúc
label_counts = pd.DataFrame(y_test.value_counts()).reset_index()
label_counts.columns = ['label', 'count']

plt.figure(figsize=(10,6))
sns.barplot(x='label', y='count', data=label_counts, palette='viridis')
plt.title('Số lượng nhãn cảm xúc trong tập kiểm tra')
plt.xlabel('Nhãn cảm xúc')
plt.ylabel('Số lượng')

for i in range(len(label_counts)):
    plt.text(i, label_counts.iloc[i]['count'] + 0.5, str(label_counts.iloc[i]['count']), ha='center')
plt.show()

# Biểu đồ heatmap cho confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Heatmap của Ma trận nhầm lẫn')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()

# Biểu đồ so sánh số lượng tham số và accuracy
num_features = [1000, 2000, 3000, 4000, 5000]
accuracies = []
for features in num_features:
    vectorizer = TfidfVectorizer(max_features=features)
    X = vectorizer.fit_transform(filtered_comments['comment'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

plt.figure(figsize=(10, 6))
sns.lineplot(x=num_features, y=accuracies, marker='o')
plt.title('Số lượng Tham số so với Accuracy')
plt.xlabel('Số lượng Tham số')
plt.ylabel('Accuracy')
plt.show()

# Kiểm tra xem file CSV của dữ liệu mới có tồn tại không
new_file_path = 'du_lieu_moi.csv'
if os.path.exists(new_file_path):
    # Đọc dữ liệu bình luận mới từ file CSV
    new_data = pd.read_csv(new_file_path, encoding="utf-8")
    new_data.columns = new_data.columns.str.strip()

    # Tiền xử lý dữ liệu mới
    new_data['comment'] = new_data['comment'].fillna('')
    new_comments_tfidf = vectorizer.transform(new_data['comment'])

    # Dự đoán nhãn cảm xúc cho dữ liệu mới
    new_data['predicted_label'] = model.predict(new_comments_tfidf)

    # Xuất file CSV với nhãn dự đoán
    new_data.to_csv('du_lieu_moi_du_doan_lr.csv', index=False)

    # Hiển thị biểu đồ cột các nhãn cảm xúc cho dữ liệu mới
    label_counts_new = pd.DataFrame(new_data['predicted_label'].value_counts()).reset_index()
    label_counts_new.columns = ['label', 'count']
    plt.figure(figsize=(10,6))
    sns.barplot(x='label', y='count', data=label_counts_new, palette='viridis')
    plt.title('Số lượng nhãn cảm xúc trong dữ liệu mới')
    plt.xlabel('Nhãn cảm xúc')
    plt.ylabel('Số lượng')
    for i in range(len(label_counts_new)):
        plt.text(i, label_counts_new.iloc[i]['count'] + 0.5, str(label_counts_new.iloc[i]['count']), ha='center')
    plt.show()
else:
    print(f"File {new_file_path} không tồn tại. Không thể dự đoán nhãn cảm xúc cho dữ liệu mới.")
