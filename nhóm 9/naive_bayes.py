import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Đọc dữ liệu từ file CSV
data = pd.read_csv("demo3.csv", encoding="utf-8")
data.columns = data.columns.str.strip()

# Đọc từ khóa từ file keyword.csv và chuẩn bị danh sách từ khóa cho từng nhãn
keyword_df = pd.read_csv("keyword.csv", encoding="utf-8")
keyword_df.columns = keyword_df.columns.str.strip()

# Kiểm tra cột trong keyword_df để đảm bảo các cột từ khóa tồn tại
required_columns = ["Tích cực", "Trung tính", "Tiêu Cực", "Toxic"]
for col in required_columns:
    if col not in keyword_df.columns:
        raise ValueError(f"Thiếu cột '{col}' trong file keyword.csv.")

keywords = {
    "positive": set(keyword_df["Tích cực"].dropna().values),
    "neutral": set(keyword_df["Trung tính"].dropna().values),
    "negative": set(keyword_df["Tiêu Cực"].dropna().values),
    "toxic": set(keyword_df["Toxic"].dropna().values),
}

data["comment"] = data["comment"].fillna("")

def calculate_scores(comment, keywords):
    scores = {label: 0 for label in keywords.keys()}
    for label, words in keywords.items():
        for word in words:
            scores[label] += comment.count(word)
    return scores

def label_comment(comment, keywords, priority_order):
    scores = calculate_scores(comment, keywords)
    if all(score == 0 for score in scores.values()):
        return "unknown"

    max_score = max(scores.values())
    max_labels = [label for label, score in scores.items() if score == max_score]

    if len(max_labels) > 1:
        return "unknown"

    for label in priority_order:
        if label in max_labels:
            return label
    return "unknown"

priority_order = ["positive", "neutral", "negative", "toxic"]
if 'label' not in data.columns or data['label'].isnull().any(): 
    data['label'] = data['comment'].apply(lambda x: label_comment(x, keywords, priority_order))
filtered_comments = data[data['label'] != 'unknown']

# Chuẩn bị dữ liệu cho mô hình Naive Bayes
X = filtered_comments["comment"]
y = filtered_comments["label"]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Khởi tạo và huấn luyện mô hình Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Lưu dữ liệu đã gán nhãn vào file CSV
output_path = "output/ListCommentFacebook_labeled_nb.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
filtered_comments[["comment", "label"]].to_csv(output_path, index=False)

# Vẽ biểu đồ tần suất nhãn cảm xúc cho dữ liệu test
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
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
plt.title('Heatmap của Ma trận nhầm lẫn')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()

# Biểu đồ so sánh số lượng tham số và accuracy
num_features = [1000, 2000, 3000, 4000, 5000]
accuracies = []
for features in num_features:
    vectorizer = CountVectorizer(max_features=features)
    X = vectorizer.fit_transform(filtered_comments['comment'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
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
    new_data['predicted_label'] = nb_model.predict(new_comments_tfidf)

    # Xuất file CSV với nhãn dự đoán
    new_data.to_csv('du_lieu_moi_du_doan_nb.csv', index=False)

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
