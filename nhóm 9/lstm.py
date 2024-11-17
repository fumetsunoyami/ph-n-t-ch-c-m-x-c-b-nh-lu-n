import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, mean_squared_error, confusion_matrix
import seaborn as sns

# Đọc và làm sạch dữ liệu từ từng file riêng biệt
df_train = pd.read_csv("demo3.csv", encoding="utf-8")
df_train.columns = df_train.columns.str.strip()

# Đọc và chuẩn bị từ điển từ khóa cho từng nhãn cảm xúc
df_keywords = pd.read_csv('keyword.csv')
keyword_dict = {
    'positive': df_keywords['Tích cực'].dropna().tolist(),
    'neutral': df_keywords['Trung tính'].dropna().tolist(),
    'negative': df_keywords['Tiêu Cực'].dropna().tolist(),
    'toxic': df_keywords['Toxic'].dropna().tolist()
}

# Hàm để gán nhãn cảm xúc
def assign_sentiment(comment):
    if pd.isna(comment):
        return "unknown"
    comment = comment.lower()
    for label, keywords in keyword_dict.items():
        if any(word in comment for word in keywords):
            return label
    return "unknown"

# Gán nhãn cho dữ liệu từ file comment.csv
df_train['label'] = df_train['comment'].apply(assign_sentiment)
df_train = df_train[df_train['label'] != 'unknown']  # Lọc bỏ các nhãn 'unknown'

# Lưu dữ liệu đã gán nhãn vào các file CSV riêng
df_train.to_csv("du_lieu_gan_nhan.csv", index=False, encoding="utf-8")

# Chuẩn bị dữ liệu cho mô hình LSTM
X_train = df_train['comment']
y_train = df_train['label']

# Chuyển đổi nhãn thành số
label_map = {'positive': 0, 'neutral': 1, 'negative': 2, 'toxic': 3}
y_train = y_train.map(label_map).values

# Tokenize and pad the sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_seq, maxlen=100)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train_padded, y_train, test_size=0.2, random_state=50)

# Build and train the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=100),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),  # Dropout để tránh overfitting
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.2),  # Dropout thêm lần nữa
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model and plot results
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Prediction and plot
y_pred = np.argmax(model.predict(X_test), axis=-1)
results_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_pred})

# Combine actual and predicted counts with fixed indices (0, 1, 2, 3)
label_indices = [0, 1, 2, 3]
actual_counts = results_df['Thực tế'].value_counts().reindex(label_indices, fill_value=0).sort_index()
predicted_counts = results_df['Dự đoán'].value_counts().reindex(label_indices, fill_value=0).sort_index()

# Create a DataFrame for combined plotting
comparison_df = pd.DataFrame({'Thực tế': actual_counts, 'Dự đoán': predicted_counts})

# Unified bar plot
plt.figure(figsize=(10, 6))
comparison_df.plot(kind='bar', color=['skyblue', 'salmon'], width=0.7)
plt.title("So sánh cảm xúc Thực tế và Dự đoán")
plt.xlabel("Cảm xúc")
plt.ylabel("Tần số")
plt.xticks(ticks=[0, 1, 2, 3], labels=['Tích cực', 'Trung lập', 'Tiêu cực', 'Toxic'], rotation=0)

# Adding labels above bars for both actual and predicted
for i in range(len(comparison_df)):
    plt.text(i - 0.15, actual_counts[i] + 0.5, str(actual_counts[i]), color="skyblue", ha="center")
    plt.text(i + 0.15, predicted_counts[i] + 0.5, str(predicted_counts[i]), color="salmon", ha="center")

plt.tight_layout()
plt.show()

# F1, Precision, Recall 
f1_macro = f1_score(y_test, y_pred, average='macro') 
f1_micro = f1_score(y_test, y_pred, average='micro') 
precision = precision_score(y_test, y_pred, average='macro') 
recall = recall_score(y_test, y_pred, average='macro') 
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse) 
# Create a DataFrame to hold the metrics 
metrics_df = pd.DataFrame({ 
    'Metric': ['F1 Macro', 'F1 Micro', 'Precision', 'Recall', 'MSE', 'RMSE'], 
    'Score': [f1_macro, f1_micro, precision, recall, mse, rmse] 
    })
print(metrics_df)
# Prediction and print confusion matrix 
y_pred = np.argmax(model.predict(X_test), axis=-1) 
results_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_pred}) 
# Print classification report 
print(classification_report(y_test, y_pred, target_names=label_map.keys())) 
# Generate confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred) 
print('Confusion Matrix:') 
print(conf_matrix)

# Biểu đồ heatmap cho confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title('Heatmap của Ma trận nhầm lẫn')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()

# Biểu đồ so sánh số lượng tham số và accuracy
'''num_words = [1000, 2000, 3000, 4000, 5000]
accuracies = []
for words in num_words:
    tokenizer = Tokenizer(num_words=words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_padded = pad_sequences(X_train_seq, maxlen=100)
    X_train, X_test, y_train, y_test = train_test_split(X_train_padded, y_train, test_size=0.2, random_state=50)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=words, output_dim=64, input_length=100),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
sns.lineplot(x=num_words, y=accuracies, marker='o')
plt.title('Số lượng Tham số so với Accuracy')
plt.xlabel('Số lượng Tham số')
plt.ylabel('Accuracy')
plt.show()'''


# Đọc và xử lý dữ liệu mới từ file CSV nếu tồn tại
new_file_path = 'du_lieu_moi.csv'
if os.path.exists(new_file_path):
    df_new = pd.read_csv(new_file_path, encoding="utf-8")
    df_new.columns = df_new.columns.str.strip()
    df_new['comment'] = df_new['comment'].fillna('')
    
    # Tokenize and pad the sequences for new data
    X_new_seq = tokenizer.texts_to_sequences(df_new['comment'])
    X_new_padded = pad_sequences(X_new_seq, maxlen=100)
    
    # Dự đoán nhãn cảm xúc cho dữ liệu mới
    y_new_pred = np.argmax(model.predict(X_new_padded), axis=-1)
    df_new['predicted_label'] = y_new_pred
    df_new['predicted_label'] = df_new['predicted_label'].map({0: 'positive', 1: 'neutral', 2: 'negative', 3: 'toxic'})
    
    # Lưu dữ liệu dự đoán vào file CSV
    df_new.to_csv('du_lieu_moi_du_doan_lstm.csv', index=False, encoding="utf-8")
    
    # Vẽ biểu đồ cho dữ liệu mới
    new_label_counts = df_new['predicted_label'].value_counts().reindex(['positive', 'neutral', 'negative', 'toxic'], fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=new_label_counts.index, y=new_label_counts.values, palette='viridis')
    plt.title("Số lượng nhãn cảm xúc trong dữ liệu mới")
    plt.xlabel("Nhãn cảm xúc")
    plt.ylabel("Số lượng")
    for i in range(len(new_label_counts)):
        plt.text(i, new_label_counts[i] + 0.5, str(new_label_counts[i]), ha='center')
    plt.tight_layout()
    plt.show()
else:
    print(f"File {new_file_path} không tồn tại. Không thể dự đoán nhãn cảm xúc cho dữ liệu mới.")
