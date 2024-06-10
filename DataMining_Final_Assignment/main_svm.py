import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
from imblearn.over_sampling import RandomOverSampler


# Load data
data = pd.read_csv('./mbti_1.csv')

# 数据增强，添加一个特征
def calculate_average_length(data):
    return len(data.split(' ')) / len(data.split('|||'))

data['average_length'] = data['posts'].apply(calculate_average_length)

# 清洗数据，去掉posts中的网址并用单词“link”代替之，去掉奇怪的字符仅保留单词
def preprocess_text(text):
    # Remove links
    temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', text)
    # Remove non-alphabetic characters
    temp = re.sub("[^a-zA-Z]", " ", temp)
    # Convert to lowercase
    temp = temp.lower()
    # Remove space >1
    temp = re.sub(' +', ' ', temp)
    return temp

data['posts'] = data['posts'].apply(preprocess_text)

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 将MBTI类型转换为整数标签
label_map = {'INTJ': 0, 'INTP': 1, 'INFJ': 2, 'INFP': 3, 'ISTJ': 4, 'ISTP': 5, 'ISFJ': 6, 'ISFP': 7,
             'ENTJ': 8, 'ENTP': 9, 'ENFJ': 10, 'ENFP': 11, 'ESTJ': 12, 'ESTP': 13, 'ESFJ': 14, 'ESFP': 15}
train_labels = train_data['type'].map(label_map)
test_labels = test_data['type'].map(label_map)

# 把较少的MBTI种类上采样
ros = RandomOverSampler(random_state=42)
train_data, train_labels = ros.fit_resample(train_data, train_labels)

# 文本向量化
vectorizer = CountVectorizer()
X_train_text = vectorizer.fit_transform(train_data['posts'])
X_test_text = vectorizer.transform(test_data['posts'])

# 获取平均长度特征
X_train_avg_length = train_data['average_length'].values.reshape(-1, 1)
X_test_avg_length = test_data['average_length'].values.reshape(-1, 1)

# 合并文本特征和平均长度特征
X_train = hstack((X_train_text, X_train_avg_length))
X_test = hstack((X_test_text, X_test_avg_length))


# 使用SVM进行分类
# 定义SVM模型
model = svm.SVC(kernel='linear', C=0.8, random_state=1)

# 训练模型
model.fit(X_train, train_labels)

# 预测测试集
predictions = model.predict(X_test)

# 计算准确率
test_accuracy = accuracy_score(test_labels, predictions)
print(f'Test Accuracy: {test_accuracy:.4f}')