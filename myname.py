# %% [code] {"execution":{"iopub.status.busy":"2024-08-17T20:15:22.918146Z","iopub.execute_input":"2024-08-17T20:15:22.918960Z","iopub.status.idle":"2024-08-17T20:15:22.923751Z","shell.execute_reply.started":"2024-08-17T20:15:22.918922Z","shell.execute_reply":"2024-08-17T20:15:22.922279Z"}}
def mlmodel():
    code = '''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Assuming x_train, y_train, x_test, and y_test are predefined
model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = r2_score(y_test, y_predict)
    '''
    print(code)
    
def traintest():
    code = '''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    '''
    print(code)
def onehotencode():
    code = '''
from sklearn.preprocessing import OneHotEncoder
categorical_cols = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse=False, drop='first') 
encoded_cols = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
print(df_encoded)
    '''
    print(code)
def logistic():
    code = '''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    '''
    print(code)
def decisiontree():
    code = '''
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    '''
    print(code)
def randomforest():
    code = '''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    '''
    print(code)
def gradientboosting():
    code = '''
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    '''
    print(code)
def svm():
    code = '''
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    '''
    print(code)
def naivebayes():
    code = '''
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    '''
    print(code)
def xgboost():
    code = '''
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    '''
    print(code)
def plot():
    code = '''
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Line Plot")
plt.show()

plt.scatter([1, 2, 3], [4, 5, 6])

plt.hist([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

sns.boxplot(data=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
sns.heatmap(data)

iris = sns.load_dataset('iris')
sns.pairplot(iris)

# Covariance Plot
np.random.seed(0)
data = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100)
})
cov_matrix = data.cov()
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm')
plt.title("Covariance Matrix Heatmap")
plt.show()
    '''
    print(code)
def kmeans():
    code = '''
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
kmeans = KMeans(n_clusters=3, random_state=0)
data['Cluster'] = kmeans.fit_predict(data_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Feature1', y='Feature2', hue='Cluster', palette='Set1', style='Cluster', s=100)
plt.title("K-Means Clustering (Feature1 vs Feature2)")
plt.show()
    '''
    print(code)
def imbalanced():
    code = '''
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# Apply SMOTE for over-sampling
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Apply RandomUnderSampler for under-sampling
under_sampler = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = under_sampler.fit_resample(X, y)
    '''
    print(code)
def dataset():
    code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Custom dataset class (using normal method)
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# Example data
x_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
y_train = [0, 1, 0]

# Create dataset and dataloader using CustomDataset
train_dataset = CustomDataset(torch.tensor(x_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Example using TensorDataset and DataLoader
features = torch.tensor(x_train, dtype=torch.float32)
labels = torch.tensor(y_train, dtype=torch.long)
dataset = TensorDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("DataLoader created with CustomDataset and TensorDataset.")
    '''
    print(code)
def neuralnetwork():
    code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # Linear layer (fully connected)
        self.fc1 = nn.Linear(in_features=784, out_features=128)  # Example for MNIST-like data
        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # Recurrent layers
        self.rnn = nn.RNN(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
        self.gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Output layer
        self.fc2 = nn.Linear(in_features=64, out_features=10)  # Example for 10 classes

    def forward(self, x):
        # Applying convolutional layer
        x = self.conv1(x)
        x = F.relu(x)  # Applying ReLU activation
        x = self.pool(x)  # Applying MaxPooling
        x = self.flatten(x)  # Flattening the output for fully connected layers
        x = self.fc1(x)  # Fully connected layer
        x = F.relu(x)
        x = self.dropout(x)  # Applying Dropout
        
        # Choose one of the following RNN, LSTM, GRU layers based on your requirement
        x, _ = self.rnn(x.unsqueeze(1))  # Applying RNN
        x, _ = self.lstm(x)  # Applying LSTM
        x, _ = self.gru(x)  # Applying GRU

        x = self.fc2(x[:, -1, :])  # Output layer
        x = F.softmax(x, dim=1)  # Applying Softmax for multi-class classification
        return x

# Model, Loss Function, and Optimizer
model = NeuralNet()

# Different optimizers (Uncomment the one you want to use)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)
# optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Different loss functions (Uncomment the one you want to use)
# criterion = nn.CrossEntropyLoss()  # Standard for multi-class classification
# criterion = nn.BCEWithLogitsLoss()  # For binary classification or multi-label classification
# criterion = nn.MSELoss()  # For regression tasks
# criterion = nn.NLLLoss()  # Negative Log Likelihood Loss (used with LogSoftmax)
    '''
    print(code)
def train():
    code = '''
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in DataLoader(CustomDataset(torch.tensor(x_test, dtype=torch.float32),
                                                   torch.tensor(y_test, dtype=torch.long)), batch_size=32):
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
    '''
    print(code)
def nlp():
    code = '''
pip install torch nltk emoji
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import emoji
nltk.download('punkt')
nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = emoji.replace_emoji(text, replace='')
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [lemmatizer.lemmatize(stemmer.stem(word.lower())) for word in words]
        cleaned_sentences.append(" ".join(words))
    return " ".join(cleaned_sentences)
cleaned_data = [(clean_text(text), label) for text, label in data]
vocab = set()
for text, _ in cleaned_data:
    words = word_tokenize(text)
    vocab.update(words)
vocab = sorted(vocab)
word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
word2idx['<PAD>'] = 0  # Padding token
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    max_len = max(lengths)
    padded_texts = [F.pad(text, (0, max_len - len(text)), value=0) for text in texts]
    return torch.stack(padded_texts), torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        output = self.fc(hn[-1])
        return output
vocab_size = len(word2idx)
embedding_dim = 100
hidden_dim = 64
output_dim = 2
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
    '''
    print(code)








# %% [code] {"execution":{"iopub.status.busy":"2024-08-17T20:15:36.634495Z","iopub.execute_input":"2024-08-17T20:15:36.634891Z","iopub.status.idle":"2024-08-17T20:15:36.642732Z","shell.execute_reply.started":"2024-08-17T20:15:36.634861Z","shell.execute_reply":"2024-08-17T20:15:36.641415Z"}}


# %% [code]
