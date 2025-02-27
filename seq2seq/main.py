import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import random
import torch.nn.functional as F
import glob
import unicodedata
from torch.utils.data import default_collate
from collections import defaultdict

""" 
0 colk z a r a z a r a, 1 denim g o ts u g o ts u, 2 gaze sh u w a sh u w a, 3 kanaami ch i k u ch i k u, 4 koutaku s u b e s u b e 
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 学習に使用するデバイスを設定

CHECKPOINT_DIR = "research/.results/result0121/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SK_ENCODER_LATEST = os.path.join(CHECKPOINT_DIR, "skEncoder_params_latest.pth")
SK_DECODER_LATEST = os.path.join(CHECKPOINT_DIR, "skDecoder_params_latest.pth")
ONO_ENCODER_LATEST = os.path.join(CHECKPOINT_DIR, "onoEncoder_params_latest.pth")
ONO_DECODER_LATEST = os.path.join(CHECKPOINT_DIR, "onoDecoder_params_latest.pth")

# 触覚モデルのパラメータ
hidden_size = 256  # GRUの隠れ層サイズ
batch_size = 10  # バッチサイズ
sk_num_epochs = 500  # 学習エポック数
learning_rate = 1e-4  # 学習率
sequence_length = 1000  # シーケンスの長さ
input_size = 6  # データの列数（ラベルを除く）
output_size = input_size
num_layers = 6

# オノマトペモデルのパラメータ
SOS_token = 0
EOS_token = 1
embedding_size = 256
max_length = 60
n_iters = 500

class Lang:
    def __init__(self, filename):
        self.filename = filename
        self.word2index = {}
        self.word2count = {}
        self.sentences = []  # 単語のリスト
        self.labels = []  # ラベルのリスト
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # SOSとEOSのカウント

        # txtファイルをデータフレームとして読み込んで保存
        self.data = pd.read_csv(self.filename)

        # データフレームから単語とラベルをリストとして保存
        self.sentences = self.data['word'].tolist()
        self.labels = self.data['label'].astype(int).tolist()

        assert len(self.sentences) == len(self.labels), "文とラベルの数が異なります。"

        # 初期化されたallow_list
        self.allow_list = [True] * len(self.sentences)
        self.label2word = {lbl: sent for sent, lbl in zip(self.sentences, self.labels)}

    def get_sentences(self):
        return self.sentences[::]

    def get_sentence(self, index):
        return self.sentences[index], self.labels[index]

    def choice(self):
        while True:
            index = random.randint(0, len(self.allow_list) - 1)
            if self.allow_list[index]:
                break
        # データフレームから動的に単語とラベルを取得
        sentence = self.data.loc[index, 'word']
        label = self.data.loc[index, 'label']
        return sentence, label, index

    def get_allow_list(self, max_length):
        allow_list = []
        for sentence in self.sentences:
            if len(sentence.split()) < max_length:
                allow_list.append(True)
            else:
                allow_list.append(False)
        return allow_list

    def load_file(self, allow_list=[]):
        if allow_list:
            self.allow_list = [x and y for (x, y) in zip(self.allow_list, allow_list)]
        self.target_sentences = []
        for i, sentence in enumerate(self.sentences):
            if self.allow_list[i]:
                self.addSentence(sentence)
                self.target_sentences.append(sentence)

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class SkLang:
    def __init__(self, filename, dir):
        self.filename = filename
        self.word2index = {}
        self.word2count = {}
        self.sentences = []
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # SOSとEOSをカウント

        self.data = []  # 触覚データファイルのパスリスト
        self.labels = []  # 触覚データのラベルリスト
        self.ono = []  # オノマトペテキストリスト
        name_to_label = {}  # オノマトペ単語とラベルのマッピング

        # CSVファイルの読み込み
        df = pd.read_csv(filename)
        num = df.shape[0]  # CSVファイルの行数を取得（データセットの数）
        for i in range(num):  # すべてのラベルを作成
            word = df.iloc[i, 2]  # 単語 (3列目)
            phoneme = df.iloc[i, 1]  # 音素 (2列目)
            ono_label = df.iloc[i, 0]  # ラベル (1列目)
            # 単語とラベルの辞書を作成・更新
            name_to_label[word] = i  # 単語をキー、インデックスを値として辞書に追加
            # オノマトペリストと音素リストに追加
            self.ono.append(word)
            self.sentences.append(phoneme)
        # 許可リストを初期化
        self.allow_list = [True] * len(self.sentences)

        # 触覚データファイルの読み込み
        target_dir = os.path.join(dir, "*")
        for path in glob.glob(target_dir):
            folder_name = os.path.basename(path)
            folder_name = unicodedata.normalize("NFKC", folder_name)

            if folder_name in name_to_label:
                label = name_to_label[folder_name]
                for file_path in glob.glob(os.path.join(path, "*.csv")):
                    self.data.append(file_path)  # ファイルパスを保存
                    self.labels.append(label)  # ラベルを保存
            else:
                print(f"警告: {folder_name}に対応するラベルが見つかりません。フォルダをスキップします。")

        # ラベルごとのデータを確認
        label_to_files = {label: [] for label in range(len(self.ono))}
        for file, label in zip(self.data, self.labels):
            label_to_files[label].append(file)
        self.label_to_files = label_to_files  # ラベルごとのファイルリストを保存

    def get_sentences(self):
        return self.sentences[::]

    def get_sentence(self, index):
        return self.sentences[index]

    def choice(self):
        while True:
            index = random.randint(0, len(self.allow_list) - 1)
            if self.allow_list[index]:
                break
        return self.sentences[index], index

    def get_allow_list(self, max_length):
        allow_list = []
        for sentence in self.sentences:
            if len(sentence.split()) < max_length:
                allow_list.append(True)
            else:
                allow_list.append(False)
        return allow_list

    def load_file(self, allow_list=[]):
        if allow_list:
            self.allow_list = [x and y for (x, y) in zip(self.allow_list, allow_list)]
        self.target_sentences = []
        for i, sentence in enumerate(self.sentences):
            if self.allow_list[i]:
                self.addSentence(sentence)
                self.target_sentences.append(sentence)

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __len__(self):
        """触覚データのサンプル数を返す"""
        return len(self.data)

    def __getitem__(self, index):
        sk_data_path = self.data[index]
        label = self.labels[index]
        ono = self.ono[label]
        phoneme = self.sentences[label]

        # 触覚データのCSVファイルを読み込む
        sk_data = pd.read_csv(sk_data_path)
        sk_tensor = torch.tensor(sk_data.values, dtype=torch.float32)

        return sk_tensor, label, ono, phoneme

def tensorFromSentence(lang, sentence, label):
    indexes = [lang.word2index[word] for word in sentence.split()]
    indexes.append(EOS_token)
    tensor = torch.tensor(indexes, dtype=torch.long).to(device).view(-1, 1)
    label_tensor = torch.tensor(label, dtype=torch.long).to(device)
    return tensor, label_tensor

def tensorsFromPair(lang):
    # langから任意の文1つを選択して入力/出力として使用
    sentence, label, _ = lang.choice()
    input_tensor, input_label_tensor = tensorFromSentence(lang, sentence, label)
    # 出力も同じ文とラベルを使用
    output_tensor, output_label_tensor = tensorFromSentence(lang, sentence, label)
    return input_tensor, output_tensor, input_label_tensor, output_label_tensor

# 触覚エンコーダークラス
class SkEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def forward(self, _input, hidden):
        out, new_hidden = self.gru(_input, hidden)
        last_layer_output = new_hidden[-1, :, :]  # 最後のレイヤーの結果を選択
        return last_layer_output, new_hidden

# 触覚デコーダークラス
class SkDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(output_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, _input, hidden):
        gru_output, hidden = self.gru(_input, hidden)
        output = self.linear(gru_output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

# オノマトペエンコーダークラス
class OnoEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)

    def forward(self, _input, hidden):
        embedded = self.embedding(_input).view(1, 1, -1)
        out, new_hidden = self.gru(embedded, hidden)
        return out, new_hidden

# オノマトペデコーダークラス
class OnoDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, _input, hidden):
        embedded = self.embedding(_input).view(1, 1, -1)
        relu_embedded = F.relu(embedded)
        gru_output, hidden = self.gru(relu_embedded, hidden)
        result = self.softmax(self.linear(gru_output[0]))
        return result, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)

# モデルのパラメータを保存する関数
def save_model_parameters(skEncoder, skDecoder, onoEncoder, onoDecoder, filename_postfix=""):
    skEncoder_filename = os.path.join(CHECKPOINT_DIR, f"skEncoder_params_{filename_postfix}.pth")
    skDecoder_filename = os.path.join(CHECKPOINT_DIR, f"skDecoder_params_{filename_postfix}.pth")
    onoEncoder_filename = os.path.join(CHECKPOINT_DIR, f"onoEncoder_params_{filename_postfix}.pth")
    onoDecoder_filename = os.path.join(CHECKPOINT_DIR, f"onoDecoder_params_{filename_postfix}.pth")

    torch.save(skEncoder.state_dict(), skEncoder_filename)
    torch.save(skDecoder.state_dict(), skDecoder_filename)
    torch.save(onoEncoder.state_dict(), onoEncoder_filename)
    torch.save(onoDecoder.state_dict(), onoDecoder_filename)

    print(f"[モデル保存] {skEncoder_filename}, {skDecoder_filename}, {onoEncoder_filename}, {onoDecoder_filename}")

# 既に保存されたモデルパラメータがあれば読み込む関数
def load_model_parameters_if_exists(skEncoder, skDecoder, onoEncoder, onoDecoder):
    if os.path.exists(SK_ENCODER_LATEST):
        skEncoder.load_state_dict(torch.load(SK_ENCODER_LATEST))
        print(f"[モデルロード] {SK_ENCODER_LATEST}")
    if os.path.exists(SK_DECODER_LATEST):
        skDecoder.load_state_dict(torch.load(SK_DECODER_LATEST))
        print(f"[モデルロード] {SK_DECODER_LATEST}")
    if os.path.exists(ONO_ENCODER_LATEST):
        onoEncoder.load_state_dict(torch.load(ONO_ENCODER_LATEST))
        print(f"[モデル로드] {ONO_ENCODER_LATEST}")
    if os.path.exists(ONO_DECODER_LATEST):
        onoDecoder.load_state_dict(torch.load(ONO_DECODER_LATEST))
        print(f"[モデルロード] {ONO_DECODER_LATEST}")

# ono_wordをインデックスシーケンスに変換する関数
def wordToIndexes(lang, word):
    indexes = [lang.word2index[w] for w in word.split()]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).to(device)

def main():
    # 触覚データとオノマトペデータの読み込み
    skono_train_dataset = SkLang('dkim/datasets/match.csv', 'dkim/datasets/sk')
    skono_train_dataloader = DataLoader(skono_train_dataset, batch_size=len(skono_train_dataset), shuffle=True, drop_last=True)
    input_lang = Lang('dkim/datasets/ono/onomatope-jaconv-label.txt')
    output_lang = Lang('dkim/datasets/ono/onomatope-jaconv-label.txt')
    INPUT_lang = Lang('dkim/datasets/ono/onomatope-jaconv-label.txt')
    OUTPUT_lang = Lang('dkim/datasets/ono/onomatope-jaconv-label.txt')
    input_lang.load_file(input_lang.get_allow_list(max_length))
    output_lang.load_file(output_lang.get_allow_list(max_length))
    INPUT_lang.load_file(input_lang.get_allow_list(max_length))
    OUTPUT_lang.load_file(output_lang.get_allow_list(max_length))

    # モデルの初期化
    onoCriterion = nn.CrossEntropyLoss()
    skCriterion = nn.MSELoss()
    onoEncoder = OnoEncoder(input_lang.n_words, embedding_size, hidden_size).to(device)
    onoDecoder = OnoDecoder(hidden_size, embedding_size, output_lang.n_words).to(device)
    skEncoder = SkEncoder(input_size, hidden_size, num_layers=num_layers).to(device)
    skDecoder = SkDecoder(hidden_size, output_size, num_layers=num_layers).to(device)

    onoEncoder_optimizer = optim.SGD(onoEncoder.parameters(), lr=learning_rate)
    onoDecoder_optimizer = optim.SGD(onoDecoder.parameters(), lr=learning_rate)
    skEncoder_optimizer = optim.Adam(skEncoder.parameters(), lr=learning_rate)
    skDecoder_optimizer = optim.Adam(skDecoder.parameters(), lr=learning_rate)

    load_model_parameters_if_exists(skEncoder, skDecoder, onoEncoder, onoDecoder)

    tactile_losses, onomatopoeia_losses, mse_losses, total_losses = [], [], [], []

    # 触覚データの生成
    for data in skono_train_dataloader:
        sk_data, sk_labels, _, _ = data
        sk_data = sk_data.squeeze().numpy()
        sk_labels = sk_labels.numpy()
    sk_data = np.concatenate(sk_data, axis=0)
    print(f"sk_dataのサイズ: {sk_data.shape}")
    reshaped_data = sk_data.flatten().reshape(-1, sequence_length, input_size)
    reshaped_labels = np.repeat(sk_labels, reshaped_data.shape[0] // len(sk_labels))
    reshaped_data = torch.tensor(reshaped_data, dtype=torch.float32)
    reshaped_labels = torch.tensor(reshaped_labels, dtype=torch.long)
    train_dataloader = DataLoader(TensorDataset(reshaped_data, reshaped_labels), batch_size=batch_size, shuffle=True, drop_last=True)

    for sk_epoch in range(sk_num_epochs):
        print(f"\n===== エポック [{sk_epoch + 1}/{sk_num_epochs}] =====")
        #--------------------------------------------------------------------------------------------------------------------------------------------音素復元単体
        for iter_count in range(n_iters):
            input_tensor, output_tensor, input_label, output_label = tensorsFromPair(input_lang)
            input_data = input_tensor.to(device)
            output_data = output_tensor.to(device)

            onoEncoder_hidden = onoEncoder.initHidden()
            onoEncoder_optimizer.zero_grad()
            onoDecoder_optimizer.zero_grad()

            for i in range(input_data.size(0)):
                _, onoEncoder_hidden = onoEncoder(input_data[i], onoEncoder_hidden)

            ono_loss = 0
            onoDecoder_input = torch.tensor([[SOS_token]], device=device)
            onoDecoder_hidden = onoEncoder_hidden

            for i in range(output_data.size(0)):
                onoDecoder_output, onoDecoder_hidden = onoDecoder(onoDecoder_input, onoDecoder_hidden)
                onoDecoder_input = output_data[i]
                ono_loss += onoCriterion(onoDecoder_output, output_data[i].view(-1).long())
                if onoDecoder_input.item() == EOS_token:
                    break

            ono_loss.backward()
            onoEncoder_optimizer.step()
            onoDecoder_optimizer.step()

            onomatopoeia_losses.append((ono_loss.item() / output_tensor.size(0)))

            if (iter_count + 1) % 100 == 0:
                print(f"オノマトペ学習 - イテレーション [{iter_count + 1}/{n_iters}], 損失: {ono_loss.item() / output_tensor.size(0):.6f}")

        # 予測された音素シーケンスを得る
        with torch.no_grad():
            onoDecoder_input = torch.tensor([[SOS_token]], device=device)
            onoDecoder_hidden = onoEncoder_hidden.clone()
            predicted_tokens = []

            for i in range(output_data.size(0)):
                onoDecoder_output, onoDecoder_hidden = onoDecoder(onoDecoder_input, onoDecoder_hidden)
                pred_token = torch.argmax(onoDecoder_output, dim=1)
                predicted_tokens.append(pred_token.item())
                onoDecoder_input = pred_token.unsqueeze(0).unsqueeze(0)
                if pred_token.item() == EOS_token:
                    break

            # インデックスを単語(音素)に変換
            predicted_sequence = [output_lang.index2word[idx] for idx in predicted_tokens if idx not in [SOS_token, EOS_token]]
            ground_truth_sequence = [output_lang.index2word[idx.item()] for idx in output_data if idx.item() not in [SOS_token, EOS_token]]
            input_sequence = [input_lang.index2word[idx.item()] for idx in input_data if idx.item() not in [SOS_token, EOS_token]]

            print("-----オノマトペ音素の比較-----")
            print("入力オノマトペ音素:", " ".join(input_sequence))
            print("正解オノマトペ音素:", " ".join(ground_truth_sequence))
            print("予測オノマトペ音素:", " ".join(predicted_sequence))
            print("----------------------------")

        #--------------------------------------------------------------------------------------------------------------------------------------------触覚データクラス復元
        sk_hidden_dict = {}
        for inputs, labels in train_dataloader:
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)
            skEncoder_hidden = skEncoder.initHidden(inputs.size(0))
            skEncoder_optimizer.zero_grad()
            skDecoder_optimizer.zero_grad()
            onoEncoder_optimizer.zero_grad()
            onoDecoder_optimizer.zero_grad()

            # encoder step
            input_length = inputs.size(1)
            skEncoder_outputs = torch.zeros(inputs.size(0), input_length, hidden_size, device=device)
            for t in range(input_length):
                skEncoder_output, skEncoder_hidden = skEncoder(inputs[:, t, :].unsqueeze(1), skEncoder_hidden)
                skEncoder_outputs[:, t, :] = skEncoder_output.squeeze(1)

            # skHidden save
            sk_hidden_dict[labels] = skEncoder_hidden

            # decoder step
            sk_loss = 0
            skDecoder_input = torch.zeros((inputs.size(0), 1, output_size), device=device)
            skDecoder_hidden = skEncoder_hidden
            for t in range(input_length):
                skDecoder_output, skDecoder_hidden = skDecoder(skDecoder_input, skDecoder_hidden)
                skDecoder_input = skDecoder_output
                sk_loss += skCriterion(skDecoder_output.squeeze(1), inputs[:, t, :])
        sk_loss = sk_loss / len(train_dataloader)

        print(f"触覚データ学習 - エポック [{sk_epoch + 1}/{sk_num_epochs}], 損失: {sk_loss:.6f}")

        #--------------------------------------------------------------------------------------------------------------------------------------------音素復元
        ono_hidden_dict = {}
        for iter_count2 in range(15):
            INPUT_tensor, OUTPUT_tensor, INPUT_label, OUTPUT_label = tensorsFromPair(input_lang)
            INPUT_ono_word = INPUT_lang.label2word[iter_count2]
            INPUT_ono_word_tensor = wordToIndexes(INPUT_lang, INPUT_ono_word).view(-1, 1).to(device)
            OUTPUT_data = INPUT_ono_word_tensor.clone()  # 入力と同じ文を使用

            onoEncoder_hidden = onoEncoder.initHidden()

            for i in range(INPUT_ono_word_tensor.size(0)):
                _, onoEncoder_hidden = onoEncoder(INPUT_ono_word_tensor[i].unsqueeze(0), onoEncoder_hidden)

            # onoHiddenを保存してmseを計算
            mse_ONO_hidden = onoEncoder_hidden
            ono_hidden_dict[iter_count2] = mse_ONO_hidden

            # この後ONO_lossの計算などが必要なロジックを実行
            ONO_loss = 0
            onoDecoder_input = torch.tensor([[SOS_token]], device=device)
            onoDecoder_hidden = onoEncoder_hidden

            for i in range(OUTPUT_data.size(0)):
                onoDecoder_output, onoDecoder_hidden = onoDecoder(onoDecoder_input, onoDecoder_hidden)
                onoDecoder_input = OUTPUT_data[i]
                ONO_loss += onoCriterion(onoDecoder_output, OUTPUT_data[i].view(-1).long())
                if onoDecoder_input.item() == EOS_token:
                    break

        with torch.no_grad():
            onoDecoder_input = torch.tensor([[SOS_token]], device=device)
            onoDecoder_hidden = onoEncoder_hidden.clone()
            predicted_tokens = []

            # デコーダーを再度動かして予測された音素シーケンスを得る
            for i in range(OUTPUT_data.size(0)):
                onoDecoder_output, onoDecoder_hidden = onoDecoder(onoDecoder_input, onoDecoder_hidden)
                pred_token = torch.argmax(onoDecoder_output, dim=1)
                predicted_tokens.append(pred_token.item())
                onoDecoder_input = pred_token.unsqueeze(0).unsqueeze(0)
                if pred_token.item() == EOS_token:
                    break

            # インデックスを音素(単語)に変換
            predicted_sequence = [INPUT_lang.index2word[idx] for idx in predicted_tokens if idx not in [SOS_token, EOS_token]]
            ground_truth_sequence = [INPUT_lang.index2word[idx.item()] for idx in OUTPUT_data if idx.item() not in [SOS_token, EOS_token]]
            input_sequence = [INPUT_lang.index2word[idx.item()] for idx in INPUT_ono_word_tensor if idx.item() not in [SOS_token, EOS_token]]

            print("-----オノマトペ音素の比較 -----")
            print("入力オノマトペ音素:", " ".join(input_sequence))
            print("正解オノマトペ音素:", " ".join(ground_truth_sequence))
            print("予測オノマトペ音素:", " ".join(predicted_sequence))
            print("-------------------------")

        #--------------------------------------------------------------------------------------------------------------------------------------------両方のコンテキストベクトルを近づける
        """
         0,z a r a z a r a,0ざらざら
         1,g o ts u g o ts u,1ごつごつ
         2,sh u w a sh u w a,2しゅわしゅわ
         3,ch i k u ch i k u,3ちくちく
         4,s u b e s u b e,4すべすべ
        """
        for j in range(15):
            sk_label_hiddens = defaultdict(list)
            labels_tensor = list(sk_hidden_dict.keys())[j]
            hidden_states = sk_hidden_dict[labels_tensor]
            for i, lbl in enumerate(labels_tensor):
                label_val = lbl.item()
                sample_hidden = hidden_states[:, i, :]
                last_layer_hidden = sample_hidden[-1, :]
                sk_label_hiddens[label_val].append(last_layer_hidden.unsqueeze(0))  # (1, hidden_size)

            for lbl in sk_label_hiddens:
                sk_label_hiddens[lbl] = torch.stack(sk_label_hiddens[lbl])  # (N, hidden_size)

            mse_loss = 0
            common_labels = set(ono_hidden_dict.keys()).intersection(sk_label_hiddens.keys())

            for lbl in common_labels:
                ono_vec = ono_hidden_dict[lbl].squeeze(0).squeeze(0)  # (hidden_size,)
                sk_vecs = sk_label_hiddens[lbl]  # (N, hidden_size)

                # ono_vecをsk_vecsのサイズに合わせて拡張
                mse_loss_label = F.mse_loss(sk_vecs, ono_vec.expand_as(sk_vecs))
                mse_loss += mse_loss_label

        mse_loss = mse_loss / len(common_labels)
        print(f"近付ける学習 - エポック [{sk_epoch + 1}/{sk_num_epochs}], MSE損失: {mse_loss.item():.6f}")

        #--------------------------------------------------------------------------------------------------------------------------------------------損失計算
        loss = sk_loss * 0.01 + ONO_loss * 0.2 + mse_loss
        loss.backward()

        mse_losses.append(mse_loss.item())
        total_losses.append(loss.item())  # 総損失を保存
        tactile_losses.append(sk_loss.item())  # 触覚データの損失を保存

        skEncoder_optimizer.step()
        skDecoder_optimizer.step()
        onoEncoder_optimizer.step()
        onoDecoder_optimizer.step()
        print(f"学習完了, total損失: {loss.item():.6f}")

        #--------------------------------------------------------------------------------------------------------------------------------------------触覚データクラスのコンテキストベクトルの保存
        with torch.no_grad():
            context_vectors_path = f'dkim/results/tactile_context_vectors_{sk_epoch + 1}.csv'
            with open(context_vectors_path, 'w') as f:
                f.write(','.join(map(str, range(hidden_size))) + ',label\n')
                for label, vectors in sk_hidden_dict.items():
                    for vector in vectors:
                        vector_np = vector.detach().cpu().numpy()
                        f.write(','.join(map(str, vector_np)) + f',{label}\n')
            print(f"Saved tactile context vectors for epoch {sk_epoch + 1} to {context_vectors_path}")

        #--------------------------------------------------------------------------------------------------------------------------------------------オノマトペ音素のコンテキストベクトルの保存
        with torch.no_grad():
            output_path = f"dkim/results/onomatope_context_vectors_{sk_epoch + 1}.csv"
            context_vectors = []
            data = pd.read_csv('dkim/datasets/match.csv')
            onomatopoeia_data = data.iloc[:, 1]  # 音素データ
            labels = data.iloc[:, 0]  # ラベル

            for idx, phoneme in enumerate(onomatopoeia_data):
                input_tensor, _ = tensorFromSentence(input_lang, phoneme, labels[idx])
                hidden = onoEncoder.initHidden()
                for i in range(input_tensor.size(0)):
                    _, hidden = onoEncoder(input_tensor[i], hidden)

                # 最後のhidden stateをコンテキストベクトルとして使用
                context_vector = hidden[-1].detach().cpu().numpy().squeeze()
                # 増強なしで直接保存リストに追加
                context_vectors.append(np.append(context_vector, labels.iloc[idx]))

            with open(output_path, 'w') as f:
                header = ",".join(map(str, range(context_vectors[0].shape[0]-1)))
                f.write(f"{header},label\n")
                for vector in context_vectors:
                    f.write(",".join(map(str, vector)) + "\n")

            print(f"Saved onomatopoeia context vectors for epoch {sk_epoch + 1} to {output_path}")

        #--------------------------------------------------------------------------------------------------------------------------------------------モデル中間保存
        if (sk_epoch + 1) % 50 == 0:
            save_model_parameters(skEncoder, skDecoder, onoEncoder, onoDecoder,
                                  filename_postfix=f"epoch{sk_epoch+1}")
            # 最新バージョンのチェックポイントも更新
            torch.save(skEncoder.state_dict(), SK_ENCODER_LATEST)
            torch.save(skDecoder.state_dict(), SK_DECODER_LATEST)
            torch.save(onoEncoder.state_dict(), ONO_ENCODER_LATEST)
            torch.save(onoDecoder.state_dict(), ONO_DECODER_LATEST)
            print(f"[checkpoint] epoch {sk_epoch+1} でモデルを保存しました。")

    #--------------------------------------------------------------------------------------------------------------------------------------------グラフ保存
    plt.figure()
    plt.plot(onomatopoeia_losses, label='Onomatopoeia Training Loss')
    plt.xlabel('Epoch (500)')
    plt.ylabel('Loss')
    plt.title('Onomatopoeia Training Loss Trend (500 Iterations)')
    plt.ylim(0, 5)
    plt.legend()
    plt.savefig(f'dkim/results/onomatopoeia_loss_graph_{sk_num_epochs}.png')
    plt.close()
    print("Saved onomatopoeia loss graph.")

    plt.figure()
    plt.plot(mse_losses, label='Mapping Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mse Loss Trend between Context Vectors')
    plt.legend()
    plt.savefig(f'dkim/results/mapping_loss_graph_{sk_num_epochs}.png')
    plt.close()
    print("Saved mse loss graph.")

    plt.figure()
    plt.plot(total_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss Trend')
    plt.ylim(0, 10)
    plt.legend()
    plt.savefig(f'dkim/results/total_loss_graph_{sk_num_epochs}.png')
    plt.close()
    print("Saved total loss graph.")

    plt.figure()
    plt.plot(tactile_losses, label='Tactile Data Training Loss')
    plt.xlabel('Epoch (150000)')
    plt.ylabel('Loss')
    plt.title('Tactile Data Training Loss Trend (500*300 Iterations)')
    plt.ylim(0, 30)
    plt.legend()
    plt.savefig(f'dkim/results/tactile_loss_graph_detailed_{sk_num_epochs}.png')
    plt.close()
    print("Saved tactile data loss graph.")

    save_model_parameters(skEncoder, skDecoder, onoEncoder, onoDecoder, filename_postfix=f"")
    # 最新バージョンのチェックポイントも更新
    torch.save(skEncoder.state_dict(), SK_ENCODER_LATEST)
    torch.save(skDecoder.state_dict(), SK_DECODER_LATEST)
    torch.save(onoEncoder.state_dict(), ONO_ENCODER_LATEST)
    torch.save(onoDecoder.state_dict(), ONO_DECODER_LATEST)

if __name__ == '__main__':
    main()