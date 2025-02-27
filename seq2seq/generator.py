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
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 学習に使用するデバイスを設定

# 触覚モデルのパラメータ
hidden_size = 256  # GRUの隠れ層サイズ
batch_size = 1  # バッチサイズ
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
        self.sentences = []  # 単語一覧
        self.labels = []  # ラベル一覧
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # SOSとEOSをカウント

        # txtファイルをデータフレームとして読み込み保存
        self.data = pd.read_csv(self.filename)

        # データフレームから単語とラベルをリストに保存
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
        num = df.shape[0]  # CSVファイルの行数（データセットの数）
        for i in range(num):
            word = df.iloc[i, 2]  # 単語 (3列目)
            phoneme = df.iloc[i, 1]  # 音素 (2列目)
            ono_label = df.iloc[i, 0]  # ラベル (1列目)

            name_to_label[word] = i
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
                    self.data.append(file_path)
                    self.labels.append(label)
            else:
                print(f"警告: {folder_name}に対応するラベルが見つかりません。フォルダをスキップします。")

        label_to_files = {label: [] for label in range(len(self.ono))}
        for file, label in zip(self.data, self.labels):
            label_to_files[label].append(file)
        self.label_to_files = label_to_files

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
        return len(self.data)

    def __getitem__(self, index):
        sk_data_path = self.data[index]
        label = self.labels[index]
        ono = self.ono[label]
        phoneme = self.sentences[label]

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
    # langから任意の文を1つ選んで入力/出力でともに使用
    sentence, label, _ = lang.choice()
    input_tensor, input_label_tensor = tensorFromSentence(lang, sentence, label)
    output_tensor, output_label_tensor = tensorFromSentence(lang, sentence, label)
    return input_tensor, output_tensor, input_label_tensor, output_label_tensor

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
        last_layer_output = new_hidden[-1, :, :]  # 最終レイヤーの結果を選択
        return last_layer_output, new_hidden

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

def main():
    # skono_train_datasetのロード
    skono_train_dataset = SkLang('dkim/datasets/match.csv', 'dkim/datasets/sk')
    skono_train_dataloader = DataLoader(skono_train_dataset, batch_size=len(skono_train_dataset), shuffle=False, drop_last=True)
    input_lang = Lang('dkim/datasets/ono/onomatope-jaconv-label.txt')
    output_lang = Lang('dkim/datasets/ono/onomatope-jaconv-label.txt')
    input_lang.load_file(input_lang.get_allow_list(max_length))
    output_lang.load_file(output_lang.get_allow_list(max_length))

    skEncoder = SkEncoder(input_size, hidden_size, num_layers=num_layers).to(device)
    skDecoder = SkDecoder(hidden_size, output_size, num_layers=num_layers).to(device)
    onoEncoder = OnoEncoder(input_lang.n_words, embedding_size, hidden_size).to(device)
    onoDecoder = OnoDecoder(hidden_size, embedding_size, output_lang.n_words).to(device)

    # 事前学習済みモデルパラメータを読み込み
    skEncoder.load_state_dict(torch.load('dkim/seq2seq/models/skEncoder.pth', map_location=device))
    skDecoder.load_state_dict(torch.load('dkim/seq2seq/models/skDecoder.pth', map_location=device))
    onoEncoder.load_state_dict(torch.load('dkim/seq2seq/models/onoEncoder.pth', map_location=device))
    onoDecoder.load_state_dict(torch.load('dkim/seq2seq/models/onoDecoder.pth', map_location=device))

    # skono_train_dataset読み込み
    for data in skono_train_dataloader:
        sk_data, sk_labels, _, _ = data
        sk_data = sk_data.squeeze().numpy()
        sk_labels = sk_labels.numpy()

    # (batch, seq_len, input_size)形式へ変換
    reshaped_data = sk_data.flatten().reshape(-1, sequence_length, input_size)
    reshaped_labels = np.repeat(sk_labels, reshaped_data.shape[0] // len(sk_labels))
    reshaped_data = torch.tensor(reshaped_data, dtype=torch.float32)
    reshaped_labels = torch.tensor(reshaped_labels, dtype=torch.long)
    train_dataloader = DataLoader(TensorDataset(reshaped_data, reshaped_labels), batch_size=batch_size, shuffle=False, drop_last=True)

    # 例としての入力文章
    input_sentence = "s u b e s u b e"
    input_tensor = torch.tensor(
        [input_lang.word2index[word] for word in input_sentence.split()] + [EOS_token],
        dtype=torch.long,
    ).to(device).view(-1, 1)

    # onoEncoderを使ったコンテキストベクトル生成
    onoEncoder_hidden = onoEncoder.initHidden()
    for i in range(input_tensor.size(0)):
        _, onoEncoder_hidden = onoEncoder(input_tensor[i], onoEncoder_hidden)

    # skDecoderを使って触覚データを復元
    skDecoder_input = torch.zeros((1, 1, output_size), device=device)  # 初回入力: 0テンソル
    skDecoder_hidden = onoEncoder_hidden.repeat(num_layers, 1, 1)  # hidden stateサイズ調整

    decoded_sequence = []
    for _ in range(sequence_length):
        skDecoder_output, skDecoder_hidden = skDecoder(skDecoder_input, skDecoder_hidden)
        decoded_sequence.append(skDecoder_output.squeeze(0).detach().cpu().numpy())
        skDecoder_input = skDecoder_output

    # 復元された触覚データを配列に変換
    generated_tactile_data = np.concatenate(decoded_sequence, axis=0)

    # 結果出力およびCSV保存
    output_csv_path = "generated_tactile_data.csv"
    pd.DataFrame(generated_tactile_data).to_csv(output_csv_path, index=False, header=False)
    print(f"オノマトペ '{input_sentence}' に対する触覚データを {output_csv_path} に保存しました。")

    # =========================================
    # 触覚データ復元 (skEncoder, skDecoder)
    # =========================================
    # --- 正解オノマトペのマッピング ---
    file_to_onomatopoeia = {
        0: "z a r a z a r a",
        1: "g o ts u g o ts u",
        2: "sh u w a sh u w a",
        3: "ch i k u ch i k u",
        4: "s u b e s u b e"
    }

    # --- 正解率計算 ---
    with torch.no_grad():
        total_correct = 0
        file_correct_counts = defaultdict(int)
        file_total_counts = defaultdict(int)

        for j, (inputs, labels) in enumerate(train_dataloader, 1):
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)

            skEncoder_hidden = skEncoder.initHidden(inputs.size(0))

            # --- Encoder Step ---
            for t in range(inputs.size(1)):
                skEncoder_output, skEncoder_hidden = skEncoder(inputs[:, t, :].unsqueeze(1), skEncoder_hidden)

            # --- Decoder Step ---
            onoDecoder_input = torch.tensor([[SOS_token]], device=device)
            onoDecoder_hidden = skEncoder_hidden[-1].unsqueeze(0)

            predicted_tokens = []
            for _ in range(max_length):
                onoDecoder_output, onoDecoder_hidden = onoDecoder(onoDecoder_input, onoDecoder_hidden)
                pred_token = torch.argmax(onoDecoder_output, dim=1)
                predicted_tokens.append(pred_token.item())
                onoDecoder_input = pred_token.unsqueeze(0).unsqueeze(0)
                if pred_token.item() == EOS_token:
                    break

            # --- 予測結果に変換 ---
            predicted_sequence = [
                output_lang.index2word[idx]
                for idx in predicted_tokens
                if idx not in [SOS_token, EOS_token]
            ]

            predicted_onomatopoeia = " ".join(predicted_sequence)
            file_id = labels.item()
            correct_onomatopoeia = file_to_onomatopoeia.get(file_id, "")

            # --- 正解かどうかを判定 ---
            file_total_counts[file_id] += 1
            if predicted_onomatopoeia == correct_onomatopoeia:
                file_correct_counts[file_id] += 1
                total_correct += 1

            print(f"ファイル {file_id}番 - 予測オノマトペ: {predicted_onomatopoeia}, 正解: {correct_onomatopoeia}")

        # --- ファイル別の正解率を出力 ---
        for file_id, total_count in file_total_counts.items():
            correct_count = file_correct_counts[file_id]
            accuracy = correct_count / total_count * 100
            print(f"ファイル {file_id}番 正解率: {accuracy:.2f}% ({correct_count}/{total_count})")

        # --- 全体正解率を出力 ---
        overall_accuracy = total_correct / sum(file_total_counts.values()) * 100
        print(f"全体正解率: {overall_accuracy:.2f}%")

if __name__ == '__main__': 
    main()


""" 
0 colk z a r a z a r a, 1 denim g o ts u g o ts u, 2 gaze sh u w a sh u w a, 3 kanaami ch i k u ch i k u, 4 koutaku s u b e s u b e 

# === 264種類のオノマトペ生成 ===

        for _ in range(264):
            # 중복 검사 while 루프
            while True:
                input_tensor, output_tensor, _, _ = tensorsFromPair(input_lang)
                input_data = input_tensor.to(device)
                output_data = output_tensor.to(device)

                # 현재 문장의 음소 시퀀스 추출
                input_sequence_str_list = [
                    input_lang.index2word[idx.item()]
                    for idx in input_data
                    if idx.item() not in [SOS_token, EOS_token]
                ]
                phoneme_str = " ".join(input_sequence_str_list)

                if phoneme_str not in used_sentences:
                    # 중복 아님 → 등록하고 탈출
                    used_sentences.add(phoneme_str)
                    break
                # 중복이라면 while True 계속 반복 → 새 문장 뽑기

            # onoEncoder로 인코딩
            onoEncoder_hidden = onoEncoder.initHidden()
            for i in range(input_data.size(0)):
                _, onoEncoder_hidden = onoEncoder(input_data[i], onoEncoder_hidden)

            # 컨텍스트 벡터(마지막 hidden state) 추출
            context_vector = onoEncoder_hidden[-1].detach().cpu().numpy().squeeze()

            # 리스트에 저장
            context_vectors.append(context_vector)
            phoneme_labels.append(phoneme_str)

            # ----- デコーダ 로직 (예측) -----
            onoDecoder_input = torch.tensor([[SOS_token]], device=device)
            onoDecoder_hidden = onoEncoder_hidden

            for i in range(output_data.size(0)):
                onoDecoder_output, onoDecoder_hidden = onoDecoder(onoDecoder_input, onoDecoder_hidden)
                onoDecoder_input = output_data[i]
                if onoDecoder_input.item() == EOS_token:
                    break

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

                predicted_sequence = [
                    output_lang.index2word[idx]
                    for idx in predicted_tokens
                    if idx not in [SOS_token, EOS_token]
                ]
                ground_truth_sequence = [
                    output_lang.index2word[idx.item()]
                    for idx in output_data
                    if idx.item() not in [SOS_token, EOS_token]
                ]
                input_sequence = [
                    input_lang.index2word[idx.item()]
                    for idx in input_data
                    if idx.item() not in [SOS_token, EOS_token]
                ]

                print("-----オノマトペ音素の比較-----")
                print("入力オノマトペ音素:", " ".join(input_sequence))
                print("正解オノマトペ音素:", " ".join(ground_truth_sequence))
                print("予測オノマトペ音素:", " ".join(predicted_sequence))
                print("----------------------------")

        # 264개 생성 후, CSV 파일로 컨텍스트 벡터 저장
        output_csv = "ono_context_vectors.csv"
        hidden_dim = context_vectors[0].shape[0]
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [f"h{i}" for i in range(hidden_dim)] + ["phoneme"]
            writer.writerow(header)

            for vec, label in zip(context_vectors, phoneme_labels):
                row = list(vec) + [label]
                writer.writerow(row)

        print(f"オノマトペ音素のコンテキストベクトルを {output_csv} に保存しました。")"""