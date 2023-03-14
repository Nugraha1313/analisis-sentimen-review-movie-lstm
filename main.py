import ast
import sys
import time
# from random import random
import numpy as np
import pandas as pd
import torch
import torchtext
import torchtext.vocab as vocab
import torch.nn as nn
import seaborn as sns
# import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from nltk.corpus import stopwords
from collections import Counter
import re
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from gui import Ui_MainWindow

# datasetFilePath = ""
# isReadDataset = False
global glove
glove = vocab.GloVe(name='6B', dim=100)

class window(QtWidgets.QMainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        # self.ui.importDatasetPushButton.clicked.connect(self.read_dataset)
        self.ui.predictButton.clicked.connect(self.app)
        # QtGui.QGuiApplication.processEvents()

    def clearOutput(self):
        self.ui.outputText.clear()
        self.ui.outputText.setPlainText('')
        # self.ui.outputText.setText('')

    def read_dataset(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '', "CSV Files (*.csv)")
        global datasetFilePath
        datasetFilePath = fname[0]
        # isReadDataset = True
        self.app()

    def app(self):
        self.clearOutput()
        # self.ui.clearOutputText()
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")

        """## Importing the Dataset

        Now we import the dataset
        """

        # df = pd.read_csv("rottentomatoes-400k.csv")
        # df.head()

        """## Normalize Scores

        The scores are a bit funky, I'll need one decimal scores
        """

        # df["Score"] = df["Score"].round(-1) / 100
        # df.head()

        """## Checking the value counts

        Yeah, just check it
        """

        # dd = df["Score"].value_counts().sort_index()
        # dd_indexes = np.array(dd.index.tolist())
        # sns.barplot(x=dd_indexes, y=dd)
        # print(dd)

        """## Balancing the dataset

        Oof that's a bit something else, We'll need to balance out the dataset. As you can see, the positive reviews weighs more than the negative ones. We need to sort that out.

        > Since I don't actually need all the data and this is just a just-for-fun notebook to start out with Kaggle, I'll just get ahead and pick the `0.9` and `1.0` for positive and `0.0` through `0.4` for negative reviews. It's also nice for me to wait less time so yeah.
        """

        # positive_reviews_slice = df[df.Score >= 0.9]
        # negative_reviews_slice = df[df.Score <= 0.4]

        # Resample them to 50k samples each for balance
        # positive_reviews_slice = resample(
        #     positive_reviews_slice, replace=False, n_samples=50000, random_state=123
        # )
        # negative_reviews_slice = resample(
        #     negative_reviews_slice, replace=False, n_samples=50000, random_state=123
        # )

        # dd_slice = pd.concat([positive_reviews_slice, negative_reviews_slice])
        # dd_slice["Score"] = np.where(dd_slice["Score"] <= 0.4, "negative", "positive")

        # dd_slice_indexes = np.array(dd_slice["Score"].value_counts().index.tolist())
        # sns.barplot(x=dd_slice_indexes, y=dd_slice["Score"].value_counts())

        """## Split Training and Testing Data

        Let's split the training and testing data
        """

        # df = dd_slice

        # X, y = df["Review"].values, df["Score"].values
        # x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)

        # print(f"\nShape of the train data is {x_train.shape}")
        # print(f"\nShape of the test data is {x_test.shape}")
        # df.head()

        """## Tokenization function

        Let's make the tokenization function
        """

        def preprocess_string(s):
            s = re.sub(r"[^\w\s]", "", s)
            s = re.sub(r"\s+", "", s)
            s = re.sub(r"\d", "", s)
            return s

        def tokenize(x_train, y_train, x_val, y_val):
            # load pre-trained word vectors
            # glove = vocab.GloVe(name='6B', dim=100)
            # tokenize and encode sentences
            final_list_train, final_list_test = [], []
            for sent in x_train:
                encoded_sent = []
                for word in sent.lower().split():
                    word = preprocess_string(word)
                    if word in glove.stoi:
                        encoded_sent.append(glove.stoi[word])
                final_list_train.append(encoded_sent)

            for sent in x_val:
                encoded_sent = []
                for word in sent.lower().split():
                    word = preprocess_string(word)
                    if word in glove.stoi:
                        encoded_sent.append(glove.stoi[word])
                final_list_test.append(encoded_sent)

            encoded_train = [1 if label == 'positive' else 0 for label in y_train]
            encoded_test = [1 if label == 'positive' else 0 for label in y_val]
            return np.array(final_list_train, dtype=object), np.array(encoded_train), np.array(final_list_test,dtype=object), np.array(encoded_test), glove.stoi

        """And then, tokenize em up!"""

        # x_train, y_train, x_test, y_test, vocab = tokenize(x_train, y_train, x_test, y_test)

        # print(f"\nLength of vocabulary is {len(vocab)}")
        # print('vocab : ', vocab)


        """## Analysing the review length

        Let's analyse the review length
        """

        # rev_len = [len(i) for i in x_train]
        # pd.Series(rev_len).hist()
        # plt.show()
        # pd.Series(rev_len).describe()

        """## Paddings

        Now that we know that our dataset's review text has a mean length of 6-7 words, we will pad each of the sequence to the max length.
        """

        def padding_(sentences, seq_len):
            features = np.zeros((len(sentences), seq_len), dtype=int)
            for ii, review in enumerate(sentences):
                if len(review) != 0:
                    features[ii, -len(review):] = np.array(review)[:seq_len]
            return features

        # x_train_pad = padding_(x_train, 52)
        # x_test_pad = padding_(x_test, 52)

        """## Batching Up and Loading The Tensors

        We now will set the `batch_size` and load the tensors. Make sure to shuffle your data.
        """

        # Create Tensor datasets
        # train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
        # valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

        # Dataloaders
        batch_size = 50

        # train_loader = DataLoader(
        #     train_data, shuffle=True, batch_size=batch_size, drop_last=True
        # )
        # valid_loader = DataLoader(
        #     valid_data, shuffle=True, batch_size=batch_size, drop_last=True
        # )

        """### Obtain one batch of training data

        Just to check.
        """

        # dataiter = iter(train_loader)
        # sample_x, sample_y = next(dataiter)

        # print("Sample input size: ", sample_x.size())
        # print("Sample input: \n", sample_x)
        # print("Sample input: \n", sample_y)

        """## Modelling

        Now we're getting somewhere.
        """

        class SentimentRNN(nn.Module):
            def __init__(
                    self,
                    no_layers,
                    vocab_size,
                    hidden_dim,
                    embedding_dim,
                    drop_prob=0.5,
                    bidirectional=False,
            ):
                super(SentimentRNN, self).__init__()

                self.output_dim = output_dim
                self.hidden_dim = hidden_dim

                self.no_layers = no_layers
                self.vocab_size = vocab_size

                # embedding and LSTM layers
                self.embedding = nn.Embedding(vocab_size, embedding_dim)

                # lstm
                self.lstm = nn.LSTM(
                    input_size=embedding_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=no_layers,
                    batch_first=True,
                )

                # dropout layer
                self.dropout = nn.Dropout(0.3)

                # linear and sigmoid layer
                self.fc = nn.Linear(self.hidden_dim, output_dim)
                self.sig = nn.Sigmoid()

            def forward(self, x, hidden):
                batch_size = x.size(0)

                # embeddings and lstm_out
                embeds = self.embedding(x)
                lstm_out, hidden = self.lstm(embeds, hidden)

                lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

                # dropout and fully connected layer
                out = self.dropout(lstm_out)
                out = self.fc(out)

                # sigmoid function
                sig_out = self.sig(out)

                # reshape to be batch_size first
                sig_out = sig_out.view(batch_size, -1)

                sig_out = sig_out[:, -1]  # get last batch of labels

                # return last sigmoid output and hidden state
                return sig_out, hidden

            def init_hidden(self, batch_size):
                # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
                # initialized to zero, for hidden state and cell state of LSTM
                h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
                c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
                hidden = (h0, c0)
                return hidden

        no_layers = 2
        vocab_size = 400001  # extra 1 for padding
        embedding_dim = 64
        output_dim = 1
        hidden_dim = 256

        model = SentimentRNN(
            no_layers=no_layers,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            drop_prob=0.5,
        )

        # print(model)

        """## Training

        Let's train the data, shall we?

        ### Prepping up criterion, optimization, and prediction

        Let's use `BCELoss` for the criterion.
        """

        # lr = 0.001
        # criterion = nn.BCELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Function to predict accuracy
        def acc(pred, label):
            pred = torch.round(pred.squeeze())
            return torch.sum(pred == label.squeeze()).item()

        """### Actually training the data

        Let's start the training.
        """

        # clip = 5
        # epochs = 5
        # valid_loss_min = np.Inf
        # # train for some number of epochs
        # epoch_tr_loss, epoch_vl_loss = [], []
        # epoch_tr_acc, epoch_vl_acc = [], []
        #
        # for epoch in range(epochs):
        #     train_losses = []
        #     train_acc = 0.0
        #     model.train()
        #     # initialize hidden state
        #     h = model.init_hidden(batch_size)
        #     for inputs, labels in train_loader:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         # Creating new variables for the hidden state, otherwise
        #         # we'd backprop through the entire training history
        #         h = tuple([each.data for each in h])
        #
        #         model.zero_grad()
        #         output, h = model(inputs, h)
        #
        #         # calculate the loss and perform backprop
        #         loss = criterion(output.squeeze(), labels.float())
        #         loss.backward()
        #         train_losses.append(loss.item())
        #         # calculating accuracy
        #         accuracy = acc(output, labels)
        #         train_acc += accuracy
        #         # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #         nn.utils.clip_grad_norm_(model.parameters(), clip)
        #         optimizer.step()
        #
        #     val_h = model.init_hidden(batch_size)
        #     val_losses = []
        #     val_acc = 0.0
        #     model.eval()
        #     for inputs, labels in valid_loader:
        #         val_h = tuple([each.data for each in val_h])
        #
        #         inputs, labels = inputs.to(device), labels.to(device)
        #
        #         output, val_h = model(inputs, val_h)
        #         val_loss = criterion(output.squeeze(), labels.float())
        #
        #         val_losses.append(val_loss.item())
        #
        #         accuracy = acc(output, labels)
        #         val_acc += accuracy
        #
        #     epoch_train_loss = np.mean(train_losses)
        #     epoch_val_loss = np.mean(val_losses)
        #     epoch_train_acc = train_acc / len(train_loader.dataset)
        #     epoch_val_acc = val_acc / len(valid_loader.dataset)
        #     epoch_tr_loss.append(epoch_train_loss)
        #     epoch_vl_loss.append(epoch_val_loss)
        #     epoch_tr_acc.append(epoch_train_acc)
        #     epoch_vl_acc.append(epoch_val_acc)
        #     print(f"Epoch {epoch+1}")
        #     print(f"train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}")
        #     print(f"train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}")
        #     if epoch_val_loss <= valid_loss_min:
        #         torch.save(model.state_dict(), "state_dict.pt")
        #         print(
        #             "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
        #                 valid_loss_min, epoch_val_loss
        #             )
        #         )
        #         valid_loss_min = epoch_val_loss
        #     print(25 * "==")

        # fig = plt.figure(figsize = (20, 6))
        # plt.subplot(1, 2, 1)
        # plt.plot(epoch_tr_acc, label='Train Acc')
        # plt.plot(epoch_vl_acc, label='Validation Acc')
        # plt.title("Accuracy")
        # plt.legend()
        # plt.grid()
        #
        # plt.subplot(1, 2, 2)
        # plt.plot(epoch_tr_loss, label='Train loss')
        # plt.plot(epoch_vl_loss, label='Validation loss')
        # plt.title("Loss")
        # plt.legend()
        # plt.grid()
        #
        # plt.show()

        # Prediction Phase
        def predict_text(text):
            self.ui.outputText.clear()
            with open('vocabs.txt', 'r', encoding="utf-8") as file:
                vocab_read = file.read()
            vocab = ast.literal_eval(vocab_read)
            PATH = "state_dict.pt"
            model.load_state_dict(torch.load(PATH, map_location=device))
            model.eval()
            model.to(device)
            word_seq = np.array(
                [
                    vocab[preprocess_string(word)]
                    for word in text.split()
                    if preprocess_string(word) in vocab.keys()
                ]
            )
            word_seq = np.expand_dims(word_seq, axis=0)
            pad = torch.from_numpy(padding_(word_seq, 500))
            inputs = pad.to(device)
            batch_size = 1
            h = model.init_hidden(batch_size)
            h = tuple([each.data for each in h])
            output, h = model(inputs, h)
            return output.item()

        inputSentence = self.ui.predictSentence.toPlainText()
        print("=" * 70)
        print(inputSentence)
        print("=" * 70)
        pro = predict_text(inputSentence.lower())
        status = "Positive" if pro >= 0.5 else "Negative"
        pro = (1 - pro) if status == "Negative" else pro
        print(f"Predicted sentiment is {status} with a probability of {pro}")
        output = status
        self.ui.outputText.append(output)
        output = ''

def app():
    app = QtWidgets.QApplication(sys.argv)
    win = window()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    app()