#导入模块
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import Callback

#文件


train_file = "/home/cp/DataSet/CONLL2003/train.txt"
val_file = "/home/cp/DataSet/CONLL2003/valid.txt"
test_file = "/home/cp/DataSet/CONLL2003/test.txt"
embedding_file = '/home/cp/DataSet/glove.6B/glove.6B.100d.txt'

#参数：
window_length =5

num_classes = 9

batch_size = 128
epoch = 1



####1 ：建立单词词典索引 ####
def get_word_index(datafile):
    """
    :param datafile: 文件列表
    :return: 词典{单词：索引}
    """

    word_list = []#保存所有的单词
    tag_set =set()
    word_2_index = dict()
    for file in datafile:
        with open(file,encoding='utf-8') as f:
            for line in f:
                if line == '\r\n' or line == '\n': continue
                line = line.replace('\r\n', '')
                split_line = line.split()
                word_list.append(split_line[0])
                tag_set.add(split_line[-1])
    # 统计单词出现的次数
    for word in word_list:
        if word in word_2_index:
            word_2_index[word] +=1
        else:
            word_2_index[word]  =1
    #根据单词的词频来确定索引
    word_count = list(word_2_index.items())
    word_count.sort(key=lambda x:x[0],reverse=True)#词频出现次数高的在前面
    sorted_voc = [w[0] for w in word_count]#单词
    word_2_index = dict(list(zip(sorted_voc,list(range(1,len(sorted_voc)+1)))))
    print("单词的长度是",len(word_2_index))

    #加入未登录词
    word_2_index["padding"] =0
    word_2_index['unknow'] =len(word_2_index)
    tag_2_index = {t :i for i,t in enumerate(tag_set)}
    return word_2_index,tag_2_index



def padd_sentence(_index,word_2_index,window):
    """
    :param _index: 单词索引 或是标签索引
    :param word_2_index: 单词词典
    :param window: 窗口大小
    :return: 填充的句子
    """
    pad = window//2
    num =len(_index)
    pad_sent = []
    #在一句话的前后补0
    for i in range(pad):
        _index.insert(0,word_2_index['padding'])
        _index.append(word_2_index['padding'])
    for i in range(num):
        pad_sent.append(_index[i:i+window])
    return pad_sent


def text_to_index_array(FILE_NAME, max_len):
    words = []  # 单词对应的索引
    tag = []  # 标签的额索引
    sentence = []
    sentence_tag = []

    # get max words in sentence
    sentence_length = 0

    for line in open(FILE_NAME):
        if line in ['\n', '\r\n']:
            index_line = padd_sentence(words, word_2_index, 5)
            sentence.extend(index_line)
            sentence_tag.extend(tag)

            sentence_length = 0
            words = []
            tag = []
        else:
            assert (len(line.split()) == 4)
            if sentence_length > max_len:
                max_len = sentence_length
            sentence_length += 1

            word = line.split()[0]
            if word in word_2_index:
                words.append(word_2_index[word])  # 单词转索引
            else:
                # charVec.append(0)	# 索引字典里没有的词转为数字0
                words.append(word_2_index['retain-unknown'])
            # print(words)

            t = line.split()[3]

            # Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc
            if t.endswith('O'):  # none
                # tag.append(np.asarray([1, 0, 0, 0, 0]))
                tag.append(tags.index('None'))
            elif t.endswith('PER'):
                # tag.append(np.asarray([0, 1, 0, 0, 0]))
                tag.append(tags.index('Person'))
            elif t.endswith('LOC'):
                # tag.append(np.asarray([0, 0, 1, 0, 0]))
                tag.append(tags.index('Location'))
            elif t.endswith('ORG'):
                # tag.append(np.asarray([0, 0, 0, 1, 0]))
                tag.append(tags.index('Organisation'))
            elif t.endswith('MISC'):
                # tag.append(np.asarray([0, 0, 0, 0, 1]))
                tag.append(tags.index('Misc'))
            else:
                # tag.append(np.asarray([0, 0, 0, 0, 0]))
                print("error in input" + str(t))



    index_line = padd_sentence(words, word_2_index, 5)

    sentence.extend(index_line)

    sentence_tag.extend(tag)


    print("max sentence size is : " + str(max_len))
    assert (len(sentence) == len(sentence_tag))

    return np.asarray(sentence), sentence_tag


def get_label_by_index(index):
    return [index_2_label.get(i) for i in index]


def read_embedding_list(file_path):
    embedding_word_dict = {}
    embedding_list = []
    f = open(file_path, encoding='utf-8')

    for index, line in enumerate(f):

        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue
        embedding_list.append(coefs)
        embedding_word_dict[word.lower()] = len(embedding_word_dict)  # 将嵌入词典的所有词变为小写!!
    f.close()
    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list( words_dict):
    embedding_list, embedding_word_dict = read_embedding_list(embedding_file)
    embedding_dim =100

    embeddings_matrix = np.zeros((len(words_dict), embedding_dim))

    for i, word in enumerate(words_dict):
        if word in embedding_word_dict:
            word_id = embedding_word_dict[word]
            embedding_vector = embedding_list[word_id]


            embeddings_matrix[i, :] = embedding_vector[:embedding_dim]
        else:
            embedding_i = np.random.uniform(-0.25, 0.25, embedding_dim)
            embeddings_matrix[i, :] = embedding_i
    return embeddings_matrix



if __name__ == '__main__':

    data_file = [train_file, val_file, test_file]
    tags = ['None', 'Person', 'Location', 'Organisation', 'Misc']

    word_2_index,tag_2_index = get_word_index(data_file)

    nb_words = len(word_2_index)

    index_2_word = {w:i for i ,w in word_2_index.items()}
    index_2_label = {i:t for t,i in tag_2_index.items()}


    train_x ,train_y = text_to_index_array(train_file,max_len=window_length)
    train_y = to_categorical(np.asarray(train_y), num_classes)

    dev_x, dev_y = text_to_index_array(val_file, max_len=window_length)
    test_x, test_y = text_to_index_array(test_file, max_len=window_length)

    test_x = pad_sequences(test_x, maxlen=5)
    test_y = to_categorical(np.asarray(test_y), num_classes)
    print('Shape of data  tensor:', test_x.shape)
    print('Shape of label tensor:', test_y.shape)

    dev_x = pad_sequences(dev_x, maxlen=5)
    dev_y = to_categorical(np.asarray(dev_y), num_classes)
    print('Shape of dev data  tensor:', dev_x.shape)
    print('Shape of dev label tensor:', dev_y.shape)

    word_embedding_matrix = clear_embedding_list( word_2_index)
    print(word_embedding_matrix.shape)#(14619, 100)


    #建立模型
    model = Sequential()
    model.add(Embedding(input_dim=nb_words,output_dim=word_embedding_matrix.shape[1],
                        weights=[word_embedding_matrix],input_length = window_length,trainable =False ))
    blstm = Bidirectional(LSTM(units=100,
                                    return_sequences=False,
                                    kernel_regularizer=l2(1e-4),
                                    bias_regularizer=l2(1e-4)),
                          name='privateLSTM')
    model.add(blstm)
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    plot_model(model=model,to_file='./figure.png',show_shapes=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=epoch,
                        validation_data= (dev_x,dev_y)

                         )
