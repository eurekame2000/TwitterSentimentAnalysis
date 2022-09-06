

#模型预测并生成结果文件
# 对测试集进行预测
import numpy as np
import pandas as pd
import pandas as pd
import nltk
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from nltk import SnowballStemmer

# 导入paddlehub和paddle包
import paddlehub as hub
import re
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# train = pd.read_csv('../input/sentiment-data/train.csv')         # 有标签的训练数据文件
# test = pd.read_csv('../input/sentiment-data/test.csv')           # 要进行预测的测试数据文件
# sub = pd.read_csv('../input/sentiment-data/sample.csv')      # 提交结果文件范例

train = pd.read_csv('data/train.csv')         # 有标签的训练数据文件
test = pd.read_csv('data/test.csv')           # 要进行预测的测试数据文件
sub = pd.read_csv('data/sample.csv')      # 提交结果文件范例

# 文本清洗

chat_words_str = """
AFAIK=As Far As I Know
AFK=Away From Keyboard
ASAP=As Soon As Possible
ATK=At The Keyboard
ATM=At The Moment
A3=Anytime, Anywhere, Anyplace
BAK=Back At Keyboard
BBL=Be Back Later
BBS=Be Back Soon
BFN=Bye For Now
B4N=Bye For Now
BRB=Be Right Back
BRT=Be Right There
BTW=By The Way
B4=Before
B4N=Bye For Now
CU=See You
CUL8R=See You Later
CYA=See You
FAQ=Frequently Asked Questions
FC=Fingers Crossed
FWIW=For What It's Worth
FYI=For Your Information
GAL=Get A Life
GG=Good Game
GN=Good Night
GMTA=Great Minds Think Alike
GR8=Great!
G9=Genius
IC=I See
ICQ=I Seek you (also a chat program)
ILU=ILU: I Love You
IMHO=In My Honest/Humble Opinion
IMO=In My Opinion
IOW=In Other Words
IRL=In Real Life
KISS=Keep It Simple, Stupid
LDR=Long Distance Relationship
LMAO=Laugh My A.. Off
LOL=Laughing Out Loud
LTNS=Long Time No See
L8R=Later
MTE=My Thoughts Exactly
M8=Mate
NRN=No Reply Necessary
OIC=Oh I See
PITA=Pain In The A..
PRT=Party
PRW=Parents Are Watching
ROFL=Rolling On The Floor Laughing
ROFLOL=Rolling On The Floor Laughing Out Loud
ROTFLMAO=Rolling On The Floor Laughing My A.. Off
SK8=Skate
STATS=Your sex and age
ASL=Age, Sex, Location
THX=Thank You
TTFN=Ta-Ta For Now!
TTYL=Talk To You Later
U=You
U2=You Too
U4E=Yours For Ever
WB=Welcome Back
WTF=What The F...
WTG=Way To Go!
WUF=Where Are You From?
W8=Wait...
7K=Sick:-D Laugher
"""
chat_words_map_dict = {}
chat_words_list = []
for line in chat_words_str.split("\n"):
    # 去除多余空格
    line = ' '.join(line.split())
    if line != "":
        cw = line.split("=")[0]
        cw_expanded = line.split("=")[1]
        chat_words_list.append(cw)
        chat_words_map_dict[cw] = cw_expanded
chat_words_list = set(chat_words_list)

STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
    Clean text
    :param text: the string of text
    :return: text string after cleaning
    """
    print("初始句子：" + text)

    # 去除非ascii码
    text = re.sub(r'[^\x00-\x7F]+', '\'', text)
    print("去除非ascii码" + text)
    text = re.sub(r'@\S+', ' ', text)  # 删除 @用户
    print("qu@" + text)
    text = re.sub('#[^\s]+', ' ', text)  # 去除井号话题
    print("去#" + text)

    # remove URL
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    print("url" + text)

    # remove HTML Tags
    html_pattern = re.compile('<.*?>')
    text = html_pattern.sub(r'', text)
    print("html" + text)

    text = re.sub(r'\s+', ' ', text)  # 删除多余空格换行符
    print("去空格换行" + text)
    # 中英符号替换
    text = re.sub('？', '?', text)
    text = re.sub('！', '!', text)
    print("中英符号替换" + text)

    # 转小写
    text = text.lower()
    print("lower" + text)

    # 缩略词更改
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cant", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)

    mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have",
               "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
               "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
               "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
               "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
               "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
               "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
               "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
               "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
               "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
               "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
               "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
               "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
               "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
               "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "this's": "this is",
               "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
               "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
               "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
               "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
               "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
               "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
               "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
               "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
               "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
               "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
               "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
               "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
               "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
               "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
               "you're": "you are", "you've": "you have"}
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    print("改缩略" + text)

    # 符号替换
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"%", " percent ", text)
    print("语义符号替换" + text)

    #     correct spelling
    mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                    'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                    'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                    'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                    'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                    'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                    'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
                    'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',
                    'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017',
                    '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                    "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                    'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
    text = ' '.join([mispell_dict[t] if t in mispell_dict else t for t in text.split(" ")])
    print("拼写改正" + text)

    # Chat Words Conversion
    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    text = " ".join(new_text)
    print("cw" + text)
    #

    #     # 去停用词
    #     print("去停用词")
    #     stop = set(stopwords.words('english'))
    #     words=text.split()
    #     tmp=""
    #     for w in words :
    #        if not w in stop:
    #             tmp+=w+" "
    #     text=tmp

    #     # stemming
    #     stemmer = PorterStemmer()
    #     text = " ".join([stemmer.stem(word) for word in text.split()])
    #     print("stem" + text)

    #     # Lemmatization
    #     lemmatizer = WordNetLemmatizer()
    #     text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    #     print("Lemmatization" + text)

    #     """remove the frequent words"""
    #     cnt = Counter()
    #     FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
    #     text = " ".join([word for word in str(text).split() if word not in FREQWORDS])
    #     print("freqent" + text)

    #     """remove the rare words"""
    #     n_rare_words = 10
    #     RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words - 1:-1]])
    #     text = " ".join([word for word in str(text).split() if word not in RAREWORDS])
    #     print("rare" + text)

    #     #删除长度小于3的单词
    #     new_text = []
    #     for w in text.split():
    #         if len(w)>=3:
    #             new_text.append(w+" ")
    #     text = " ".join(new_text)
    #     print("delete shortwords" + text)

    # 删除特殊符号
    pattern = r'[^a-zA-z0-9.?!\s]'
    text = re.sub(pattern, '', text)
    text = re.sub('[_]', ' ', text)
    print("specialchara" + text)

    # 去除多余空格
    text = ' '.join(text.split())
    print("去空格" + text)

    print("文本清理结果：" + text)

    return text

process = lambda x: clean_text(x)
test['text'] = test['text'].apply(process)


# 将输入数据处理为list格式
new = pd.DataFrame(columns=['text'])
new['text'] = test["text"]
# 首先将pandas读取的数据转化为array
data_array = np.array(new)
# 然后转化为list形式
data_list =data_array.tolist()


#加载预训练模型
# 设置要求进行分类的类别
sentiment_list=list(train.sentiment.unique().astype(str))  # 注意此处需要将数值类别转换为字符串
print(sentiment_list)
sentiment_map = {
    idx: sentiment_text for idx, sentiment_text in enumerate(sentiment_list)
}
test_text_len=test['text'].map(len).max()

# 加载训练好的模型
model = hub.Module(
    name="ernie_v2_eng_large",
    task='seq-cls',
    load_checkpoint='./ckpt/best_model/model.pdparams',
    num_classes=len(train.sentiment.unique()),
    label_map=sentiment_map)

# 对测试集数据进行预测
predictions = model.predict(data_list, max_seq_len=test_text_len, batch_size=3, use_gpu=False)

# 生成要提交的结果文件
sub = pd.read_csv('data/test.csv',header=0,names=['id','sentiment'])
sub['sentiment'] = predictions
sub.to_csv('data/submission.csv',index=False)
print("done！")