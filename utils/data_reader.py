# Empathetic Dialogue dataset (ED)
import torch
import torch.utils.data as data
import os
from utils import config
import pickle
import numpy as np
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=1)

import nltk
nltk.download('punkt')
from transformers import  RobertaTokenizer
import matplotlib.pyplot as plt
from matplotlib import cm

special_tokens_context = ["<unk>"]
tokenizer_rob_context = RobertaTokenizer.from_pretrained("roberta-base", add_special_tokens=special_tokens_context)

special_tokens_all = ["<unk>"]
tokenizer_rob_all = RobertaTokenizer.from_pretrained("roberta-base", add_special_tokens=special_tokens_all)
tokenizer= RobertaTokenizer.from_pretrained("roberta-base")

class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word 
        self.n_words = len(init_index2word)  # Count default tokens
      
    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())
    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def get_vocab_keys(self):
        return list(self.word2index.keys())
    def get_vocab_values(self):
        return list(self.word2index.values())


def clean(sentence, word_pairs,  vocab, tokenizer):
    
    for k, v in word_pairs.items():
        sentence = sentence.replace(k,v)
    sentence_rob = tokenizer_rob_all.tokenize(sentence)
    
    tokens = {'input_ids': []}
    tokens2 = {'attention_mask': []}

    # Use the appropriate tokenizer based on the input
    # in case we want to consider specal tokens in the trarget response but not in the context
    if tokenizer == "context":
        text_tok = tokenizer_rob_context.encode_plus(sentence, add_special_tokens=False)

    else:
        text_tok = tokenizer_rob_all.encode_plus(sentence , add_special_tokens=False)

    tokens['input_ids'].append(text_tok['input_ids']) ##[0]
    tokens2['attention_mask'].append(text_tok['attention_mask']) ##[0]

    sentence = sentence.lower()
    sentence_nltk = nltk.word_tokenize(sentence)

    # Pad sequences
    # Step 1
    max_len = max(len(tokens['input_ids'][0]), len(sentence_nltk))

    # Step 2
    num_ones = max_len - len(tokens['input_ids'][0])
    ones = [1] * num_ones
    zeros = [0] * num_ones
    # Step 3

    tokens['input_ids'][0].extend(ones)
    tokens2['attention_mask'][0].extend(zeros)

    num_ones_to_add = max_len - len(sentence_nltk)
    ones_nltk = ["unk"] * num_ones_to_add
    nltk_token = sentence_nltk + ones_nltk

    return sentence_nltk, tokens, tokens2, sentence_rob  #sentence_nltk, tokens, tokens2, sentence_rob


def read_langs(vocab):

    word_pairs = {"it's":"it is", "don't":"do not", "doesn't":"does not", "didn't":"did not", "you'd":"you would", "you're":"you are", "you'll":"you will", "i'm":"i am", "they're":"they are", "that's":"that is", "what's":"what is", "couldn't":"could not", "i've":"i have", "we've":"we have", "can't":"cannot", "i'd":"i would", "i'd":"i would", "aren't":"are not", "isn't":"is not", "wasn't":"was not", "weren't":"were not", "won't":"will not", "there's":"there is", "there're":"there are"}

    emo_Plutchik = {'angry': 'Anger','annoyed': 'Anger','furious': 'Anger','jealous': 'Anger', 
                'afraid': 'Fear', 'terrified': 'Fear', 'anxious': 'Fear', 'apprehensive': 'Fear','embarrassed': 'Fear', 
                'sad': 'Sadness', 'lonely': 'Sadness','devastated': 'Sadness', 'disappointed': 'Sadness', 
                'guilty': 'Remorse','ashamed': 'Remorse',
                'surprised': 'Surprise', 'impressed': 'Surprise',
                'disgusted': 'Disgust',
                'excited' : 'Joy','grateful': 'Joy', 'joyful': 'Joy', 'content': 'Joy','confident': 'Joy', 
                'anticipating': 'Anticipation',  'hopeful': 'Anticipation','prepared': 'Anticipation',
                'nostalgic': 'Love','caring': 'Love','sentimental': 'Love',
                'trusting': 'Trust', 'faithful': 'Trust','proud': 'Trust'
    }
    sentim_map = {'Trust': 'Positive','Love': 'Positive','Anticipation': 'Positive','Joy': 'Positive','Surprise': 'Positive',
                'Disgust': 'Negative', 'Remorse': 'Negative', 'Sadness': 'Negative', 'Fear': 'Negative','Anger':  'Negative'}
    
    ## Train set
    train_context = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_dialog_texts.train.npy',allow_pickle=True)
    train_target = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_target_texts.train.npy',allow_pickle=True)
    train_emotion = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_emotion_texts.train.npy',allow_pickle=True)
    train_situation = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_situation_texts.train.npy',allow_pickle=True)
    train_current_turn = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_current_turn_texts.train.npy',allow_pickle=True)
    train_history = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_previous_turns_texts.train.npy',allow_pickle=True)

    df_train = pd.DataFrame({'context': train_context.flatten(), 'response': train_target.flatten(), 'current_turn': train_current_turn.flatten(),'train_history':train_history.flatten() , 'emotion': train_emotion.flatten(), 'situation': train_situation.flatten()})

    pd.options.display.max_colwidth = 800

    print('df_train: ', df_train['context'].iloc[0:25])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        print(df_train.iloc[0:7])
    print('df_train', df_train.shape)
    print(df_train.iloc[0:7])

    df_train['emo_Plutchik'] = [emo_Plutchik[emotion] for emotion in df_train.emotion]
    
    df_train['stance'] = [sentim_map[emotion] for emotion in df_train.emo_Plutchik]
    print("Train emo_Plutchik emotions\n",df_train['emo_Plutchik'].value_counts())
    print('Train stance', df_train['stance'].value_counts())
    my_colors = cm.inferno_r(np.linspace(.4, .8, 30))
    plt.figure(figsize=(12,5))
    df_train['emo_Plutchik'].value_counts().plot(kind="bar", stacked=True, color=my_colors)
    
    plt.ylabel('Count')
    plt.savefig('emo_Plutchik.png')

    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_target_texts.train2.npy', df_train.response.values.tolist())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_emotion_texts.train2.npy', df_train.emo_Plutchik.values.tolist())  
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_stance_texts.train2.npy', df_train.stance.values.tolist())  
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_dialog_texts.train2.npy', df_train.context.values.reshape(df_train.shape[0], 1).tolist())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_previous_turns_texts.train2.npy', df_train.train_history.values.reshape(df_train.shape[0], 1).tolist())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_current_turn_texts.train2.npy', df_train.current_turn.values.tolist())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_situation_texts.train2.npy', df_train.situation.values.tolist())

    # Upload train files
    train_context = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_dialog_texts.train2.npy',allow_pickle=True)
    train_target = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_target_texts.train2.npy',allow_pickle=True)
    train_emotion = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_emotion_texts.train2.npy',allow_pickle=True)
    train_situation = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_situation_texts.train2.npy',allow_pickle=True)
    train_stance = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_stance_texts.train2.npy',allow_pickle=True)
    train_current_turn = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_current_turn_texts.train2.npy',allow_pickle=True)
    train_history = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_previous_turns_texts.train2.npy',allow_pickle=True)


    ##Development set
    dev_context = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_dialog_texts.dev.npy',allow_pickle=True)
    dev_target = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_target_texts.dev.npy',allow_pickle=True)
    dev_emotion = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_emotion_texts.dev.npy',allow_pickle=True)
    dev_situation = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_situation_texts.dev.npy',allow_pickle=True)
    dev_current_turn = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_current_turn_texts.dev.npy',allow_pickle=True)
    dev_history = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_previous_turns_texts.dev.npy',allow_pickle=True)

    df_dev = pd.DataFrame({'context': dev_context.flatten(), 'response': dev_target.flatten(), 'current_turn': dev_current_turn.flatten(),'dev_history':dev_history.flatten() , 'emotion': dev_emotion.flatten(), 'situation': dev_situation.flatten()})

    print('df_dev', df_dev.shape)

    df_dev['emo_Plutchik'] = [emo_Plutchik[emotion] for emotion in df_dev.emotion]
    
    df_dev['stance'] = [sentim_map[emotion] for emotion in df_dev.emo_Plutchik]
    print("Dev emo_Plutchik emotions\n",df_dev['emo_Plutchik'].value_counts())
    print('Dev stance', df_dev['stance'].value_counts())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_target_texts.dev2.npy', df_dev.response.values.tolist())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_situation_texts.dev2.npy', df_dev.situation.values.tolist())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_emotion_texts.dev2.npy', df_dev.emo_Plutchik.values.tolist())  
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_stance_texts.dev2.npy', df_dev.stance.values.tolist())  
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_current_turn_texts.dev2.npy', df_dev.current_turn.values.tolist())  
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_dialog_texts.dev2.npy', df_dev.context.values.reshape(df_dev.shape[0], 1).tolist())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_previous_turns_texts.dev2.npy', df_dev.dev_history.values.reshape(df_dev.shape[0], 1).tolist())


    ## Upload dev files
    dev_context = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_dialog_texts.dev2.npy',allow_pickle=True)
    dev_target = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_target_texts.dev2.npy',allow_pickle=True)
    dev_emotion = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_emotion_texts.dev2.npy',allow_pickle=True)
    dev_situation = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_situation_texts.dev2.npy',allow_pickle=True)
    dev_stance = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_stance_texts.dev2.npy',allow_pickle=True)
    dev_current_turn = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_current_turn_texts.dev2.npy',allow_pickle=True)
    dev_history = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_previous_turns_texts.dev2.npy',allow_pickle=True)


    ##Test
    test_context = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_dialog_texts.test.npy',allow_pickle=True)
    test_target = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_target_texts.test.npy',allow_pickle=True)
    test_emotion = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_emotion_texts.test.npy',allow_pickle=True)
    test_situation = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_situation_texts.test.npy',allow_pickle=True)
    test_current_turn = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_current_turn_texts.test.npy',allow_pickle=True)
    test_history = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_previous_turns_texts.test.npy',allow_pickle=True)
   
    df_test = pd.DataFrame({'context': test_context.flatten(), 'response': test_target.flatten(), 'current_turn': test_current_turn.flatten(),'test_history':test_history.flatten() , 'emotion': test_emotion.flatten(), 'situation': test_situation.flatten()})

    # new_row = pd.DataFrame([{'context': "I never expected him to gift me that beautiful necklace for our anniversary!", 'response': "Wow, that sounds like such a thoughtful gesture! It's so lovely when someone surprises us in such special ways", 'current_turn': "Wow, that sounds like such a thoughtful gesture! It's so lovely when someone surprises us in such special ways", 'test_history': "I never expected him to gift me that beautiful necklace for our anniversary!. Wow, that sounds like such a thoughtful gesture! It's so lovely when someone surprises us in such special ways", 'emotion': "surprised", 'situation': "I never expected him to gift me that beautiful necklace for our anniversary!"}])
    # # Appending old dataframe to new row
    # df_test = new_row.append(df_test, ignore_index=True)


    print('df_test', df_test.shape)

    df_test['emo_Plutchik'] = [emo_Plutchik[emotion] for emotion in df_test.emotion]

    df_test['stance'] = [sentim_map[emotion] for emotion in df_test.emo_Plutchik]
    print("Test emo_Plutchik emotions\n",df_test['emo_Plutchik'].value_counts())
    print('Test stance', df_test['stance'].value_counts())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_target_texts.test2.npy', df_test.response.values.tolist())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_situation_texts.test2.npy', df_test.situation.values.tolist())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_emotion_texts.test2.npy', df_test.emo_Plutchik.values.tolist())  
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_stance_texts.test2.npy', df_test.stance.values.tolist())  
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_dialog_texts.test2.npy', df_test.context.values.reshape(df_test.shape[0], 1).tolist())
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_current_turn_texts.test2.npy', df_test.current_turn.values.tolist())  
    np.save('empathetic-dialogue/multi-turn/ED/multi-turn/sys_previous_turns_texts.test2.npy', df_test.test_history.values.reshape(df_test.shape[0], 1).tolist())

    ## Upload test files
    test_context = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_dialog_texts.test2.npy',allow_pickle=True)
    test_target = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_target_texts.test2.npy',allow_pickle=True)
    test_emotion = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_emotion_texts.test2.npy',allow_pickle=True)
    test_situation = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_situation_texts.test2.npy',allow_pickle=True)
    test_stance = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_stance_texts.test2.npy',allow_pickle=True)
    test_current_turn = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_current_turn_texts.test2.npy',allow_pickle=True)
    test_history = np.load('empathetic-dialogue/multi-turn/ED/multi-turn/sys_previous_turns_texts.test2.npy',allow_pickle=True)

    data_train = {'context':[],'target':[],'emotion':[] ,'stance':[],'current_turn':[],'history':[]}
    data_dev = {'context':[],'target':[],'emotion':[],'stance':[],'current_turn':[],'history':[]}
    data_test = {'context':[],'target':[],'emotion':[],'stance':[],'current_turn':[],'history':[]}

  
    data_train_bert = {'context':[],'attention_mask':[],  'target':[], 'emotion':[],'stance':[],'context_text':[],'target_text':[],'attention_mask2':[]}
    data_dev_bert = {'context':[],'attention_mask':[], 'target':[],'emotion':[],'stance':[],'context_text':[],'target_text':[],'attention_mask2':[]}
    data_test_bert = {'context':[],'attention_mask':[], 'target':[],'emotion':[],'stance':[],'context_text':[],'target_text':[],'attention_mask2':[]}


    for context in train_context:

        u_list = []
        u_list_bert = []
        u_list_bert_att =[]
        rob_id1 = []
        u_rob_text = []
        conversation_str = context[0] # Assumes conversation is the first element in the array
        utterances = conversation_str.split('\n')
        for u in utterances:

            u, u_b, u_b_a, text_rob = clean(u, word_pairs, vocab, "context")
            u_list.append(u)
            u_list_bert.append(u_b) #['input_ids']
            u_list_bert_att.append(u_b_a)
            u_rob_text.append(text_rob)

            vocab.index_words(u)
        data_train['context'].append(u_list)
        data_train_bert['context'].append(u_list_bert)
        data_train_bert['attention_mask'].append(u_list_bert_att)
        data_train_bert['context_text'].append(u_rob_text)



    for context in train_history:
        u_list = []
        u_list_bert = []
        u_list_bert_att =[]
        rob_id1 = []
        u_rob_text = []
        conversation_str = context[0] # Assumes conversation is the first element in the array
        utterances = conversation_str.split('\n ')
        for u in utterances:

            u, u_b, u_b_a, text_rob = clean(u, word_pairs, vocab, "context")
            u_list.append(u)
            u_list_bert.append(u_b) #['input_ids']
            u_list_bert_att.append(u_b_a)
            u_rob_text.append(text_rob)
            vocab.index_words(u)
        data_train['history'].append(u_list)


    for target in train_current_turn:
        target, target_b, target_b_a, text_rob = clean(target, word_pairs, vocab, "all") 
        target_b_a = [target_b_a]
        data_train['current_turn'].append(target)
        vocab.index_words(target)

    for target in train_target:
        target, target_b, target_b_a, text_rob = clean(target, word_pairs, vocab, "all") 
        data_train_bert['target'].append(target_b)
        data_train_bert['target_text'].append(text_rob)
        target_b_a = [target_b_a]
        data_train_bert['attention_mask2'].append(target_b_a)
        data_train['target'].append(target)
        vocab.index_words(target)

    for emotion in train_emotion:
        data_train['emotion'].append(emotion)
        data_train_bert['emotion'].append(emotion)
    for stance in train_stance:
        data_train['stance'].append(stance)
        data_train_bert['stance'].append(stance)

    assert len(data_train['context']) == len(data_train['target']) == len(data_train['emotion']) == len(data_train['current_turn'])== len(data_train['history'])
    assert len(data_train_bert['context']) == len(data_train_bert['target']) == len(data_train_bert['emotion']) == len(data_train_bert['stance'] ) 

    for context in dev_context:
        u_list = []
        u_list_bert = []
        u_list_bert_attention = []
        u_rob_text = []
        conversation_str = context[0] # Assumes conversation is the first element in the array
        utterances = conversation_str.split('\n ')

        for u in utterances:
            u, u_b, u_b_a, text_rob= clean(u, word_pairs, vocab, "context")
            u_list.append(u)
            u_list_bert.append(u_b)
            u_list_bert_attention.append(u_b_a)
            u_rob_text.append(text_rob)
            vocab.index_words(u)

        data_dev['context'].append(u_list)
        data_dev_bert['context'].append(u_list_bert)
        data_dev_bert['attention_mask'].append(u_list_bert_attention)
        data_dev_bert['context_text'].append(u_rob_text)


    for context in dev_history:
        u_list = []
        u_list_bert = []
        u_list_bert_att =[]
        rob_id1 = []
        u_rob_text = []
        conversation_str = context[0] # Assumes conversation is the first element in the array
        utterances = conversation_str.split('\n ')
        for u in utterances:
            u, u_b, u_b_a, text_rob = clean(u, word_pairs, vocab, "context")
            u_list.append(u)
            vocab.index_words(u)
        data_dev['history'].append(u_list)


    for target in dev_current_turn:
        target, target_b, target_b_a, text_rob = clean(target, word_pairs, vocab, "all")
        data_dev['current_turn'].append(target)
        vocab.index_words(target)

    for target in dev_target:
        target, target_b, target_b_a, text_rob= clean(target, word_pairs, vocab, "all")
        data_dev_bert['target'].append(target_b) 
        data_dev_bert['target_text'].append(text_rob)
        target_b_a = [target_b_a]
        data_dev_bert['attention_mask2'].append(target_b_a)
        data_dev['target'].append(target)
        vocab.index_words(target)

    for emotion in dev_emotion:
        data_dev['emotion'].append(emotion)
        data_dev_bert['emotion'].append(emotion)
    for stance in dev_stance:
        data_dev['stance'].append(stance)
        data_dev_bert['stance'].append(stance)

    assert len(data_dev['context']) == len(data_dev['target']) == len(data_dev['emotion']) == len(data_dev['current_turn'])== len(data_dev['history'])
    assert len(data_dev_bert['context']) == len(data_dev_bert['target']) == len(data_dev_bert['emotion']) == len(data_dev_bert['stance']) 


    for context in test_context:
        u_list = []
        u_list_b = []
        u_list_b_attention = []
        u_rob_text = []
        conversation_str = context[0] # Assumes conversation is the first element in the array
        utterances = conversation_str.split('\n ')

        for u in utterances:
            u, u_b, u_b_a, text_rob = clean(u, word_pairs, vocab, "context")
            u_list.append(u)
            u_list_b.append(u_b) 
            u_list_b_attention.append(u_b_a)
            u_rob_text.append(text_rob)
            vocab.index_words(u)

        data_test['context'].append(u_list)
        data_test_bert['context'].append(u_list_b)
        data_test_bert['attention_mask'].append(u_list_b_attention)
        data_test_bert['context_text'].append(u_rob_text)

    for context in test_history:
        u_list = []
        u_list_bert = []
        u_list_bert_att =[]
        rob_id1 = []
        u_rob_text = []
        conversation_str = context[0] # Assumes conversation is the first element in the array
        utterances = conversation_str.split('\n ')
        for u in utterances:
            u, u_b, u_b_a, text_rob = clean(u, word_pairs, vocab, "context")
            u_list.append(u)
            vocab.index_words(u)

        data_test['history'].append(u_list)


    for target in test_current_turn:
        target, target_b, target_b_a, text_rob = clean(target, word_pairs, vocab, "all") 
        data_test['current_turn'].append(target)
        vocab.index_words(target)

    for target in test_target:
        target, target_b, target_b_a, text_rob = clean(target, word_pairs, vocab, "all") 
        data_test_bert['target'].append(target_b) 
        data_test_bert['target_text'].append(text_rob)
        target_b_a = [target_b_a]
        data_test_bert['attention_mask2'].append(target_b_a)
        data_test['target'].append(target)
        vocab.index_words(target)

    for emotion in test_emotion:
        data_test['emotion'].append(emotion)
        data_test_bert['emotion'].append(emotion)
    for stance in test_stance:
        data_test['stance'].append(stance)
        data_test_bert['stance'].append(stance)

    assert len(data_test['context']) == len(data_test['target']) == len(data_test['emotion']) == len(data_test['current_turn'])== len(data_test['history'])
    assert len(data_test_bert['context']) == len(data_test_bert['target']) == len(data_test_bert['emotion']) == len(data_test_bert['stance'])

    return data_train, data_dev, data_test, vocab, data_train_bert, data_dev_bert, data_test_bert, train_emotion, dev_emotion, test_emotion


def load_dataset():
    vocab_b = tokenizer_rob_all.get_vocab()

    if(os.path.exists('empathetic-dialogue/dataset_preprocED.p')):
        print("LOADING empathetic_dialogue")
        with open('empathetic-dialogue/dataset_preprocED.p', "rb") as f:
            [data_tra, data_val, data_tst, vocab, data_tra_b, data_val_b, data_tst_b, data_train_emo, data_dev_emo, data_test_emo ] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab, data_tra_b, data_val_b, data_tst_b, data_train_emo, data_dev_emo, data_test_emo   = read_langs(vocab=Lang({config.UNK_Rob_idx: "UNK", config.PAD_Rob_idx: "PAD", config.EOS_Rob_idx: "EOS", config.SOS_Rob_idx: "SOS", config.USR_idx:"USR", config.SYS_idx:"SYS", config.CLS_idx:"CLS", config.SPE_idx:"SPE"})) 
        with open('empathetic-dialogue/dataset_preprocED.p', "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab, data_tra_b, data_val_b, data_tst_b, data_train_emo, data_dev_emo, data_test_emo], f)
            print("Saved PICKLE")

    # print('SOS token Roberta:', tokenizer_rob_all.convert_ids_to_tokens(0))
    # print('EOS token Roberta:', tokenizer_rob_all.convert_ids_to_tokens(2))
    # print('PAD token Roberta:', tokenizer_rob_all.convert_ids_to_tokens(1))
    # print('UNK token Roberta:', tokenizer_rob_all.convert_ids_to_tokens(3))
    # print('mask token Roberta:', tokenizer_rob_all.convert_ids_to_tokens(4))

    print('the index of EOS in Roberta vocab:', vocab_b['</s>'])
    print('the index of SOS in Roberta vocab:', vocab_b['<s>']) 


    print('Roberta Tokenizer Number of Vocab:',len(tokenizer_rob_all))
    print('Created Vocab:',vocab.n_words)


    print('the index of EOS in vocab:', vocab.word2index['EOS'])
    print('the index of SOS in vocab:', vocab.word2index['SOS']) 
    print('the index of UNK_idx in vocab:', vocab.word2index['UNK']) 
    print('the index of USR_idx in vocab:', vocab.word2index['USR']) 
    print('the index of SYS_idx in vocab:', vocab.word2index['SYS']) 


    for i in range(3):
        # print('[situation]:', ' '.join(data_tra['situation'][i]))
        print('[emotion]:', data_tra['emotion'][i])
        print('[history]:', [' '.join(u) for u in data_tra['history'][i]])
        print('[current_turn]:', ' '.join(data_tra['current_turn'][i]))
        print('[context]:', [' '.join(u) for u in data_tra['context'][i]])
        print('[target]:', ' '.join(data_tra['target'][i]))
        print(" ")

    return data_tra, data_val, data_tst, vocab


##############################
# #Daily Dialogue dataset (DD)

# import torch
# import torch.utils.data as data
# from utils import config
# import pickle
# import numpy as np
# import pandas as pd
# import pprint
# pp = pprint.PrettyPrinter(indent=1)
# import nltk
# nltk.download('punkt')
# from transformers import  RobertaTokenizer
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import os

# special_tokens_context = ["<unk>"]
# tokenizer_rob_context = RobertaTokenizer.from_pretrained("roberta-base", add_special_tokens=special_tokens_context)

# # special_tokens_all = ["<pad>", "<unk>", "</s>"]
# special_tokens_all = ["<unk>"]
# tokenizer_rob_all = RobertaTokenizer.from_pretrained("roberta-base", add_special_tokens=special_tokens_all)

# tokenizer= RobertaTokenizer.from_pretrained("roberta-base")


# class Lang:
#     def __init__(self, init_index2word):

#         self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
#         self.word2count = {str(v): 1 for k, v in init_index2word.items()}
#         self.index2word = init_index2word 
#         self.n_words = len(init_index2word)  # Count default tokens
      
#     def index_words(self, sentence):
#         for word in sentence:
#             self.index_word(word.strip())

#     def index_word(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1

#     def get_vocab_keys(self):
#         return list(self.word2index.keys())

#     def get_vocab_values(self):
#         return list(self.word2index.values())



# def clean(sentence, word_pairs,  vocab, tokenizer):
    
#     for k, v in word_pairs.items():
#         sentence = sentence.replace(k,v)

#     sentence_rob = tokenizer_rob_all.tokenize(sentence)

#     tokens = {'input_ids': []}
#     tokens2 = {'attention_mask': []}

#     # Roberat
#     # Use the appropriate tokenizer based on the input
#     if tokenizer == "context":
#         text_tok = tokenizer_rob_context.encode_plus(sentence, add_special_tokens=False)

#     else:
#         text_tok = tokenizer_rob_all.encode_plus(sentence , add_special_tokens=False)

#     tokens['input_ids'].append(text_tok['input_ids'])
#     tokens2['attention_mask'].append(text_tok['attention_mask']) 

#     sentence = sentence.lower()
#     sentence_nltk = nltk.word_tokenize(sentence)

#     # Pad sequences
#     # Step 1
#     max_len = max(len(tokens['input_ids'][0]), len(sentence_nltk))

#     # print('tokens', tokens)
#     # Step 2
#     num_ones = max_len - len(tokens['input_ids'][0])
#     ones = [1] * num_ones
#     zeros = [0] * num_ones
#     # Step 3

#     tokens['input_ids'][0].extend(ones)
#     tokens2['attention_mask'][0].extend(zeros)

#     num_ones_to_add = max_len - len(sentence_nltk)
#     ones_nltk = ["unk"] * num_ones_to_add
#     nltk_token = sentence_nltk + ones_nltk

#     return sentence_nltk, tokens, tokens2, sentence_rob

# def read_langs(vocab):
#     word_pairs = {"it's":"it is", "don't":"do not", "doesn't":"does not", "didn't":"did not", "you'd":"you would", "you're":"you are", "you'll":"you will", "i'm":"i am", "they're":"they are", "that's":"that is", "what's":"what is", "couldn't":"could not", "i've":"i have", "we've":"we have", "can't":"cannot", "i'd":"i would", "i'd":"i would", "aren't":"are not", "isn't":"is not", "wasn't":"was not", "weren't":"were not", "won't":"will not", "there's":"there is", "there're":"there are"}

#     #DD
#     sentim_map = {'NoEmo': 'Neutral','Happiness': 'Positive','Surprise': 'Positive',
#                 'Disgust': 'Negative', 'Sadness': 'Negative', 'Fear': 'Negative','Anger': 'Negative'}
    
#     # Train set
#     train_context = np.load('empathetic-dialogue/multi-turn/DD/sys_dialog_texts.train.npy',allow_pickle=True)
#     train_target = np.load('empathetic-dialogue/multi-turn/DD/sys_target_texts.train.npy',allow_pickle=True)
#     train_emotion = np.load('empathetic-dialogue/multi-turn/DD/sys_emotion_texts.train.npy',allow_pickle=True)
#     train_current_turn = np.load('empathetic-dialogue/multi-turn/DD/sys_current_turn_texts.train.npy',allow_pickle=True)
#     train_history = np.load('empathetic-dialogue/multi-turn/DD/sys_previous_turns_texts.train.npy',allow_pickle=True)

#     df_train = pd.DataFrame({'context': train_context.flatten(), 'response': train_target.flatten(), 'current_turn': train_current_turn.flatten(),'train_history':train_history.flatten() , 'emo_Plutchik': train_emotion.flatten()})

#     pd.options.display.max_colwidth = 800
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#         print(df_train.iloc[0:7])
    
#     print('df_train', df_train.shape)
#     print(df_train.iloc[0:7])

#     df_train['stance'] = [sentim_map[emotion] for emotion in df_train.emo_Plutchik]

#     print("Train emo_Plutchik emotions\n",df_train['emo_Plutchik'].value_counts())
#     print('Train stance', df_train['stance'].value_counts())

#     my_colors = cm.inferno_r(np.linspace(.4, .8, 30))
#     plt.figure(figsize=(12,5))
#     df_train['emo_Plutchik'].value_counts().plot(kind="bar", stacked=True, color=my_colors)
    
#     plt.ylabel('Count')
#     plt.savefig('emo_DD.png')

#     np.save('empathetic-dialogue/multi-turn/DD/sys_target_texts.train2.npy', df_train.response.values.tolist())
#     np.save('empathetic-dialogue/multi-turn/DD/sys_emotion_texts.train2.npy', df_train.emo_Plutchik.values.tolist())  
#     np.save('empathetic-dialogue/multi-turn/DD/sys_stance_texts.train2.npy', df_train.stance.values.tolist())  
#     np.save('empathetic-dialogue/multi-turn/DD/sys_dialog_texts.train2.npy', df_train.context.values.reshape(df_train.shape[0], 1).tolist())
#     np.save('empathetic-dialogue/multi-turn/DD/sys_previous_turns_texts.train2.npy', df_train.train_history.values.reshape(df_train.shape[0], 1).tolist())
#     np.save('empathetic-dialogue/multi-turn/DD/sys_current_turn_texts.train2.npy', df_train.current_turn.values.tolist())  


#     train_context = np.load('empathetic-dialogue/multi-turn/DD/sys_dialog_texts.train2.npy',allow_pickle=True)
#     train_target = np.load('empathetic-dialogue/multi-turn/DD/sys_target_texts.train2.npy',allow_pickle=True)
#     train_emotion = np.load('empathetic-dialogue/multi-turn/DD/sys_emotion_texts.train2.npy',allow_pickle=True)
#     # train_situation = np.load('empathetic-dialogue/sys_situation_texts.train.npy',allow_pickle=True)
#     train_stance = np.load('empathetic-dialogue/multi-turn/DD/sys_stance_texts.train2.npy',allow_pickle=True)
#     train_current_turn = np.load('empathetic-dialogue/multi-turn/DD/sys_current_turn_texts.train2.npy',allow_pickle=True)
#     train_history = np.load('empathetic-dialogue/multi-turn/DD/sys_previous_turns_texts.train2.npy',allow_pickle=True)

#     ##Development set
#     dev_context = np.load('empathetic-dialogue/multi-turn/DD/sys_dialog_texts.dev.npy',allow_pickle=True)
#     dev_target = np.load('empathetic-dialogue/multi-turn/DD/sys_target_texts.dev.npy',allow_pickle=True)
#     dev_emotion = np.load('empathetic-dialogue/multi-turn/DD/sys_emotion_texts.dev.npy',allow_pickle=True)

#     dev_current_turn = np.load('empathetic-dialogue/multi-turn/DD/sys_current_turn_texts.dev.npy',allow_pickle=True)
#     dev_history = np.load('empathetic-dialogue/multi-turn/DD/sys_previous_turns_texts.dev.npy',allow_pickle=True)


#     df_dev = pd.DataFrame({'context': dev_context.flatten(), 'response': dev_target.flatten(),'current_turn': dev_current_turn.flatten(),'history':dev_history.flatten() , 'emo_Plutchik': dev_emotion.flatten()})

#     print('df_dev', df_dev.shape)
    
#     df_dev['stance'] = [sentim_map[emotion] for emotion in df_dev.emo_Plutchik]

#     print("Dev emo_Plutchik emotions\n",df_dev['emo_Plutchik'].value_counts())
#     print('Dev stance', df_dev['stance'].value_counts())

#     np.save('empathetic-dialogue/multi-turn/DD/sys_target_texts.dev2.npy', df_dev.response.values.tolist())
#     np.save('empathetic-dialogue/multi-turn/DD/sys_emotion_texts.dev2.npy', df_dev.emo_Plutchik.values.tolist())  
#     np.save('empathetic-dialogue/multi-turn/DD/sys_stance_texts.dev2.npy', df_dev.stance.values.tolist())  
#     np.save('empathetic-dialogue/multi-turn/DD/sys_dialog_texts.dev2.npy', df_dev.context.values.reshape(df_dev.shape[0], 1).tolist())
#     np.save('empathetic-dialogue/multi-turn/DD/sys_previous_turns_texts.dev2.npy', df_dev.history.values.reshape(df_dev.shape[0], 1).tolist())
#     np.save('empathetic-dialogue/multi-turn/DD/sys_current_turn_texts.dev2.npy', df_dev.current_turn.values.tolist())  


#     dev_context = np.load('empathetic-dialogue/multi-turn/DD/sys_dialog_texts.dev2.npy',allow_pickle=True)
#     dev_target = np.load('empathetic-dialogue/multi-turn/DD/sys_target_texts.dev2.npy',allow_pickle=True)
#     dev_emotion = np.load('empathetic-dialogue/multi-turn/DD/sys_emotion_texts.dev2.npy',allow_pickle=True)
#     # dev_situation = np.load('empathetic-dialogue/sys_situation_texts.dev.npy',allow_pickle=True)
#     dev_stance = np.load('empathetic-dialogue/multi-turn/DD/sys_stance_texts.dev2.npy',allow_pickle=True)
#     dev_current_turn = np.load('empathetic-dialogue/multi-turn/DD/sys_current_turn_texts.dev2.npy',allow_pickle=True)
#     dev_history = np.load('empathetic-dialogue/multi-turn/DD/sys_previous_turns_texts.dev2.npy',allow_pickle=True)


#     ##Test set
#     test_context = np.load('empathetic-dialogue/multi-turn/DD/sys_dialog_texts.test.npy',allow_pickle=True)
#     test_target = np.load('empathetic-dialogue/multi-turn/DD/sys_target_texts.test.npy',allow_pickle=True)
#     test_emotion = np.load('empathetic-dialogue/multi-turn/DD/sys_emotion_texts.test.npy',allow_pickle=True)
#     test_current_turn = np.load('empathetic-dialogue/multi-turn/DD/sys_current_turn_texts.test.npy',allow_pickle=True)
#     test_history = np.load('empathetic-dialogue/multi-turn/DD/sys_previous_turns_texts.test.npy',allow_pickle=True)
#     df_test = pd.DataFrame({'context': test_context.flatten(), 'response': test_target.flatten(),'current_turn': test_current_turn.flatten(),'history':test_history.flatten() , 'emo_Plutchik': test_emotion.flatten()})

#     print('df_test', df_test.shape)

#     df_test['stance'] = [sentim_map[emotion] for emotion in df_test.emo_Plutchik]
#     print("Test emo_Plutchik emotions\n",df_test['emo_Plutchik'].value_counts())
#     print('Test stance', df_test['stance'].value_counts())
#     np.save('empathetic-dialogue/multi-turn/DD/sys_target_texts.test2.npy', df_test.response.values.tolist())
#     np.save('empathetic-dialogue/multi-turn/DD/sys_emotion_texts.test2.npy', df_test.emo_Plutchik.values.tolist())  
#     np.save('empathetic-dialogue/multi-turn/DD/sys_stance_texts.test2.npy', df_test.stance.values.tolist())  
#     np.save('empathetic-dialogue/multi-turn/DD/sys_dialog_texts.test2.npy', df_test.context.values.reshape(df_test.shape[0], 1).tolist())
#     np.save('empathetic-dialogue/multi-turn/DD/sys_current_turn_texts.test2.npy', df_test.current_turn.values.tolist())  
#     np.save('empathetic-dialogue/multi-turn/DD/sys_previous_turns_texts.test2.npy', df_test.history.values.reshape(df_test.shape[0], 1).tolist())



#     test_context = np.load('empathetic-dialogue/multi-turn/DD/sys_dialog_texts.test2.npy',allow_pickle=True)
#     test_target = np.load('empathetic-dialogue/multi-turn/DD/sys_target_texts.test2.npy',allow_pickle=True)
#     test_emotion = np.load('empathetic-dialogue/multi-turn/DD/sys_emotion_texts.test2.npy',allow_pickle=True)
#     # test_situation = np.load('empathetic-dialogue/sys_situation_texts.test.npy',allow_pickle=True)
#     test_stance = np.load('empathetic-dialogue/multi-turn/DD/sys_stance_texts.test2.npy',allow_pickle=True)
#     test_current_turn = np.load('empathetic-dialogue/multi-turn/DD/sys_current_turn_texts.test2.npy',allow_pickle=True)
#     test_history = np.load('empathetic-dialogue/multi-turn/DD/sys_previous_turns_texts.test2.npy',allow_pickle=True)


#     data_train = {'context':[],'target':[],'emotion':[] ,'stance':[],'current_turn':[],'history':[]}
#     data_dev = {'context':[],'target':[],'emotion':[],'stance':[],'current_turn':[],'history':[]}
#     data_test = {'context':[],'target':[],'emotion':[],'stance':[],'current_turn':[],'history':[]}

#     data_train_bert = {'context':[],'attention_mask':[],  'target':[], 'emotion':[],'stance':[],'context_text':[],'target_text':[],'attention_mask2':[]}
#     data_dev_bert = {'context':[],'attention_mask':[], 'target':[],'emotion':[],'stance':[],'context_text':[],'target_text':[],'attention_mask2':[]}
#     data_test_bert = {'context':[],'attention_mask':[], 'target':[],'emotion':[],'stance':[],'context_text':[],'target_text':[],'attention_mask2':[]}


#     for context in train_context:
        
#         u_list = []
#         u_list_bert = []
#         u_list_bert_att =[]
#         rob_id1 = []
#         u_rob_text = []
#         conversation_str = context[0] # Assumes conversation is the first element in the array
#         utterances = conversation_str.split('\n')
#         for u in utterances:
#             u, u_b, u_b_a, text_rob = clean(u, word_pairs, vocab, "context")
#             u_list.append(u)
#             u_list_bert.append(u_b)
#             u_list_bert_att.append(u_b_a)
#             u_rob_text.append(text_rob)
#             vocab.index_words(u)

#         data_train['context'].append(u_list)
#         data_train_bert['context'].append(u_list_bert)
#         data_train_bert['attention_mask'].append(u_list_bert_att)
#         data_train_bert['context_text'].append(u_rob_text)


#     for context in train_history:
#         u_list = []
#         u_list_bert = []
#         u_list_bert_att =[]
#         rob_id1 = []
#         u_rob_text = []
#         conversation_str = context[0] # Assumes conversation is the first element in the array
#         utterances = conversation_str.split('\n ')
#         for u in utterances:
#             u, u_b, u_b_a, text_rob = clean(u, word_pairs, vocab, "context")
#             u_list.append(u)
#             u_list_bert.append(u_b) #['input_ids']
#             u_list_bert_att.append(u_b_a)
#             u_rob_text.append(text_rob)
#             vocab.index_words(u)

#         data_train['history'].append(u_list)


#     for target in train_current_turn:
#         target, target_b, target_b_a, text_rob = clean(target, word_pairs, vocab, "all")

#         target_b_a = [target_b_a]

#         data_train['current_turn'].append(target)
#         vocab.index_words(target)

#     for target in train_target:
#         target, target_b, target_b_a, text_rob = clean(target, word_pairs, vocab, "all") 
#         data_train_bert['target'].append(target_b)
#         data_train_bert['target_text'].append(text_rob) 
        
#         target_b_a = [target_b_a]
#         data_train_bert['attention_mask2'].append(target_b_a)

#         data_train['target'].append(target)
#         vocab.index_words(target)

#     for emotion in train_emotion:
#         data_train['emotion'].append(emotion)
#         data_train_bert['emotion'].append(emotion)
#     for stance in train_stance:
#         data_train['stance'].append(stance)
#         data_train_bert['stance'].append(stance)


#     assert len(data_train['context']) == len(data_train['target']) == len(data_train['emotion']) == len(data_train['current_turn'])== len(data_train['history'])
#     assert len(data_train_bert['context']) == len(data_train_bert['target']) == len(data_train_bert['emotion']) == len(data_train_bert['stance'] )

#     for context in dev_context:
#         u_list = []
#         u_list_bert = []
#         u_list_bert_attention = []
#         u_rob_text = []
#         conversation_str = context[0] # Assumes conversation is the first element in the array
#         utterances = conversation_str.split('\n ')

#         for u in utterances:
#             u, u_b, u_b_a, text_rob= clean(u, word_pairs, vocab, "context")
#             u_list.append(u)
#             u_list_bert.append(u_b) #['input_ids']
#             u_list_bert_attention.append(u_b_a)
#             u_rob_text.append(text_rob)
#             vocab.index_words(u)

#         data_dev['context'].append(u_list)
#         data_dev_bert['context'].append(u_list_bert)
#         data_dev_bert['attention_mask'].append(u_list_bert_attention)
#         data_dev_bert['context_text'].append(u_rob_text)

#     for context in dev_history:
#         u_list = []
#         u_list_bert = []
#         u_list_bert_att =[]
#         rob_id1 = []
#         u_rob_text = []
#         conversation_str = context[0] # Assumes conversation is the first element in the array
#         utterances = conversation_str.split('\n ')
#         for u in utterances:
#             u, u_b, u_b_a, text_rob = clean(u, word_pairs, vocab, "context")
#             u_list.append(u)
#             vocab.index_words(u)

#         data_dev['history'].append(u_list)


#     for target in dev_current_turn:
#         target, target_b, target_b_a, text_rob = clean(target, word_pairs, vocab, "all")
#         data_dev['current_turn'].append(target)
#         vocab.index_words(target)

#     for target in dev_target:
#         target, target_b, target_b_a, text_rob= clean(target, word_pairs, vocab, "all")
#         data_dev_bert['target'].append(target_b) 
#         data_dev_bert['target_text'].append(text_rob)
#         target_b_a = [target_b_a]
#         data_dev_bert['attention_mask2'].append(target_b_a)

#         data_dev['target'].append(target)
#         vocab.index_words(target)

#     for emotion in dev_emotion:
#         data_dev['emotion'].append(emotion)
#         data_dev_bert['emotion'].append(emotion)
#     for stance in dev_stance:
#         data_dev['stance'].append(stance)
#         data_dev_bert['stance'].append(stance)

#     assert len(data_dev['context']) == len(data_dev['target']) == len(data_dev['emotion']) == len(data_dev['current_turn'])== len(data_dev['history'])
#     assert len(data_dev_bert['context']) == len(data_dev_bert['target']) == len(data_dev_bert['emotion']) == len(data_dev_bert['stance'])


#     for context in test_context:
#         u_list = []
#         u_list_b = []
#         u_list_b_attention = []
#         u_rob_text = []
#         conversation_str = context[0] # Assumes conversation is the first element in the array
#         utterances = conversation_str.split('\n ')

#         for u in utterances:
#             u, u_b, u_b_a, text_rob = clean(u, word_pairs, vocab, "context")
#             u_list.append(u)
#             u_list_b.append(u_b)
#             u_list_b_attention.append(u_b_a)
#             u_rob_text.append(text_rob)
#             vocab.index_words(u)

#         data_test['context'].append(u_list)
#         data_test_bert['context'].append(u_list_b)
#         data_test_bert['attention_mask'].append(u_list_b_attention)
#         data_test_bert['context_text'].append(u_rob_text)

#     for context in test_history:
#         u_list = []
#         u_list_bert = []
#         u_list_bert_att =[]
#         rob_id1 = []
#         u_rob_text = []
#         conversation_str = context[0] # Assumes conversation is the first element in the array
#         utterances = conversation_str.split('\n ')
#         for u in utterances:
#             u, u_b, u_b_a, text_rob = clean(u, word_pairs, vocab, "context")
#             u_list.append(u)
#             vocab.index_words(u)
#         data_test['history'].append(u_list)


#     for target in test_current_turn:
#         target, target_b, target_b_a, text_rob = clean(target, word_pairs, vocab, "all") 
#         data_test['current_turn'].append(target)
#         vocab.index_words(target)


#     for target in test_target:
#         target, target_b, target_b_a, text_rob = clean(target, word_pairs, vocab, "all")
#         data_test_bert['target'].append(target_b) 
#         data_test_bert['target_text'].append(text_rob)
#         target_b_a = [target_b_a]
#         data_test_bert['attention_mask2'].append(target_b_a)
#         data_test['target'].append(target)
#         vocab.index_words(target)

#     for emotion in test_emotion:
#         data_test['emotion'].append(emotion)
#         data_test_bert['emotion'].append(emotion)
#     for stance in test_stance:
#         data_test['stance'].append(stance)
#         data_test_bert['stance'].append(stance)



#     assert len(data_test['context']) == len(data_test['target']) == len(data_test['emotion']) == len(data_test['current_turn'])== len(data_test['history'])
#     assert len(data_test_bert['context']) == len(data_test_bert['target']) == len(data_test_bert['emotion']) == len(data_test_bert['stance'])

#     # return data_train, data_dev, data_test, vocab
#     return data_train, data_dev, data_test, vocab, data_train_bert, data_dev_bert, data_test_bert, train_emotion, dev_emotion, test_emotion



# def load_dataset():

#     if(os.path.exists('empathetic-dialogue/dataset_preprocDD.p')):
#         print("LOADING empathetic_dialogue")
#         with open('empathetic-dialogue/dataset_preprocDD.p', "rb") as f:
#             [data_tra, data_val, data_tst, vocab, data_tra_b, data_val_b, data_tst_b, data_train_emo, data_dev_emo, data_test_emo ] = pickle.load(f)
#     else:
#         print("Building dataset...")
#         data_tra, data_val, data_tst, vocab, data_tra_b, data_val_b, data_tst_b, data_train_emo, data_dev_emo, data_test_emo   = read_langs(vocab=Lang({config.UNK_Rob_idx: "UNK", config.PAD_Rob_idx: "PAD", config.EOS_Rob_idx: "EOS", config.SOS_Rob_idx: "SOS", config.USR_idx:"USR", config.SYS_idx:"SYS", config.CLS_idx:"CLS", config.SPE_idx:"SPE"})) 
#         with open('empathetic-dialogue/dataset_preprocDD.p', "wb") as f:
#             pickle.dump([data_tra, data_val, data_tst, vocab, data_tra_b, data_val_b, data_tst_b, data_train_emo, data_dev_emo, data_test_emo], f)
#             print("Saved PICKLE")


#     # print('SOS token Roberta:', tokenizer_rob_all.convert_ids_to_tokens(0))
#     # print('EOS token Roberta:', tokenizer_rob_all.convert_ids_to_tokens(2))
#     # print('PAD token Roberta:', tokenizer_rob_all.convert_ids_to_tokens(1))
#     # print('UNK token Roberta:', tokenizer_rob_all.convert_ids_to_tokens(3))
#     # print('mask token Roberta:', tokenizer_rob_all.convert_ids_to_tokens(4))


#     print('Roberta Tokenizer Number of Vocab:',len(tokenizer_rob_all))
#     print('Created Vocab:',vocab.n_words)
#     # assuming `concat_vocab` is an instance of the `Lang` class

#     print('the index of EOS in vocab:', vocab.word2index['EOS'])
#     print('the index of SOS in vocab:', vocab.word2index['SOS']) 
#     print('the index of UNK_idx in vocab:', vocab.word2index['UNK']) 
#     print('the index of USR_idx in vocab:', vocab.word2index['USR']) 
#     print('the index of SYS_idx in vocab:', vocab.word2index['SYS']) 



#     for i in range(3):
#         # print('[situation]:', ' '.join(data_tra['situation'][i]))
#         print('[emotion]:', data_tra['emotion'][i])
#         print('[history]:', [' '.join(u) for u in data_tra['history'][i]])
#         print('[current_turn]:', ' '.join(data_tra['current_turn'][i]))
#         print('[context]:', [' '.join(u) for u in data_tra['context'][i]])
#         print('[target]:', ' '.join(data_tra['target'][i]))
#         print(" ")

#     return data_tra, data_val, data_tst, vocab