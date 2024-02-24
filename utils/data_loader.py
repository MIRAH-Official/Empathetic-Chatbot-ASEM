
import torch
import torch.utils.data as data
import logging 
from utils import config
import pprint
pp = pprint.PrettyPrinter(indent=1)
from model.common_layer import write_config
from utils.data_reader import load_dataset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data 

        #ED
        self.sent_map = {'Positive': 0, 'Negative': 1}
        #DD
        # self.sent_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}

        # Using ED dataset
        self.emo_map = {'Anger': 0, 'Fear': 1, 'Sadness': 2,'Remorse': 3, 'Surprise': 4, 'Disgust': 5,'Joy': 6,'Anticipation': 7, 'Love': 8, 'Trust': 9}
        # Using DD dataset
        # self.emo_map = {'NoEmo': 0, 'Anger': 1, 'Disgust': 2,'Fear': 3, 'Happiness': 4, 'Sadness': 5,'Surprise': 6}
        

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["history_text"] = self.data["history"][index]
        item["target_text"] = self.data["target"][index]
        item["current_turn_text"] = self.data["current_turn"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["context"], item["context_mask"], item["last_turn"], item["last_turn_mask"] = self.preprocess(item["context_text"])
        item["history"], item["history_mask"], item["last_turn2"], item["last_turn_mask2"] = self.preprocess(item["history_text"])
        item["all_turns"], item["all_context_mask"], item["num_turns"] = self.preprocess2(item["context_text"])
        item["target"], _ = self.preprocess(item["target_text"], anw=True)
        item["current_turn"],item["current_turn_mask"] = self.preprocess(item["current_turn_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)
        item["stance_text"] = self.data["stance"][index]
        item["stance"], item["stance_label"] = self.preprocess_stn(item["stance_text"], self.sent_map)

        return item


    def preprocess(self, arr, anw=False):
        if(anw):
            X_mask2 = []
            # convert answer to a sequence of word ids
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_Rob_idx for word in arr] + [config.EOS_Rob_idx]
            spk2 = self.vocab.word2index["USR"] 
            X_mask2 += [spk2 for _ in range(len(sequence))] 
            return torch.LongTensor(sequence), torch.LongTensor(X_mask2)
        else:
            # separate last turn input from context input
            last_turn_input = arr[-1]
            # convert context input to a sequence of word ids
            X_dial = [config.CLS_idx]
            X_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_Rob_idx for word in sentence] #+ [config.SPE_idx]

                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                X_mask += [spk for _ in range(len(sentence))] #+ [config.SPE_idx]
                # Add separator token after each turn
                if i < len(arr) - 1:
                    X_dial += [config.SPE_idx]
                    X_mask += [config.SPE_idx]

            # convert last turn input to a sequence of word ids
            X_last_turn = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_Rob_idx for word in last_turn_input]
            X_last_turn_mask = [self.vocab.word2index["USR"] for _ in range(len(X_last_turn))] 

            assert len(X_dial) == len(X_mask)
            return torch.LongTensor(X_dial), torch.LongTensor(X_mask), torch.LongTensor(X_last_turn), torch.LongTensor(X_last_turn_mask)

    #ALL TURNS
    def preprocess2(self, arr, anw=False):
        
        num_turns = 0 
        """Converts words to ids."""
        if(anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_Rob_idx for word in arr] + [config.EOS_Rob_idx]
            return torch.LongTensor(sequence)
        else:
            X_dial = [config.CLS_idx]
            X_mask = [config.CLS_idx]

            for i, sentence in enumerate(arr):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_Rob_idx for word in sentence]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                X_mask += [spk for _ in range(len(sentence))] 
                num_turns += 1
            assert len(X_dial) == len(X_mask)

            return torch.LongTensor(X_dial), torch.LongTensor(X_mask), num_turns

    def preprocess_emo(self, emotion, emo_map):
        program = [0]*len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]

    def preprocess_stn(self, stance, sent_map):
        program_s = [0]*len(sent_map)
        program_s[sent_map[stance]] = 1
        return program_s, sent_map[stance]

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long() ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 



    data.sort(key=lambda x: len(x["context"]), reverse=True) ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    
    ## input
    input_batch, input_lengths     = merge(item_info['context'])

    history_batch, history_lengths     = merge(item_info['history'])

    all_input_batch, all_input_lengths     = merge(item_info['all_turns'])
    all_mask_input, mask_input_lengths = merge(item_info['all_context_mask'])
    mask_input, _ = merge(item_info['context_mask'])
    mask_input_hist, _ = merge(item_info['history_mask'])

    mask_input_current_tur,_ = merge(item_info['current_turn_mask'])

    last_turn, _ = merge(item_info["last_turn"])
    last_turn_mask, _ = merge(item_info["last_turn_mask"])
    ## Target
    target_batch, target_lengths   = merge(item_info['target'])

    current_turn_batch, current_turn_lengths   = merge(item_info['current_turn'])

    if config.USE_CUDA:
        input_batch = input_batch.cuda()
        mask_input = mask_input.cuda()
        mask_input_hist = mask_input_hist.cuda()
        mask_input_current_tur = mask_input_current_tur.cuda()
        target_batch = target_batch.cuda()
        last_turn = last_turn.cuda()
        last_turn_mask = last_turn_mask.cuda()
        all_input_batch = all_input_batch.cuda()
        all_mask_input = all_mask_input.cuda()

        history_batch = history_batch.cuda()
        current_turn_batch = current_turn_batch.cuda()

 
    d = {}
    d["input_batch"] = input_batch
    d["history_batch"] = history_batch
    d["current_turn_batch"] = current_turn_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    d["mask_input_hist"] = mask_input_hist
    d["mask_input_current_tur"] = mask_input_current_tur
    d["last_turn"] = last_turn
    d["last_turn_mask"] = last_turn_mask
    d["all_input_batch"] = all_input_batch
    d["all_mask_input"] = all_mask_input
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    ##program
    d["target_program"] = item_info['emotion']
    d["program_label"] = item_info['emotion_label']
    d["num_turns"] = item_info['num_turns']
    
    # program (stance)
    d["stance_program"] = item_info['stance']
    d["program_stance"] = item_info['stance_label']
    ##text
    d["input_txt"] = item_info['context_text']
    d["target_txt"] = item_info['target_text']
    d["program_txt"] = item_info['emotion_text']
    d["program_stn_txt"] = item_info['stance_text']
    return d 


def prepare_data_seq(batch_size=32):  
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    logging.info("Vocab  {} ".format(vocab.n_words))
    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)
                                                #  
                                                #  generator=torch.Generator().manual_seed(seed)) # drop_last=True,



    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)

                                                #  generator=torch.Generator().manual_seed(seed)) #, drop_last=True

    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                 batch_size=1,
                                                 shuffle=False, collate_fn=collate_fn)

    write_config()

    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map), len(dataset_train.sent_map)