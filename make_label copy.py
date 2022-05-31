
import json
import pandas as pd
from convert_m2_with_errortag import make_error_tag

def tokenizer_token(word_list, error_tag, error_type, tokenizer, max_len):
    type_list = ['None', 'R:PREP', 'R:VERB:TENSE', 'R:NOUN', 'R:OTHER', 'R:MORPH', 'R:VERB', 'U:ADV', 'M:PUNCT', 'M:VERB', 'R:WO', 'M:PREP', 'M:DET', 'R:VERB:FORM', 'U:PREP', 'M:PRON', 'M:VERB:TENSE', 'R:NOUN:NUM', 'U:DET', 'R:ORTH', 'UNK', 'M:CONJ', 'U:VERB:TENSE', 'U:PRON', 'R:ADV', 'R:SPELL', 'M:NOUN', 'U:NOUN', 'U:PUNCT', 'R:DET', 'R:VERB:SVA', 'R:PUNCT', 'M:NOUN:POSS', 'U:VERB', 'U:PART', 'R:CONTR', 'U:OTHER', 'M:VERB:FORM', 'R:ADJ', 'R:ADJ:FORM', 'M:OTHER', 'M:PART', 'M:ADV', 'R:PRON', 'M:CONTR', 'U:CONTR', 'R:PART', 'M:ADJ', 'U:CONJ', 'R:NOUN:INFL', 'U:VERB:FORM', 'R:NOUN:POSS', 'R:VERB:INFL', 'R:CONJ', 'U:ADJ', 'U:NOUN:POSS']

    type_to_class = {k:v  for v, k in enumerate(type_list)}
    class_to_type = {str(v):k  for v, k in enumerate(type_list)}  

    token_text = tokenizer(word_list, is_split_into_words=True)
    
    token_error_tag = []
    token_error_type = []
    for i_token in range(0, len(word_list)):
        start, end = token_text.word_to_tokens(i_token)
        # print(start, end)
        for i_word in range(start, end):
            token_error_tag.append(error_tag[i_token])

            if error_type[i_token] not in type_list: # This is used for conll14 test set who didn't use ERRANT as scorer
                token_error_type.append(type_to_class['None']) 
            else:
                token_error_type.append(type_to_class[error_type[i_token]])
    if len(token_error_tag) >  max_len:
        return token_error_tag[:max_len], token_error_type[:max_len]
    else:
        for i in range(len(token_error_tag), max_len):
            token_error_tag.append(0)
            token_error_type.append(type_to_class['None'])
        return token_error_tag[:max_len], token_error_type[:max_len]

def make_label(dataset_name, type):
    path = '/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/m2_' + type+'/'
    if dataset_name != 'wi':
        print(dataset_name)
        file_name = dataset_name + '.' + type 
        data = make_error_tag(path+file_name)
        return data
    else:
        print(dataset_name)
        if type == 'train':
            wi_list = ['A','B','C']
        else:
            wi_list = ['A', 'B', 'C', 'N'] 
        data = []
        for i in wi_list:
            file_name = dataset_name + '.'+ i + '.' + type
            data += make_error_tag(path+file_name)
        return data

    

def read_text_file(file_path):
    # text_list = []
    with open(file_path, 'r') as file :
        text_list = file.read().splitlines()
    return text_list
def collect_text_file(dataset_name, type):
    path = '/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/text_data/'
    if dataset_name == 'lang8': #lang-8 dataset
        if type == 'train':
            return {'text':read_text_file(path+'train/lang8.train.auto.bea19_src.txt'),
             'label': read_text_file('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/text_data/train/lang8.train.auto.bea19_tgt.txt')}
    elif dataset_name == 'fce':
        if type == 'train':
            return {'text':read_text_file('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/text_data/train/fce.train.gold.bea19_src.txt'),
             'label': read_text_file('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/text_data/train/fce.train.gold.bea19_tgt.txt')}
        elif type == 'valid':
            return {'text':read_text_file('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/text_data/test/fce.dev.gold.bea19_src.txt'),
             'label': read_text_file('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/text_data/test/fce.dev.gold.bea19_tgt.txt')}
        else:
            return {'text':read_text_file('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/text_data/test/fce.test.gold.bea19_src.txt'),
             'label': read_text_file('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/text_data/test/fce.test.gold.bea19_tgt.txt')}
    else: #wi_loc dataset
        file_class = ['A', 'B', 'C']
        text_file = []
        label_file = []
        if type == 'train':
            for i in file_class:
                text_file += read_text_file(path + 'train/'+ i + '.train.gold.bea19_src.txt')
                label_file += read_text_file(path + 'train/'+i + '.train.gold.bea19_tgt.txt')
            return {'text':text_file,'label': label_file}
        else:
            for i in file_class:
                text_file += read_text_file(path + 'test/'+i + '.dev.gold.bea19_src.txt')
                label_file += read_text_file(path + 'test/'+i + '.dev.gold.bea19_tgt.txt')
            return {'text':text_file,'label': label_file}
    
def read_josn_file(file_path):
    # dict_json = {'text':[], 'id':[], 'cefr':[], 'edit':[], }
    text_file = []

    for each_line in open(file_path, 'r'):
        json_file = json.loads(each_line)
        json_file = make_label_from_dict(json_file)
        text_file.append(json_file)
    return text_file

# make the origin file into Hugging Face type
def make_label_from_dict(dict_file):
    data = {'userid':'', 'id':0, 'cefr':'', 'edits':{'start':[], 'end':[], 'text':[]}, 'text':[]}
    keys = list(dict_file.keys())
    keys.remove('edits')
    for each_key in keys: #except for edits
        data[each_key] = dict_file[each_key]
    for corrects in dict_file['edits'][0][1:]:
        if len(corrects) != 0:
            for correct in corrects:
                # print(correct)
                data['edits']['start'].append(correct[0])
                data['edits']['end'].append(correct[1])
                if correct[2] is None:
                    data['edits']['text'].append("")
                else:
                    data['edits']['text'].append(correct[2])
    # print(data)
    return data
     

def make_interval(text, edit_start, edit_end, tokenizer):
    interval = []
    if edit_start[0] != 0:
        interval.append((0, edit_start[0], 'C')) #correct
    for i in range(0, len(edit_start)-1):
        interval.append((edit_start[i], edit_end[i], 'R')) 
        interval.append((edit_end[i], edit_start[i+1], 'C'))
    interval.append((edit_start[-1],edit_end[-1], 'R'))
    interval.append((edit_end[-1], len(text), 'C'))
    return interval

map_class_to_num = {'C':0, 'R':1}

def classify_label(text, edit_start, edit_end, tokenizer, max_len):
    if edit_start != []:
        interval = make_interval(text, edit_start, edit_end, tokenizer)
        token_plus =  tokenizer(text)
        label = []
        current_tokenid = -1
        for each_interval in interval:
            for i in range(each_interval[0], each_interval[1]+1):
                token_id = token_plus.char_to_token(i)
                if token_id != current_tokenid and token_id is not None:
                    current_tokenid = token_id
                    label.append(map_class_to_num[each_interval[2]])
                    # print(each_interval[0], each_interval[1], each_interval[2], current_tokenid)
                elif each_interval[0] == each_interval[1]:
                    label.append(2)
                    current_tokenid = token_id
                    # print(each_interval[0], each_interval[1], each_interval[2], current_tokenid)
        if len(label) >= max_len:
            return label[:max_len]
        else:
            label += [0]*(max_len-len(label))
            return label
    else:
        # token_plus =  tokenizer(text)
        label = [0]*max_len
        return label


def text_label(text, edit_start, edit_end, edit_text, tokenizer):
    if edit_start != []:
        interval = make_interval(text, edit_start, edit_end, tokenizer)
        correct_text = ""
        num_currect = 0
        for i_interval in interval:
            if i_interval[2] == 'C':
                correct_text += text[i_interval[0]: i_interval[1]]
            else:
                # print(edit_text[num_currect])
                if edit_text[num_currect] is not None:
                    correct_text += edit_text[num_currect]
                num_currect += 1
        return correct_text
    else:
        return text


if __name__ == '__main__':
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    test_label = make_label('wi','train')
    print(test_label[0])
    # print(len(test_label['wrong']), len(test_label['wrong']), len(test_label['error_tag']),len(test_label['error_type']))

    
   
    
    