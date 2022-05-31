
import pandas as pd

        
        
def num_error_tag(input_files):
    tag_list = []
    with open(input_files+ '.m2', 'r') as input_file:
        for line in input_file:
            if line.startswith('A'):
                    line = line[2:]
                    info = line.split("|||")
                    error_type = info[1]
                    if error_type not in tag_list:
                        tag_list.append(error_type)
    print(tag_list) 
    

def make_error_tag(input_files):
    words = []
    head_ptr = 0
    start_id = 0
    end_id = 0
    output = []
    
    # for file_name in list_files:
    with open(input_files+ '.m2', 'r') as input_file:
        prompt_sent = []
        correct_sent = []
        error_tags = []
        error_types = []
        for line in input_file:
            line = line.strip()
            if line.startswith('S') and head_ptr == 0:
                data_frame = {'origin':[], 'corrected':[],'prompt':[], 'error_tag' : [], 'error_type' : []}
                line = line[2:]
                words = line.split()
                for i in range(0, len(words)+1):
                    error_tags.append(0)
                    error_types.append('None')
            elif line.startswith('A'):
                line = line[2:]
                info = line.split("|||")
                start_id, end_id = info[0].split()
                start_id = int(start_id)
                end_id = int(end_id)
                error_type = info[1]
                annotator = int(info[5])
                error_modified = info[2]
                if annotator == 0:
                    if start_id == -1:
                        correct_sent = words
                        prompt_sent = words
                        head_ptr =  len(words)
                    else:
                        prompt_sent += words[head_ptr:start_id]
                        correct_sent += words[head_ptr:start_id]
                        correct_sent += [error_modified]
                        if start_id == end_id:
                            prompt_sent += ([ '[ ' ] + ['NONE'] + [ '|' , error_modified ,']'])
                            # print(start_id, len(error_tags), words, len(error_tags))
                            error_tags[start_id] = 1
                            error_types[start_id] = error_type
                        else:
                            prompt_sent += ([ '[ ' ] + words[start_id:end_id] + [ '|' , error_modified ,']'])
                            for i in range(start_id, end_id):
                                error_tags[i] = 1
                                error_types[start_id] = error_type       
                        head_ptr =  end_id 
                else:
                    # print(annotator)
                    continue
            elif line.startswith('S') :
                line = line[2:]
                correct_sent += words[head_ptr:]
                prompt_sent += words[head_ptr:]
                data_frame['corrected'] = correct_sent 
                data_frame['prompt'] = prompt_sent 
                data_frame['origin'] = words
                data_frame['error_tag'] = error_tags
                data_frame['error_type'] = error_types
                output.append(data_frame)

                data_frame = {'origin':[], 'corrected':[],'prompt':[], 'error_tag' : [], 'error_type' : []}
                prompt_sent = []
                correct_sent = []
                error_tags = []
                error_types = []
                head_ptr = 0
                words = line.split()
                for i in range(0, len(words)+1):
                    error_tags.append(0)
                    error_types.append('None')

        correct_sent += words[head_ptr:]
        data_frame['prompt'] = prompt_sent
        data_frame['corrected'] = correct_sent
        data_frame['origin'] = words
        data_frame['error_tag'], data_frame['error_type'] = error_tags, error_types
        output.append(data_frame)

        return output



                    
                    
                    

                
                    
                    

                    
    
    
def make_m2_to_py(tt):
    input_path = '/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/m2/'
    output_src_path = '/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/text_data/'
    output_tgt_path = '/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/text_data/'
    train_file = ['A.train.gold.bea19', 'B.train.gold.bea19', 'C.train.gold.bea19', 'lang8.train.auto.bea19', 'fce.train.gold.bea19']
    dev_file = ['A.dev.gold.bea19', 'B.dev.gold.bea19', 'C.dev.gold.bea19', 'N.dev.gold.bea19', 'lang8.train.auto.bea19', 'fce.dev.gold.bea19', 'fce.test.gold.bea19']

    words = []
    corrected = []
    sid = eid = 0
    prev_sid = prev_eid = -1
    pos = 0

    if tt == 'train':
        list_files = train_file
    else:
        list_files = dev_file
    
    for file_name in list_files:
        with open(input_path + file_name + '.m2', 'r') as input_file, open(output_src_path + file_name  + '_etag_src.txt', 'w+') as output_src_file, open(output_tgt_path + file_name  + '_etag_tgt.txt', 'w+') as output_tgt_file:
            for line in input_file:
                line = line.strip()
                if line.startswith('S'):
                    line = line[2:]
                    words = line.split()
                    corrected = ['<S>'] + words[:]
                    output_src_file.write(line + '\n')
                elif line.startswith('A'):
                    line = line[2:]
                    info = line.split("|||")
                    sid, eid = info[0].split()
                    sid = int(sid) + 1; eid = int(eid) + 1; #start position and end position
                    error_type = info[1] 
                    if error_type == "Um":
                        continue
                    for idx in range(sid, eid):
                        corrected[idx] = ""
                    if sid == eid:
                        if sid == 0: continue	# Originally index was -1, indicating no op
                        if sid != prev_sid or eid != prev_eid:
                            pos = len(corrected[sid-1].split())
                        cur_words = corrected[sid-1].split()
                        cur_words.insert(pos, info[2])
                        pos += len(info[2].split())
                        corrected[sid-1] = " ".join(cur_words)
                    else:
                        corrected[sid] = info[2]
                        pos = 0
                    prev_sid = sid
                    prev_eid = eid
                else:
                    target_sentence = ' '.join([word for word in corrected if word != ""])
                    assert target_sentence.startswith('<S>'), '(' + target_sentence + ')'
                    target_sentence = target_sentence[4:]
                    output_tgt_file.write(target_sentence + '\n')
                    prev_sid = -1
                    prev_eid = -1
                    pos = 0


if __name__ == '__main__':
    
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    num_error_tag('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/m2_train/fce.train')
    test_tag = make_error_tag('/mnt/lustre/home/evan_chen/Cinnamon_Code/GEC_dataset/dataset/m2/test_train', tokenizer, max_len = 40)
    test_index = 3
    print(test_tag[test_index])
    
    token_error_tag, token_error_type = tokenizer_token(test_tag[test_index]['origin'], test_tag[test_index]['error_tag'], test_tag[test_index]['error_type'], tokenizer, max_len = 80)
    print(token_error_tag, test_tag[test_index]['error_tag'])
    

    # make_m2_to_py('test')

        
    