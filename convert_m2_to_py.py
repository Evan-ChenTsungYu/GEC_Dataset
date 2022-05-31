

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
        with open(input_path + file_name + '.m2', 'r') as input_file, open(output_src_path + file_name  + '_src.txt', 'w+') as output_src_file, open(output_tgt_path + file_name  + '_tgt.txt', 'w+') as output_tgt_file:
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
                    sid = int(sid) + 1; eid = int(eid) + 1;
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
    make_m2_to_py('train')
    make_m2_to_py('test')

        
    