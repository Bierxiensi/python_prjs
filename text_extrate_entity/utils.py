import json

def bio_tagging(text, entities):
    bio_tags = ['O'] * len(text)
    for start, end, label in entities:
        for i in range(start, end+1):
            if i == start:
                bio_tags[i] = 'B-' + label
            else:
                bio_tags[i] = 'I-' + label
    tokens = []
    for char in text:
        tokens.append(char)
    token_bio_tags = [(token, bio_tag) for token, bio_tag in zip(tokens, bio_tags)]
    return token_bio_tags

def generate_bio():
    with open('entity.txt', 'w', encoding='utf-8') as file:
        with open("./data/entity.json", "r", encoding="utf-8") as f:
            object_data = json.loads(f.read())
            for item in object_data:
                text = item['text']
                entities = item['entities']
                tagged_tokens = bio_tagging(text, entities)
                print(tagged_tokens)
                for token, bio_tag in tagged_tokens:
                    file.write(f"{token} {bio_tag}\n")
                file.write(f"\n")

def generate_test():
    text = "F-15鹰式战斗机，是一款美国开发生产的全天候、高机动性的战术战斗机。针对获得与维持空优而设计的它，是美国空军现役的主力战斗机之一。F-15是由1962年展开的F-X计划发展出来。按照原先的欧美标准被归类为第三代战斗机（现在已和俄罗斯标准统一为第四代战机），与F-16，美国海军的F-14、F-18，法国的幻影2000，俄罗斯的米格-29、米格-31、米格-35、Su-27、Su-30，中国的J-10、J-11等是同一世代。F-15已经出口到日本、以色列、韩国、新加坡、沙特等国家。"
    with open('test.txt', 'w', encoding='utf-8') as file:
        for char in text:
            file.write(f"{char}\n")

def read_data(file_path):
    # 读取数据集
    with open(file_path, "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]

    # 添加原文句子以及该句子的标签

    # 读取空行所在的行号
    index = [-1]
    index.extend([i for i, _ in enumerate(content) if ' ' not in _])
    index.append(len(content))

    # 按空行分割，读取原文句子及标注序列
    sentences, tags = [], []
    for j in range(len(index)-1):
        sent, tag = [], []
        segment = content[index[j]+1: index[j+1]]
        for line in segment:
            sent.append(line.split()[0])
            tag.append(line.split()[-1])

        sentences.append(''.join(sent))
        tags.append(tag)

    # 去除空的句子及标注序列，一般放在末尾
    sentences = [_ for _ in sentences if _]
    tags = [_ for _ in tags if _]

    return sentences, tags

def label2id():
    train_sents, train_tags = read_data('./entity.txt')
    # 标签转换成id，并保存成文件
    unique_tags = []
    for seq in train_tags:
        for _ in seq:
            if _ not in unique_tags:
                unique_tags.append(_)

    label_id_dict = dict(zip(unique_tags, range(1, len(unique_tags) + 1)))
    with open("label2id.json", "w", encoding="utf-8") as g:
        g.write(json.dumps(label_id_dict, ensure_ascii=False, indent=2))
# label2id()