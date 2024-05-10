import json
import os
import json
import keras
import numpy as np
from tqdm import tqdm
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.backend import K, batch_gather
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import open
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
import numpy as np

model_root_path = rootPath + '/model/'
corpus_root_path = rootPath + '/corpus/'
rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_process(train_data_file_path, valid_data_file_path, max_len, params_path):
    train_data = json.load(open(train_data_file_path, encoding='utf-8'))

    if valid_data_file_path:
        train_data_ret = train_data
        valid_data_ret = json.load(open(valid_data_file_path, encoding='utf-8'))
    else:
        split = int(len(train_data) * 0.8)
        train_data_ret, valid_data_ret = train_data[:split], train_data[split:]
    p2s_dict = {}
    p2o_dict = {}
    predicate = []

    for content in train_data:
        for spo in content.get('new_spo_list'):
            s_type = spo.get('s').get('type')
            p_key = spo.get('p').get('entity')
            o_type = spo.get('o').get('type')
            if p_key not in p2s_dict:
                p2s_dict[p_key] = s_type
            if p_key not in p2o_dict:
                p2o_dict[p_key] = o_type
            if p_key not in predicate:
                predicate.append(p_key)
    i2p_dict = {i: key for i, key in enumerate(predicate)}
    p2i_dict = {key: i for i, key in enumerate(predicate)}
    save_params = {}
    save_params['p2s_dict'] = p2s_dict
    save_params['i2p_dict'] = i2p_dict
    save_params['p2o_dict'] = p2o_dict
    save_params['maxlen'] = max_len
    save_params['num_classes'] = len(i2p_dict)
    # 数据保存
    json.dump(save_params,
              open(params_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    return train_data_ret, valid_data_ret, p2s_dict, p2o_dict, i2p_dict, p2i_dict

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class Data_Generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size, tokenizer, p2i_dict, maxlen):
        super().__init__(data, batch_size=batch_size)
        self.tokenizer = tokenizer
        self.p2i_dict = p2i_dict
        self.maxlen = maxlen

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        # elif len(caches) == self.buffer_size:
                        #     isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(d['text'], max_length=self.maxlen)
            # 整理三元组 {s: [(o_start,0_end, p)]}/{s_token_ids:[]}
            spoes = {}
            for spo in d['new_spo_list']:
                s = spo['s']
                p = spo['p']
                o = spo['o']
                s_token = self.tokenizer.encode(s['entity'])[0][1:-1]
                p = self.p2i_dict[p['entity']]
                o_token = self.tokenizer.encode(o['entity'])[0][1:-1]
                s_idx = search(s_token, token_ids)  # s_idx s起始位置
                o_idx = search(o_token, token_ids)  # o_idx o起始位置
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s_token) - 1)  # s s起始结束位置，s的类别
                    o = (o_idx, o_idx + len(o_token) - 1, p)  # o o起始结束位置及p的id,o的类别
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签，采用二维向量分别标记subject的起始位置和结束位置
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(self.p2i_dict), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(batch_subject_labels, padding=np.zeros(2))
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels,
                                                           padding=np.zeros((3, 2)))
                    yield [
                              batch_token_ids, batch_segment_ids,
                              batch_subject_labels, batch_subject_ids, batch_object_labels

                          ], None
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []

def evaluate(tokenizer,data,predict):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()

    class SPO(tuple):
        """用来存三元组的类
        表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
        使得在判断两个三元组是否等价时容错性更好。
        """

        def __init__(self, spo):
            self.spox = (
                tuple(spo[0]),
                spo[1],
                tuple(spo[2]),
            )

        def __hash__(self):
            return self.spox.__hash__()

        def __eq__(self, spo):
            return self.spox == spo.spox

    for d in data:
        R = set([SPO(spo) for spo in
                 [[tokenizer.tokenize(spo_str[0][0]), spo_str[1], tokenizer.tokenize(spo_str[2][0])] for
                  spo_str
                  in predict(d['text'])]])
        T = set([SPO(spo) for spo in
                 [[tokenizer.tokenize(spo_str['s']['entity']), spo_str['p']['entity'],
                   tokenizer.tokenize(spo_str['o']['entity'])] for spo_str
                  in d['new_spo_list']]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))
        s = json.dumps(
            {
                'text': d['text'],
                'spo_list': list(T),
                'spo_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            },
            ensure_ascii=False,
            indent=4)
        f.write(s + '/n')
    pbar.close()
    f.close()
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """

    def __init__(self, model, model_path, tokenizer,predict,optimizer,valid_data):
        self.EMAer = optimizer
        self.best_val_f1 = 0.
        self.model = model
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.predict = predict
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        self.EMAer.apply_ema_weights()
        f1, precision, recall = evaluate(self.tokenizer,self.valid_data,self.predict)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(self.model_path)
        self.EMAer.reset_old_weights()
        print('f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f/n' %
              (f1, precision, recall, self.best_val_f1))

class ReextractBertTrainHandler():
    def __init__(self, params, Train=False):
        self.bert_config_path = model_root_path + "chinese_L-12_H-768_A-12/bert_config.json"
        self.bert_checkpoint_path = model_root_path + "chinese_L-12_H-768_A-12/bert_model.ckpt"
        self.bert_vocab_path = model_root_path + "chinese_L-12_H-768_A-12/vocab.txt"
        self.tokenizer = Tokenizer(self.bert_vocab_path, do_lower_case=True)
        self.model_path = model_root_path + "best_model.weights"
        self.params_path = model_root_path + 'params.json'
        gpu_id = params.get("gpu_id", None)
        self._set_gpu_id(gpu_id)  # 设置训练的GPU_ID
        self.memory_fraction = params.get('memory_fraction')
        if Train:
            self.train_data_file_path = params.get('train_data_path')
            self.valid_data_file_path = params.get('valid_data_path')
            self.maxlen = params.get('maxlen', 128)
            self.batch_size = params.get('batch_size', 32)
            self.epoch = params.get('epoch')
            self.data_process()
        else:
            load_params = json.load(open(self.params_path, encoding='utf-8'))
            self.maxlen = load_params.get('maxlen')
            self.num_classes = load_params.get('num_classes')
            self.p2s_dict = load_params.get('p2s_dict')
            self.i2p_dict = load_params.get('i2p_dict')
            self.p2o_dict = load_params.get('p2o_dict')
        self.build_model()
        if not Train:
            self.load_model()

    def _set_gpu_id(self, gpu_id):
        if gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def data_process(self):
        self.train_data, self.valid_data, self.p2s_dict, self.p2o_dict, self.i2p_dict, self.p2i_dict = data_process(
            self.train_data_file_path, self.valid_data_file_path, self.maxlen, self.params_path)
        self.num_classes = len(self.i2p_dict)
        self.train_generator = Data_Generator(self.train_data, self.batch_size, self.tokenizer, self.p2i_dict,
                                              self.maxlen)

    def extrac_subject(self, inputs):
        """根据subject_ids从output中取出subject的向量表征
        """
        output, subject_ids = inputs
        subject_ids = K.cast(subject_ids, 'int32')
        start = batch_gather(output, subject_ids[:, :1])
        end = batch_gather(output, subject_ids[:, 1:])
        subject = K.concatenate([start, end], 2)
        return subject[:, 0]

    def build_model(self):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
        if self.memory_fraction:
            config.gpu_options.per_process_gpu_memory_fraction = self.memory_fraction
            config.gpu_options.allow_growth = False
        else:
            config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        # 补充输入
        subject_labels = Input(shape=(None, 2), name='Subject-Labels')
        subject_ids = Input(shape=(2,), name='Subject-Ids')
        object_labels = Input(shape=(None, self.num_classes, 2), name='Object-Labels')
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=self.bert_config_path,
            checkpoint_path=self.bert_checkpoint_path,
            return_keras_model=False,
        )
        # 预测subject
        output = Dense(units=2,
                       activation='sigmoid',
                       kernel_initializer=bert.initializer)(bert.model.output)
        subject_preds = Lambda(lambda x: x ** 2)(output)
        self.subject_model = Model(bert.model.inputs, subject_preds)
        # 传入subject，预测object
        # 通过Conditional Layer Normalization将subject融入到object的预测中
        output = bert.model.layers[-2].get_output_at(-1)
        subject = Lambda(self.extrac_subject)([output, subject_ids])
        output = LayerNormalization(conditional=True)([output, subject])
        output = Dense(units=self.num_classes * 2,
                       activation='sigmoid',
                       kernel_initializer=bert.initializer)(output)
        output = Lambda(lambda x: x ** 4)(output)
        object_preds = Reshape((-1, self.num_classes, 2))(output)
        self.object_model = Model(bert.model.inputs + [subject_ids], object_preds)
        # 训练模型
        self.train_model = Model(bert.model.inputs + [subject_labels, subject_ids, object_labels],
                                 [subject_preds, object_preds])
        mask = bert.model.get_layer('Embedding-Token').output_mask
        mask = K.cast(mask, K.floatx())
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        self.train_model.add_loss(subject_loss + object_loss)
        AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
        self.optimizer = AdamEMA(lr=1e-4)
        self.train_model.compile(optimizer=self.optimizer)

    def load_model(self):
        self.train_model.load_weights(self.model_path)

    def predict(self, text):
        """
        抽取输入text所包含的三元组
        text：str(<离开>是由张宇谱曲，演唱)
        """
        tokens = self.tokenizer.tokenize(text, max_length=self.maxlen)
        token_ids, segment_ids = self.tokenizer.encode(text, max_length=self.maxlen)
        # 抽取subject
        subject_preds = self.subject_model.predict([[token_ids], [segment_ids]])
        start = np.where(subject_preds[0, :, 0] > 0.6)[0]
        end = np.where(subject_preds[0, :, 1] > 0.5)[0]
        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))
        if subjects:
            spoes = []
            token_ids = np.repeat([token_ids], len(subjects), 0)
            segment_ids = np.repeat([segment_ids], len(subjects), 0)
            subjects = np.array(subjects)
            # 传入subject，抽取object和predicate
            object_preds = self.object_model.predict([token_ids, segment_ids, subjects])
            for subject, object_pred in zip(subjects, object_preds):
                start = np.where(object_pred[:, :, 0] > 0.6)
                end = np.where(object_pred[:, :, 1] > 0.5)
                for _start, predicate1 in zip(*start):
                    for _end, predicate2 in zip(*end):
                        if _start <= _end and predicate1 == predicate2:
                            spoes.append((subject, predicate1, (_start, _end)))
                            break
            return [
                (
                    [self.tokenizer.decode(token_ids[0, s[0]:s[1] + 1], tokens[s[0]:s[1] + 1]),
                     self.p2s_dict[self.i2p_dict[p]]],
                    self.i2p_dict[p],
                    [self.tokenizer.decode(token_ids[0, o[0]:o[1] + 1], tokens[o[0]:o[1] + 1]),
                     self.p2o_dict[self.i2p_dict[p]]],
                    (s[0], s[1] + 1),
                    (o[0], o[1] + 1)
                ) for s, p, o in spoes
            ]
        else:
            return []

    def train(self):
        evaluator = Evaluator(self.train_model, self.model_path, self.tokenizer, self.predict, self.optimizer,
                              self.valid_data)

        self.train_model.fit_generator(self.train_generator.forfit(),
                                       steps_per_epoch=len(self.train_generator),
                                       epochs=self.epoch,
                                       callbacks=[evaluator])