# -*- encoding: utf-8 -*-
import os
import sys
import numpy
import numpy as np
import codecs
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from gensim.models import Word2Vec

NUM_LIST = [str(i) for i in  range(10)]

ENG_LIST = [i for i in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")]

def character_tagging(input_file, output_file):
	'''
	他/S 忧/B 心/M 忡/M 忡/E 地/S 说/S ：/S “/S 一/B 直/E 未/B 能/E 获/B 得/E 正/B 式/E 注/B 册/E ，/S 学/B 校/E 就/S 没/S 合/B 法/E 地/B 位/E ，/S 学/B 生/E 的/S 课/B 桌/E 也/S 就/S 不/S 稳/S 。/S
	'''
	input_data = codecs.open(input_file, 'r', 'utf-8')
	output_data = codecs.open(output_file, 'w', 'utf-8')
	for line in input_data.readlines():
		word_list = line.strip().split()
		for word in word_list:
			if len(word) == 1:
				output_data.write(word + "/S ")
			else:
				output_data.write(word[0] + "/B ")
				for w in word[1:len(word)-1]:
					output_data.write(w + "/M ")
				output_data.write(word[len(word)-1] + "/E ")
		output_data.write("\n")
	input_data.close()
	output_data.close()

def words_to_ids(words_file="data/msr_training_text"):

	words_ids_dict = {}
	if os.path.exists("pkl/w2v_words_ids_dict.pkl"):
		words_ids_dict = joblib.load("pkl/w2v_words_ids_dict.pkl")
	else:
		words = codecs.open(words_file, 'r', 'utf-8').read()
		words = words.replace("\n", "")
		words = words.replace(" ", "")
		words = list(words)
		words = set(words)
		words = list(words)
		for index, word in enumerate(words):
			words_ids_dict[word] = index
		joblib.dump(words_ids_dict, "pkl/w2v_words_ids_dict.pkl")
	# print words_ids_dict
	return words_ids_dict

def create_model(dim ):

	model = Sequential()
	nb_classes = 4

	model.add(
			LSTM(
				output_dim=512,
				# input_dim = dim
				input_shape=(dim, 200)
			)
	)
	model.add(Dropout(0.2))
	# model.add(LSTM(512))
	# model.add(Dropout(0.2))
	model.add(Dense(128))
	model.add(Dense(4))
	model.add(Activation("softmax"))

	# # model.add(Embedding(maxfeatures, 100, weights=init_weight))  # 使用初使词向量可以增加准确率
	# embedding_layer = Embedding( input_dim=dim, output_dim=128 )
	# model.add(embedding_layer)
	# model.add(Dense(128, 128))
	# model.add(Dropout(0.5))
	# model.add(Activation('relu'))
	# model.add(Dense(32, nb_classes))
	# model.add(Activation('softmax'))

	# model.compile(loss='categorical_crossentropy', optimizer='sgd')
	#  model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def load_data(data_file="data/msr_training_taged", word_step=5):
	'''Load Data'''
	TRAIN_X = []
	TRAIN_Y = []
	data = codecs.open(data_file, 'r', 'utf-8')
	label_id_dict = {u'S': 0, u'B': 1, u'M': 2, u'E': 3}
	num_dict = {n: l for l, n in label_id_dict.iteritems()}
	nb_classes = 4

	w2v_model = Word2Vec.load_word2vec_format('./data/msr_training_single_word.w2v.bin', binary=True, unicode_errors='ignore')
	vocabs = w2v_model.vocab.keys()
	FIRST_WORD = w2v_model[w2v_model.vocab.keys()[0]]
	PAD_ARR = numpy.zeros_like(FIRST_WORD)

	lines = 0
	for line in data.readlines():
		word_tags = line.split()
		if not word_tags:continue
		word_tags = ["PADDING/S"] * (word_step // 2) + word_tags + ["PADDING/S"] * (word_step // 2)
		for i in range(len(word_tags) - 1 - word_step // 2):
			context = word_tags[i:i + word_step]
			TRAIN_X_TMP = []
			for j, wt in enumerate(context):
				w, t = wt.split("/")
				try:
					w_id = w2v_model[w]
				except:
					w_id = PAD_ARR
				if w in NUM_LIST:
					w_id = PAD_ARR
				if w in ENG_LIST:
					w_id = PAD_ARR
				TRAIN_X_TMP.append(w_id)

			word = word_tags[i+word_step//2].split("/")[0]
			word_tag = word_tags[i+word_step//2].split("/")[-1]
			if word in NUM_LIST+ENG_LIST:
				word_tag = u"S"

			if len(TRAIN_X_TMP)==word_step:
				X = numpy.array(TRAIN_X_TMP)
				X = numpy.array([X])
				Y = label_id_dict[word_tag]
				Y = np_utils.to_categorical([Y], nb_classes)
				yield X, Y

def predict_tag(model, label_id_dict, sentence, word_step=5):
	'''Predict word tag'''
	num_dict = {n: l for l, n in label_id_dict.iteritems()}
	print label_id_dict

	sentence = list(sentence)
	sentence = ["PADDING"] * (word_step // 2) + sentence + ["PADDING"] * (word_step // 2)

	w2v_model = Word2Vec.load_word2vec_format('./data/msr_training_single_word.w2v.bin', binary=True, unicode_errors='ignore')
	vocabs = w2v_model.vocab.keys()
	FIRST_WORD = w2v_model[w2v_model.vocab.keys()[0]]
	PAD_ARR = numpy.zeros_like(FIRST_WORD)
	TAG_TMP = None

	for i in range(len(sentence) + 1 - word_step):
		context = sentence[i:i + word_step]
		word = sentence[i+word_step//2]
		w_id_list = []
		for j, w in enumerate(context):
			if w not in vocabs:
				w_id = PAD_ARR
			else:
				w_id = w2v_model[w]
			if w in NUM_LIST:
				w_id = PAD_ARR
			if w in ENG_LIST:
				w_id = PAD_ARR

			w_id_list.append(w_id)
		TEST_X = numpy.array( [w_id_list] )
		# print TEST_X
		prob = model.predict(TEST_X)
		# tag = num_dict[prob.argmax()]
		prob_sort_list = prob.argsort().tolist()[0]
		prob_sort_list.reverse()
		# print prob_sort_list
		for prob_i in prob_sort_list:
			tag = num_dict[prob_i]
			if TAG_TMP is None and tag in [u"E", u"M"]:continue
			if TAG_TMP==u"B"   and tag in [u"B", u"S"]:continue
			if TAG_TMP==u"E"   and tag in [u"E", u"M"]:continue
			if TAG_TMP==u"M"   and tag in [u"B", u"S"]:continue
			if TAG_TMP==u"S"   and tag in [u"E", u"M"]:continue
			if TAG_TMP==u"ENG" and tag in [u"E", u"M"]:continue
			if TAG_TMP==u"NUM" and tag in [u"E", u"M"]:continue
			break

		# if word in ENG_LIST:
		# 	tag = u"ENG"
		# if word in NUM_LIST:
		# 	tag = u"NUM"

		TAG_TMP = tag
		print "%-2d" %i, word.encode("gbk"), tag, prob_sort_list, prob

if __name__ == '__main__':

	model = create_model( 1 * 5 + 0 )
	data_file = "data/msr_training_taged"
	word_step = 5
	label_id_dict = {u'S': 0, u'B': 1, u'M': 2, u'E': 3}

	if os.name == "nt":
		os.system("cls")
	else:
		os.system("clear")

	if os.path.exists("pkl/w2v-word-segment.model"):
		model.load_weights("pkl/w2v-word-segment.model")
	else:

		filepath = "pkl/weights-{epoch:03d}.hdf5"
		checkpoint = ModelCheckpoint(filepath, verbose=1, mode='auto')
		callbacks_list = [checkpoint]

		try:
			history = model.fit_generator(
				load_data(data_file, word_step),
				samples_per_epoch = 20000,
				nb_epoch = 500,
				verbose=2,
				callbacks=callbacks_list,
				nb_worker=4
			)
			plt.plot(history.history["loss"])
			plt.show()
			plt.savefig("pkl/w2v-loss.png")
			model.save("pkl/w2v-word-segment.model")

		except:
			model.save("pkl/w2v-word-segment.model")

	sentence = u"中国农业科学院北京畜牧兽医研究所肄业博士研究生魏景亮在“知乎网”实名举报“国家转基因检测中心造假”。"
	predict_tag(model, label_id_dict, sentence)

	sentence = u"在湖北省武汉市中心城区，房地产项目用地长期闲置成为“临时停车场”现象并非个例。根据规定，若房地产项目用地长期闲置，地方国土部门可无偿收回土地使用权。在武汉，当地有关部门是否知道房地产项目用地长期未开发情况？《法制日报》记者对此进行了调查。"
	predict_tag(model, label_id_dict, sentence)

