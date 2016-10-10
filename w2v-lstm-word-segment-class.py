# -*- encoding: utf-8 -*-
import os
import sys
import numpy
import codecs
import argparse
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

class WordSegment(object):
	'''word segment with dnn-lstm'''

	def __init__(self):
		'''initialize'''

		self.s_window 	= 5
		self.w2v_dim 	= 200
		self.nb_classes = 4

		self.label_id_dict    = {u'S': 0, u'B': 1, u'M': 2, u'E': 3}
		self.train_data_file  = "data/msr_training_taged"
		self.w2v_model_file   = "data/msr_training_single_word.w2v.bin"
		self.model_hdf5_file  = "pkl/w2v-word-segment.model"
		self.loss_history	  = "pkl/w2v-loss.png"
		self.check_point_file = "pkl/weights-{epoch:03d}.hdf5"

		self.NUM_LIST  = [str(i) for i in range(10)]
		self.ENG_LIST  = [i for i in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")]

		self.w2v_model = Word2Vec.load_word2vec_format(
			self.w2v_model_file,
			binary=True,
			unicode_errors='ignore'
		)

		self.lstm_model = self.create_model(self.s_window,self.w2v_dim, self.nb_classes)

		# if os.name=="nt":os.system("cls")
		# else:os.system("clear")

		self.train_model()

	def create_model(self, s_window=5, w2v_dim=200, nb_classes=4):
		'''
			create modle funtion
			s_window: train/predict context windows size, default is 5
			w2v_dim: word2vec train output dim size, default is 200
			nb_classes: train/predict tag classes, here using 4-Tag: S/B/M/E, so nb_classes=4
		'''

		model = Sequential()
		model.add(LSTM(output_dim=512, input_shape=(s_window, w2v_dim)))
		model.add(Dropout(0.2))
		model.add(Dense(128))
		model.add(Dense(nb_classes))
		model.add(Activation("softmax"))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	def train_model(self, samples_per_epoch=20000, nb_epoch=500,
					verbose=2, nb_worker=1):
		'''
			train lstm model
		'''

		if os.path.exists(self.model_hdf5_file):
			self.lstm_model.load_weights(self.model_hdf5_file)
		else:
			checkpoint = ModelCheckpoint(self.check_point_file, verbose=1, mode='auto')
			callbacks_list = [checkpoint]

			try:
				history = self.lstm_model.fit_generator(
					self.generate_train_data(self.train_data_file, self.s_window),
					samples_per_epoch = samples_per_epoch,
					nb_epoch = nb_epoch,
					verbose = verbose,
					callbacks = callbacks_list,
					nb_worker = nb_worker
				)
				plt.plot(history.history["loss"])
				plt.show()
				plt.savefig(self.loss_history)
				self.lstm_model.save(self.model_hdf5_file)

			except:
				self.lstm_model.save(self.model_hdf5_file)

	def generate_train_data(self,
					train_data_file="data/msr_training_taged",
					s_window=5,nb_classes=4):
		'''
			Load Data
		'''

		data = codecs.open(train_data_file, 'r', 'utf-8')

		FIRST_WORD = self.w2v_model[self.w2v_model.vocab.keys()[0]]
		PADDING = numpy.zeros_like(FIRST_WORD)

		for line in data.readlines():
			word_tags = line.split()
			if not word_tags:continue
			word_tags = ["PADDING/S"] * (s_window // 2) + word_tags + ["PADDING/S"] * (s_window // 2)
			for i in range(len(word_tags) - 1 - s_window // 2):
				context = word_tags[i:i + s_window]
				TRAIN_X_TMP = []
				for j, wt in enumerate(context):
					w, t = wt.split("/")
					try:
						w_id = self.w2v_model[w]
					except:
						w_id = PADDING
					if w in self.NUM_LIST:
						w_id = PADDING
					if w in self.ENG_LIST:
						w_id = PADDING
					TRAIN_X_TMP.append(w_id)

				word = word_tags[i+s_window//2].split("/")[0]
				word_tag = word_tags[i+s_window//2].split("/")[-1]
				if word in self.NUM_LIST+self.ENG_LIST:
					word_tag = u"S"

				if len(TRAIN_X_TMP)==s_window:
					X = numpy.array(TRAIN_X_TMP)
					X = numpy.array([X])
					Y = self.label_id_dict[word_tag]
					Y = np_utils.to_categorical([Y], nb_classes)
					yield X, Y

	def predict_tag(self, sentence, s_window=5):
		'''
			Predict word tag
		'''

		result = []
		num_dict = {n: l for l, n in self.label_id_dict.iteritems()}
		sentence = list(sentence)
		sentence = ["PADDING"] * (s_window // 2) + sentence + ["PADDING"] * (s_window // 2)

		vocabs = self.w2v_model.vocab.keys()
		FIRST_WORD = self.w2v_model[self.w2v_model.vocab.keys()[0]]
		PADDING = numpy.zeros_like(FIRST_WORD)
		TAG_TMP = None

		for i in range(len(sentence) + 1 - s_window):
			context = sentence[i:i + s_window]
			word = sentence[i+s_window//2]
			w_id_list = []
			for j, w in enumerate(context):
				if w not in vocabs:
					w_id = PADDING
				else:
					w_id = self.w2v_model[w]
				if w in self.NUM_LIST:
					w_id = PADDING
				if w in self.ENG_LIST:
					w_id = PADDING

				w_id_list.append(w_id)
			TEST_X = numpy.array( [w_id_list] )
			# print TEST_X
			prob = self.lstm_model.predict(TEST_X)
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

			TAG_TMP = tag
			# print "%-2d" %i, word.encode("gbk"), tag, prob_sort_list, prob
			result.append((word, tag))

		return result

	def word_tag_parse(self, word_tag_list):
		'''
		parse word_tag_list to sentence list
		:param word_tag_list: list of (word, tag)
		:return: sentence list
		'''

		result = []
		temp = ""
		for word, tag in word_tag_list:
			temp += word
			if tag in ["S", "E"]:
				result.append(temp)
				temp = ""

		if temp:result.append(temp)
		return result

	def cut(self, sentence):
		word_tag_list = self.predict_tag(sentence)
		result = self.word_tag_parse(word_tag_list)
		return result

def test():
	model = WordSegment()
	sentence = u"转基因技术的理论基础来源于进化论衍生来的分子生物学。\
	基因片段的来源可以是提取特定生物体基因组中所需要的目的基因，\
	也可以是人工合成指定序列的DNA片段."
	for i in model.cut(sentence):
		print i.encode("gbk"),
	print

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='word segment with dnn-lstm.')
	parser.add_argument('--f', dest='file', help='file to segment')
	parser.add_argument('--o', dest='output', help='write output to file')
	parser.add_argument('--s', dest='sentence', help='sentence to segment')

	args = parser.parse_args()
	file = args.file
	output = args.output
	sentence = args.sentence

	model = WordSegment()

	if output:
		fw = codecs.open(output, 'w', 'utf-8')

	if file:
		i = 0
		with codecs.open(file, 'r', 'utf-8') as f:
			for line in f.readlines():
				i += 1
				data_seg = model.cut(line.strip())
				if output:
					fw.write(" ".join(data_seg) + "\r\n")
					if i%100==0:sys.stdout.write("\r%d" %i)
				else:
					print " ".join(data_seg)
			f.close()
			if output:fw.close()

	else:
		try:
			sentence = sentence.decode("gbk")
		except:
			sentence = sentence.decode("utf8")
		sentence_seg = model.cut(sentence)
		print " ".join(sentence_seg)
