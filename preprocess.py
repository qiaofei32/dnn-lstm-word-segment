import codecs

data = codecs.open("./data/msr_training_text", 'r', 'utf-8')

f_w = open("./data/msr_training_single_word", 'w')

for line in data.readlines():
	line_new = " " . join( [i for i in line ])
	f_w.write( line_new.encode("utf8") )

f_w.close()