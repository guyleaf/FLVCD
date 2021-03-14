import pickle
from data.vocab.word import WordVocab

for task in range(1, 21):
    path = "babi-qa/task%d_train.txt" % task
    vocab_path = "babi-qa/vocab/task%d_vocab.pkl" % task
    texts = [line[line.find(" ") + 1:-1] for line in open(path)]
    texts = [" ".join(line.split("\t")[:-1]) if line.find("\t") >= 0 else line for line in texts]
    texts = [line.replace(".", "").replace("?", "") for line in texts]
    word_vocab = WordVocab(texts)
    with open(vocab_path, "wb") as f:
        pickle.dump(word_vocab, f)

# Train Answer Vocab
for task in range(1, 21):
    path = "babi-qa/task%d_train.txt" % task
    vocab_path = "babi-qa/vocab/task%d_answer_vocab.pkl" % task
    answers = [line.split("\t")[1] for line in open(path) if line.find("\t") >= 0]
    word_vocab = WordVocab(answers)
    with open(vocab_path, "wb") as f:
        pickle.dump(word_vocab, f)
