from .vocab import *


def find_word_index(source, target):
    for i, word in enumerate(source.split()):
        if word == target:
            return i
    return -1


# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        import tqdm
        for line in tqdm.tqdm(texts):
            words = line.split()
            for word in words:
                counter[word] += 1
        print(counter)
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, copy_source=None):
        if copy_source is None:
            seq = [self.stoi.get(word, self.unk_index) for word in sentence.split()]
        else:
            seq = [len(self) + copy_source[word] if word in copy_source else self.stoi.get(word, self.unk_index)
                   for word in sentence.split()]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return seq

    def from_seq(self, seq, join=False, with_pad=False, copy_source=None):
        if copy_source is not None:
            words = []
            copy_source = copy_source[0].split()
            for idx in seq:
                if idx < len(self.itos) and (not with_pad or idx != self.pad_index):
                    words.append(self.itos[idx])
                else:
                    if idx - len(self) >= len(copy_source):
                        words.append("<oov>")
                    else:
                        words.append(copy_source[idx - len(self)])

            # words = [self.itos[idx] if idx < len(self.itos) else source.split()[idx - len(self)] for idx, source in
            #          zip(seq, copy_source) if not with_pad or idx != self.pad_index]
        else:
            words = [self.itos[idx] if idx < len(self.itos) else "<%d>" % idx for idx in seq if
                     not with_pad or idx != self.pad_index]
        if join:
            return " ".join(words)
        else:
            return words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
