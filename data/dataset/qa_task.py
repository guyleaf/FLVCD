from torch.utils.data import Dataset
import torch
import numpy as np


class BabiQADataset(Dataset):
    def __init__(self, path, enc_vocab, dec_vocab, story_len, seq_len):
        self.path = path
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab
        self.story_len = story_len
        self.seq_len = seq_len

        self.data = self.get_dialog(path)
        self.data = [self.to_seq(story, query, answer) for story, query, answer in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_dialog(self, path):
        lines = open(path, "r", encoding="utf-8").readlines()
        dialog = []
        qa = []
        for i, line in enumerate(lines):
            line = line.replace(".", "").replace("?", "")
            if line == "\n" or (line[:2] == "1 " and i != 0):
                qa.extend(self.separate_dialog(dialog))
                dialog.clear()
            else:
                line = " ".join(line[:-1].split(" ")[1:])
                dialog.append(line)

        return qa

    def separate_dialog(self, dialog):
        story_history = []
        qa = []
        for line in dialog:
            if line.find("\t") >= 0:
                line = line.split("\t")
                qa.append((story_history.copy(), line[0][:-1], line[1]))
            else:
                story_history.append(line)
        return qa

    # def subsequent_mask(self, size):
    #     "Mask out subsequent positions."
    #     attn_shape = (1, size, size)
    #     subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    #     return torch.from_numpy(subsequent_mask) == 0
    #
    # def make_std_mask(self, tgt, pad):
    #     "Create a mask to hide padding and future words."
    #     tgt_mask = (tgt != pad).unsqueeze(-2)
    #     tgt_mask = tgt_mask & self.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    #     return tgt_mask

    def to_seq(self, story, query, answer):
        story.append(query)
        story_mask = [1 for _ in range(len(story))] + [0 for _ in range(self.story_len - len(story))]
        story = self.story_pad([self.enc_vocab.to_seq(s, seq_len=self.seq_len) for s in story])
        query = self.enc_vocab.to_seq(query, seq_len=self.seq_len)
        answer = self.enc_vocab.to_seq(answer, seq_len=1)
        output = {
            "story": torch.tensor(story),
            "query": torch.tensor(query),
            "answer": torch.tensor(answer),
            "story_mask": torch.tensor(story_mask).unsqueeze(-2),
            "answer_mask": torch.tensor([1]).unsqueeze(-1)
        }
        return output

    def story_pad(self, story_seq):
        if len(story_seq) > self.story_len:
            story_seq = story_seq[:self.story_len]
        else:
            story_seq.extend([[0 for _ in range(self.seq_len)] for __ in range(self.story_len - len(story_seq))])
        return story_seq
