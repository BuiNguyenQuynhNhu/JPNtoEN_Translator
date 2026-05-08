from translation.data.kftt import KFTTDataset
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("kftt_spm.model")


