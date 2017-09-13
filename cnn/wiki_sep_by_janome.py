# -*- cording: utf-8 -*-

import argparse
import mojimoji
from janome.tokenizer import Tokenizer

parser = argparse.ArgumentParser(description='generate wikipedia corpus for fasttext')
parser.add_argument("wiki", type=str,
                    help="wiki data path")
parser.add_argument("out", type=str,
                    help="output dir path")

FILE_NAME = "/wiki_sep_janome_norm.txt"

def init_output_file(out):
    with open(out + FILE_NAME, "w") as f:
        f.write("")

def norm_text(text):
    text = mojimoji.zen_to_han(text, kana=False) # 数字と英語を全て半角に
    text = text.lower()  # 英語は全て小文字に
    return text

wakati = Tokenizer()
def wakati_janome(txt):
    txt = txt[:1000]
    txt = norm_text(txt)
    txt = txt.replace("\n", " ")
    tokens = wakati.tokenize(txt)
    sep_txt = [i.surface for i in tokens]
    text = " ".join(sep_txt)
    return text

def output_wakati_text(s, out):
    text = wakati_janome(s) + "\n"
    with open(out + FILE_NAME, "a") as f:
        f.write(text)

def main(wiki_path, out):
    init_output_file(out)
    with open(wiki_path, "r") as f:
        for line in f:
            sentence = line.split("。")
            for s in sentence:
                if not s in ["", "\n"]:
                    if len(sentence) == 1:
                        output_wakati_text(s, out)
                    else:
                        output_wakati_text(s + "。", out)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.wiki, args.out)
