from typing import Dict, Iterable, List

import pandas as pd
from pathlib import Path

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer


@DatasetReader.register('classification-csv')
class ClassificationCsvReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 balance: bool = False,
                 text_col: str = "text",
                 label_col: str = "label",
                 data_dir: str = "data/CrisisNLP_volunteers_labeled_data",
                 exclude: List[str] = ["MH370", "Respiratory", "ebola"],
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.balance = balance
        self.text_col = text_col
        self.label_col = label_col
        self.data_dir = Path(data_dir)
        self.exclude = exclude

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_paths = list(self.data_dir.glob(file_path))
        dfs = []
        for fp in file_paths:
            if any([e in str(fp) for e in self.exclude]):
                continue
            df = pd.read_csv(fp, encoding = "ISO-8859-1", skipinitialspace=True)
            if self.balance and "test" not in str(fp):
                label_counts = df[self.label_col].value_counts()
                max_count = max(label_counts)
                for label, count in label_counts.items():
                    n = max_count // count - 1
                    r = max_count % count
                    df = df.append([df.loc[df[self.label_col] == label]]*n, ignore_index=True)
                    df = df.append(df.loc[df[self.label_col] == label].iloc[:r], ignore_index=True)
            dfs += [df]
        df = pd.concat(dfs)
        for i,row in df.iterrows():
            try:
                yield self.text_to_instance(row[self.text_col], row[self.label_col])
            except Exception as e:
                print(f"Exception: \n{e}\n{row}")