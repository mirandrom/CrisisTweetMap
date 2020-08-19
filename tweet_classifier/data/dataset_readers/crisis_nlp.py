from typing import Dict, Iterable, List, Optional

import logging
import pandas as pd
from overrides import overrides
from pathlib import Path

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register('csv')
class CsvDatasetReader(DatasetReader):
    """
    Reads csv datasets provided by CrisisNLP along " Twitter as a Lifeline:
    Human-annotated Twitter Corpora for NLP of Crisis-related Messages."
    (Imran, 2016).

    In particular, this dataset reader allows reading from training sets for
    multiple events, to allow evaluating model generalizability across events
    and transfer learning effectiveness.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False,
                 label_col: str = 'choose_one_category',
                 text_col: str = 'tweet_text',
                 data_dir: str = "./",
                 exclude: Optional[List[str]] = None,
                 ) -> None:
        super().__init__(lazy=lazy)
        self.label_col = label_col
        self.text_col = text_col
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.data_dir = Path(data_dir)
        self.exclude = exclude or []

    @overrides
    def _read(self, file_path_glob: str) -> Iterable[Instance]:
        """Hacky method of reading from multiple files for test/train
        """
        file_paths = list(self.data_dir.glob(file_path_glob))
        dfs = []
        for fp in file_paths:
            if any([e in str(fp) for e in self.exclude]):
                continue
            df = pd.read_csv(fp, encoding="ISO-8859-1", skipinitialspace=True)
            dfs += [df]

        df = pd.concat(dfs)
        for i,row in df.iterrows():
            try:
                yield self.text_to_instance(row[self.text_col], row[self.label_col])
            except Exception as e:
                print("Error: ", e)
                print("Invalid data:")
                print(row)

    @overrides
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(text)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)
