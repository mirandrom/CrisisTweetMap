from typing import Dict, Iterable

import logging
import pandas as pd
from overrides import overrides

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
                 text_col: str = 'tweet_text') -> None:
        super().__init__(lazy=lazy)
        self.label_col = label_col
        self.text_col = text_col
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """Hacky method of reading from multiple files for test/train
        """
        df = pd.read_csv(file_path, encoding = "ISO-8859-1")
        for i,row in df.iterrows():
            try:
                yield self.text_to_instance(row[self.text_col], row[self.label_col])
            except Exception as e:
                print("Error: ", e)
                print("Invalid data:")
                print(row)

    @overrides
    def text_to_instance(self, text: str, label: str) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(text)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        fields['label'] = LabelField(label)
        return Instance(fields)
