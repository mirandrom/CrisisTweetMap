from overrides import overrides
import numpy as np

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('tweet_predictor')
class TweetPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict["all_labels"] = all_labels
        output_dict["label_probs"] = {label_dict[i]: prob for i,prob in enumerate(output_dict["probs"])}

        output_dict["prediction"] = label_dict[np.argmax(output_dict["probs"])]
        output_dict["prediction_confidence"] = np.max(output_dict["probs"])
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        return self._dataset_reader.text_to_instance(text=text)