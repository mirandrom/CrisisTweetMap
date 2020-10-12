from overrides import overrides
from typing import List
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
        output_dict["text"] = inputs["text"]
        output_dict["label_probs"] = {label_dict[i]: prob for i,prob in enumerate(output_dict["probs"])}
        output_dict["prediction"] = label_dict[np.argmax(output_dict["probs"])]
        output_dict["prediction_confidence"] = output_dict["label_probs"][max(output_dict["label_probs"], key=output_dict["label_probs"].get)]
        return output_dict

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        outputs =  self.predict_batch_instance(instances)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        for i,output_dict in enumerate(outputs):
            outputs[i]["label_probs"] = {label_dict[i]: prob for i, prob in enumerate(output_dict["probs"])}
            outputs[i]["text"] = inputs[i]["text"]
            outputs[i]["prediction"] = label_dict[np.argmax(output_dict["probs"])]
            outputs[i]["prediction_confidence"] = output_dict["label_probs"][max(output_dict["label_probs"], key=output_dict["label_probs"].get)]
        return outputs

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        return self._dataset_reader.text_to_instance(text=text)

