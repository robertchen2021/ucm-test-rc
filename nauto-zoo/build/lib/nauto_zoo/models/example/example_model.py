import numpy as np
from nauto_zoo import Model, ModelResponse, ModelInput, ModelResponseCollection, ModelResponseTimeline, Timeline, TimelineElement
from typing import Union


class ExampleModel(Model):

    def run(self, model_input: ModelInput) -> Union[ModelResponse, ModelResponseCollection]:
        # this is only an example, in real case, add your custom inference code here
        summary = 'TRUE'  # this is short value describing result of the inference
        confidence = 100  # this tells how certain the output is, should be in range [0; 100]
        raw_output = dict({
            'internals': 'any value that is required',
            'other key': 'other value',
            'this can be float': 3.14159,
            'or other complex type': ['like list', 'of strings'],
            'or nested dictionary': {'with different': 'value types',
                                     'like numbers': 43},
            'value_from_the_input': np.mean(model_input.get('snapshot_in')[0])
        })
        self._logger.info(f'Can use logger to output some diagnostic values, like {confidence}')
        # use self._s3_client. to access s3

        if confidence < self._config['confidence_threshold']:
            summary = 'Config can be used to generate proper summary'

        response = ModelResponse(summary, confidence/100., confidence, raw_output)
        return response
        # or you can produce several responses, in this case you also need to provide judgement type
        #   and optionally subtype
        response_collection = ModelResponseCollection()
        response_collection.add(model_response=response, judgement_type="judgement_type")
        # responses can contain timeline
        timeline = Timeline("timeline_alias")
        timeline.add_element(TimelineElement(start_ns=1, end_ns=2, element_type="some_name", extra_fields={"value":"some_val"}))
        timeline.add_element(TimelineElement(start_ns=3, end_ns=4, element_type="some_name"))
        response_collection.add(
            model_response=ModelResponseTimeline(
                timeline=timeline,
                summary=summary,
                score=confidence/100.,
                confidence=confidence
            ),
            judgement_type="judgement_type"
        )
        return response_collection
