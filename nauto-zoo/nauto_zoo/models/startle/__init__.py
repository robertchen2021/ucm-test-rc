from nauto_zoo import Model, ModelInput, ModelResponseTimeline
from .startle import process_one_event, get_sensors_from_combined_recordings


class StartleDetectorModel(Model):
    def run(self, model_input: ModelInput) -> ModelResponseTimeline:
        timeline = process_one_event(model_input.sensor)
        return ModelResponseTimeline(
            summary="FALSE" if timeline.is_empty() else "TRUE",
            timeline=timeline
        )
