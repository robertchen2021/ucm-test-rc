from nauto_zoo import Model, ModelInput, ModelResponse
import os
import json
import tempfile
from .identify_no_motion import IdentifyNoMotion
import pprint
from shutil import rmtree


class NoMotionDetectorModel(Model):
    def run(self, model_input: ModelInput) -> ModelResponse:
        combined_json = model_input.get("sensor")
        temp_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(temp_dir, 'combined.json'), 'w') as f:
                json.dump(combined_json, f)
            identify_no_motion = IdentifyNoMotion(
                json_file_path=os.path.join(temp_dir, 'combined.json'),
                output_folder=temp_dir,
                logger=self._logger,
            )
            identify_no_motion.run()
            results = identify_no_motion.results
            if 'is_no_motion' in results:
                return ModelResponse(
                    summary="TRUE" if results['is_no_motion'] else "FALSE",
                    confidence=100,
                    score=0.,
                    raw_output=pprint.pformat(results, indent=2)
                )
            raise RuntimeError("This should have never happened, but `is_no_motion` was not set")
        finally:
            rmtree(temp_dir)
