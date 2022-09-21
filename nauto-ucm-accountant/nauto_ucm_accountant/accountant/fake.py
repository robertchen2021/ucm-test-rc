import random
from datetime import timedelta, date
import pathlib
import json
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import os


class Faker(object):
    def __init__(self, seed: int = 42):
        random.seed(seed)

    def create_fake_data(self, target: pathlib.Path):
        target.unlink()
        tmp_dir = pathlib.Path(TemporaryDirectory().name)
        tmp_dir.mkdir()
        self._create_good_model(tmp_dir)
        self._create_bad_model(tmp_dir)
        self._create_garbage_model(tmp_dir)
        self._create_index(tmp_dir)
        self._finalize(tmp_dir, target)

    def _create_good_model(self, target: pathlib.Path):
        self._create_model_version_stats(
            target / "Good model" / "alpha",
            num_cases=200,
            time_start=date(year=2020, month=1, day=1),
            time_end=date(year=2020, month=1, day=8),
            score_mean_positive=.65,
            score_std_positive=.4,
            score_mean_negative=.3,
            score_std_negative=.55,
            prevalence=.8,
            threshold=.2,
            coverage=.9
        )
        self._create_model_version_stats(
            target / "Good model" / "beta",
            num_cases=500,
            time_start=date(year=2020, month=1, day=9),
            time_end=date(year=2020, month=1, day=15),
            score_mean_positive=.65,
            score_std_positive=.35,
            score_mean_negative=.3,
            score_std_negative=.5,
            prevalence=.75,
            threshold=.6,
            coverage=.94
        )
        self._create_model_version_stats(
            target / "Good model" / "prod",
            num_cases=70,
            time_start=date(year=2020, month=1, day=15),
            time_end=date(year=2020, month=1, day=28),
            score_mean_positive=.75,
            score_std_positive=.3,
            score_mean_negative=.24,
            score_std_negative=.4,
            prevalence=.8,
            threshold=.55,
            coverage=.99
        )

    def _create_bad_model(self, target: pathlib.Path):
        self._create_model_version_stats(
            target / "Bad model" / "1",
            num_cases=200,
            time_start=date(year=2020, month=1, day=1),
            time_end=date(year=2020, month=1, day=5),
            score_mean_positive=.55,
            score_std_positive=.5,
            score_mean_negative=.35,
            score_std_negative=.3,
            prevalence=.15,
            threshold=.1,
            coverage=.7
        )
        self._create_model_version_stats(
            target / "Bad model" / "1.1",
            num_cases=400,
            time_start=date(year=2020, month=1, day=5),
            time_end=date(year=2020, month=1, day=11),
            score_mean_positive=.55,
            score_std_positive=.5,
            score_mean_negative=.35,
            score_std_negative=.3,
            prevalence=.15,
            threshold=.5,
            coverage=.8
        )
        self._create_model_version_stats(
            target / "Bad model" / "1.2",
            num_cases=300,
            time_start=date(year=2020, month=1, day=11),
            time_end=date(year=2020, month=1, day=17),
            score_mean_positive=.55,
            score_std_positive=.3,
            score_mean_negative=.35,
            score_std_negative=.2,
            prevalence=.15,
            threshold=.45,
            coverage=.8
        )
        self._create_model_version_stats(
            target / "Bad model" / "2",
            num_cases=250,
            time_start=date(year=2020, month=1, day=17),
            time_end=date(year=2020, month=1, day=29),
            score_mean_positive=.51,
            score_std_positive=.2,
            score_mean_negative=.47,
            score_std_negative=.2,
            prevalence=.75,
            threshold=.49,
            coverage=.8
        )

    def _create_garbage_model(self, target: pathlib.Path):
        self._create_model_version_stats(
            target / "Garbage model" / "1",
            num_cases=1000,
            time_start=date(year=2020, month=1, day=1),
            time_end=date(year=2020, month=1, day=25),
            score_mean_positive=.7,
            score_std_positive=.6,
            score_mean_negative=.7,
            score_std_negative=.6,
            prevalence=.95,
            threshold=.65,
            coverage=.6
        )

    def _create_model_version_stats(
            self,
            target: pathlib.Path,
            num_cases: int,
            time_start: date,
            time_end: date,
            score_mean_positive: float,
            score_std_positive: float,
            score_mean_negative: float,
            score_std_negative: float,
            prevalence: float = .5,
            threshold: float = .5,
            coverage: float=1.
    ):
        target.mkdir(parents=True, exist_ok=True)
        time_range_days = (time_end - time_start).days
        cases_per_day = [0] * time_range_days
        for i in range(num_cases):
            cases_per_day[random.randint(0, time_range_days - 1)] += 1
        for day_index in range(time_range_days):
            current_date = time_start + timedelta(days=day_index)
            true_labels = []
            predicted_scores = []
            predicted_labels = []
            for i in range(cases_per_day[day_index]):
                if random.random() < coverage:
                    true_labels.append(random.random() < prevalence)
                    if true_labels[-1]:
                        score = random.gauss(score_mean_positive, score_std_positive)
                    else:
                        score = random.gauss(score_mean_negative, score_std_negative)
                    predicted_scores.append(min(max(score, 0.), 1.))
                    predicted_labels.append(predicted_scores[-1] > threshold)
                    pathlib.Path(target).mkdir(parents=True, exist_ok=True)
                    with open(target / current_date.isoformat(), "wb") as fh:
                        data = {
                            "true_labels": true_labels,
                            "predicted_labels": predicted_labels,
                            "predicted_scores": predicted_scores,
                            "total_cases": cases_per_day[day_index]
                        }
                        serialized_data = bytes(json.dumps(data), "ascii")
                        fh.write(serialized_data)

    def _create_index(self, target: pathlib.Path):
        with open(target / "index.json", "w") as fh:
            fh.write(json.dumps({
                "models": [
                    {
                        "name": "Good model",
                        "versions": ["alpha", "beta", "prod"],
                    },
                    {
                        "name": "Bad model",
                        "versions": ["1", "1.1", "1.2", "2"],
                    },
                    {
                        "name": "Garbage model",
                        "versions": ["1"],
                    },
                ],
            }))

    def _finalize(self, tmp_dir: pathlib.Path, target: pathlib.Path):
        zipfile = ZipFile(target, 'w', ZIP_DEFLATED)
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                zipfile.write(
                    filename=os.path.join(root, file),
                    arcname=os.path.join(root, file).replace(str(tmp_dir) + "/", "")
                )
        zipfile.close()
