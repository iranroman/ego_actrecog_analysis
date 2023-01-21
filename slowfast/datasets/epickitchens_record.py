from .video_record import VideoRecord
from datetime import timedelta
import time


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
        timestamp.split('.')[-1]) / 100
    return sec


class EpicKitchensVideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]
        self.audio_sr = 24000

    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_time(self):
        return timestamp_to_sec(self._series['start_timestamp'])

    @property
    def stop_time(self):
        return timestamp_to_sec(self._series['stop_timestamp'])

    @property
    def start_frame(self):
        return int(round(timestamp_to_sec(self._series['start_timestamp']) * self.fps))

    @property
    def end_frame(self):
        return int(round(timestamp_to_sec(self._series['stop_timestamp']) * self.fps))

    @property
    def fps(self):
        is_100 = len(self.untrimmed_video_name.split('_')[1]) == 3
        return 50 if is_100 else 60

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def start_audio_sample(self):
        return int(round(timestamp_to_sec(self._series['start_timestamp']) * self.audio_sr))

    @property
    def end_audio_sample(self):
        return int(round(timestamp_to_sec(self._series['stop_timestamp']) * self.audio_sr))

    @property
    def num_audio_samples(self):
        return self.end_audio_sample - self.start_audio_sample

    @property
    def label(self):
        return {'verb': self._series.get('verb_class', -1),
                'noun': self._series.get('noun_class', -1)}

    @property
    def metadata(self):
        return {'narration_id': self._index}
