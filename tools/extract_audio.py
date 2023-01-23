import argparse
import subprocess
import os
import glob


def extract_wav(video_path, audio_path, sr, v='error'):
    subprocess.call([
        'ffmpeg', '-i', video_path,
        *(['-v', v, '-stats'] if v else []),
        '-vn', '-acodec', 'pcm_s16le',
        '-ac', '1', '-ar', str(sr),
        audio_path
    ])

def load_audio(path, sr):
    import librosa
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def main(video_dir='videos', audio_dir='audio', h5_file=None, sr=24000):
    if video_dir:
        os.makedirs(audio_dir, exist_ok=True)
        for vf in glob.glob(os.path.join(video_dir, '**/*.MP4'), recursive=True):
            name = os.path.basename(vf)
            name = os.path.splitext(name)[0]
            af = os.path.join(audio_dir, f'{name}.wav')
            print(vf, '->', af)
            extract_wav(vf, af, sr)

    if h5_file:
        import h5py

        futures = []
        with h5py.File(h5_file, 'w') as hf, \
             concurrent.futures.ProcessPoolExecutor() as pool:
            for af in glob.glob(os.path.join(audio_dir, '**/*.wav'), recursive=True):
                fut = pool.submit(load_audio, af, sr)
                futures[fut] = os.path.basename(af).split('.')[0]

            for future in concurrent.futures.as_completed(futures):
                try:
                    y, sr = future.result()
                    d = hf.create_dataset(futures[future], data=y)
                    d.attrs['sr'] = sr
                except Exception as e:
                    print(f'{url} generated an exception: {e}')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
    
