import os
import numpy as np
from speechbrain.pretrained import EncoderClassifier
import torchaudio
from sklearn.metrics.pairwise import cosine_similarity

# Specify the validation trial pairs
in_file = 'data/verif/VoxSRC2023_val.txt'
out_file = 'data/verif/VoxSRC2023_val_score.txt'

# Specify the path for audio files
audio_dir = 'VoxSRC2023_val/'

# Extracting embeddings
# You could use 'SpeakerRecognition' module and the function 'verify_files', but I found this is much faster.
# This model is trained with VoxCeleb1 and 2. Thus, finetuning this model is not allowed in Track 1.
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})

embeddings = {}
with open(in_file, 'r') as f:
    with open(out_file, 'w') as f_out:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            gt, wav1, wav2 = line.split(' ')
            if wav1 not in embeddings:
                signal, fs = torchaudio.load(os.path.join(audio_dir, wav1))
                emb = classifier.encode_batch(signal).cpu().numpy()
                embeddings[wav1] = np.squeeze(emb, axis=1)
            if wav2 not in embeddings:
                signal, fs = torchaudio.load(os.path.join(audio_dir, wav2))
                emb = classifier.encode_batch(signal).cpu().numpy()
                embeddings[wav2] = np.squeeze(emb, axis=1)
            score = cosine_similarity(embeddings[wav1], embeddings[wav2])
            newline = f'{score[0][0]:.3f} {wav1} {wav2}\n'
            f_out.write(newline)