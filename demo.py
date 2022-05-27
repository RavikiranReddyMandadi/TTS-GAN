from datasets.text_processing import *
from models.tacotron2 import get_pretrained_tacotron2
from models.wavenet import get_pretrained_wavenet_flow
from datasets.cmudict import CMUDict

import torch
import soundfile
import scipy.io.wavfile

cmudict_path = 'data/cmu_dictionary'
cmu_dict = CMUDict(cmudict_path)
text = "Hello world, I missed you so much."
lj_sampling_rate = 22050

sequences = encode_text(text)

tacotron2_weight_path = 'pretrained_weights/tacotron2_fp32.pt'
tacotron2 = get_pretrained_tacotron2(tacotron2_weight_path)
tacotron2 = tacotron2.cuda()
tacotron2.eval()

wn_weight_path = 'pretrained_weights/flow_wavenet_fp32.pt'
wn = get_pretrained_wavenet_flow(wn_weight_path)
wn = wn.remove_weightnorm(wn)
wn = wn.cuda()
wn.eval()

# Convert data to tensor
sequence_length = len(sequences)
sequence_length = torch.tensor(sequence_length).unsqueeze(0).cuda().long()
sequences = torch.tensor(sequences).unsqueeze(0).cuda().long()
print(sequences.shape, sequence_length)

# generate audio
with torch.no_grad():
    mel, mel_lengths, alignments = tacotron2.infer(sequences, sequence_length)
    audio = wn.infer(mel)

    print(mel.shape)
    print(mel_lengths.shape)
    print(alignments.shape)
    print(audio.shape)

audio_numpy = audio[0].data.cpu().numpy()
soundfile.write('soundfile.wav', audio_numpy, lj_sampling_rate, 'PCM_24')
scipy.io.wavfile.write('scipy.wav', lj_sampling_rate, audio_numpy)