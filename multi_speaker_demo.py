import gradio as gr
from fairseq.checkpoint_utils import load_model_ensemble_and_task
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from utils.fairseq_infer import MultiSpeakerInference
import cv2

def draw_spectrogram(waveform, sample_rate):

    fig, ax = plt.subplots(figsize=(10, 2)) 
    plt.title('Spectrogram')
    S = librosa.feature.melspectrogram(y=waveform.cpu().numpy(), sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    # out = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    fig.canvas.draw()
    img = cv2.imread("a.png")
    return img

arg_overrides = {
    'vocoder': 'hifigan', 
    'fp16': False, 
    'data': 'weights/common-voice-en-200_speaker'
}

weight_path = 'weights/common-voice-en-200_speaker/pytorch_model.pt'
models, cfg, task = load_model_ensemble_and_task(
    [weight_path], arg_overrides
)
model = models[0].cuda()

multispeaker = MultiSpeakerInference(model, cfg, task)



def inference(text, speaker_idx):
    speaker_idx = min(max(int(speaker_idx), 0), 99)

    waveform, sample_rate = multispeaker.get_waveform(text, speaker_idx)
    # fig = draw_spectrogram(waveform, sample_rate)
    return sample_rate, waveform.squeeze().cpu().numpy()    #, fig

title = 'Multi Speakers TTS Demo'

gr.Interface(
    inference,
    [gr.inputs.Textbox(label="Input Text"), gr.inputs.Number(label="Enter Speaker I'D")],
    # ["audio", "image"],
    [gr.outputs.Audio(label="Output")],
    title=title
).launch(debug=True, enable_queue=True)