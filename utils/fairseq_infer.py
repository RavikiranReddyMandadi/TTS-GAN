import torch
from torch import nn

import g2p_en

# Inference handler of multi speaker demo. 
class MultiSpeakerInference(nn.Module):

    def __init__(self, model, cfg, task):
        super().__init__()

        self.model = model
        self.cfg = cfg
        self.task = task

        self.model.eval()

        self.update_cfg_with_task_cfg(self.cfg, self.task.data_cfg)
        self.netG = self.task.build_generator([self.model], self.cfg)

    @classmethod
    def update_cfg_with_task_cfg(cls, cfg, task_cfg):
        cfg["task"].vocoder = task_cfg.vocoder.get("type", "griffin_lim")

    @classmethod
    def text_process(cls, text):

        # cmudict encode
        g2p = g2p_en.G2p()
        res = [{",": "sp", ";": "sp"}.get(p, p) for p in g2p(text)]
        return " ".join(p for p in res if p.isalnum())


    @classmethod
    def prepare_data_for_model(cls, task, text, speaker_idx):
        processed_text = cls.text_process(text)

        speaker = task.data_cfg.hub.get("speaker", speaker_idx)
        speaker = torch.Tensor([[speaker]]).long()

        src_tokens = task.src_dict.encode_line(
            processed_text, add_if_not_exist=False
        ).view(1, -1)

        src_lengths = torch.Tensor([len(processed_text.split())]).long()

        return {
            "net_input": {
                "src_tokens": src_tokens.cuda(),
                "src_lengths": src_lengths.cuda(),
                "prev_output_tokens": None,
            },
            "target_lengths": None,
            "speaker": speaker.cuda(),
        }

    @classmethod
    def get_speech(cls, task, model, netG, batch):
        prediction = netG.generate(model, batch)
        return prediction[0]["waveform"], task.sr

    def get_waveform(self, text, speaker_idx):
        batch = self.prepare_data_for_model(self.task, text, speaker_idx)

        return self.get_speech(self.task, self.model, self.netG, batch)