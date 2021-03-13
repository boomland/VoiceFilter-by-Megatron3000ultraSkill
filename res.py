import os
import glob
import torch
import librosa
import argparse

from utils.audio import Audio
from utils.hparams import HParam
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder

def main(args, hp):
    with open('out1.txt') as f:
        for line in f:
            res = line.split('\t')
            with torch.no_grad():
                model = VoiceFilter(hp)
                chkpt_model = torch.load(args.checkpoint_path, map_location='cpu')['model']
                model.load_state_dict(chkpt_model)
                model.eval()

                embedder = SpeechEmbedder(hp)
                chkpt_embed = torch.load(args.embedder_path, map_location='cpu')
                embedder.load_state_dict(chkpt_embed)
                embedder.eval()

                audio = Audio(hp)
                dvec_wav, _ = librosa.load(res[1], sr=16000)
                dvec_mel = audio.get_mel(dvec_wav)
                dvec_mel = torch.from_numpy(dvec_mel).float()
                dvec = embedder(dvec_mel)
                dvec = dvec.unsqueeze(0)

                mixed_wav, _ = librosa.load(res[0], sr=16000)
                mag, phase = audio.wav2spec(mixed_wav)
                mag = torch.from_numpy(mag).float()

                mag = mag.unsqueeze(0)
                mask = model(mag, dvec)
                est_mag = mag * mask

                est_mag = est_mag[0].cpu().detach().numpy()
                est_wav = audio.spec2wav(est_mag, phase)

                os.makedirs('/root/voicefilter/res', exist_ok=True)
                out_path = os.path.join('/root/voicefilter/res', f'{res[2]}')
                librosa.output.write_wav(out_path, est_wav, sr=16000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
 

    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)

