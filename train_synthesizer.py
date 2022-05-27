import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse
import os
import itertools
import pprint
import time
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from models.losses import discriminator_loss, feature_loss, generator_loss
from datasets.meldataset import get_dataset_filelist, MelDataset, mel_spectrogram
from utils.logger import create_logger
from utils.utils import plot_spectrogram

def parse_args():

    parser = argparse.ArgumentParser('Speech Sythesis')

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='data/LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='data/LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='data/LJSpeech-1.1/validation.txt')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    # Basic
    parser.add_argument('--model_type', type=str, default='style_mel_gan', help='model name')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log Dir')
    parser.add_argument('--model_dir', type=str, default='weights', help='Weight save dir')
    
    # Dataset
    parser.add_argument('--segment_size', default=8192, type=int)
    parser.add_argument('--n_fft', default=1024, type=int)
    parser.add_argument('--num_mels', default=80, type=int)
    parser.add_argument('--hop_size', default=256, type=int)
    parser.add_argument('--win_size', default=1024, type=int)
    parser.add_argument('--sampling_rate', default=22050, type=int)
    parser.add_argument('--fmin', default=0, type=int)
    parser.add_argument('--fmax', default=8000, type=int)
    parser.add_argument('--fmax_for_loss', default=None, type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # Training Params
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--adam_b1', default=0.8, type=float)
    parser.add_argument('--adam_b2', default=0.99, type=float)
    parser.add_argument('--lr_decay', default=0.999, type=int)
    parser.add_argument('--log_interval', type=int, default=50, help='log interval duration')



    return parser.parse_args()

def main():
    opts = parse_args()
    os.makedirs(opts.model_dir, exist_ok=True)

    logger, tb_dir = create_logger(opts)
    writer = SummaryWriter(log_dir=tb_dir)

    logger.info(pprint.pformat(opts))

    device = torch.device('cuda:0')
    last_epoch = -1

    # Define dataset
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    training_filelist, validation_filelist = get_dataset_filelist(opts)

    trainset = MelDataset(
        training_filelist, opts.segment_size, opts.n_fft, opts.num_mels,
        opts.hop_size, opts.win_size, opts.sampling_rate, opts.fmin,
        opts.fmax, n_cache_reuse=0, shuffle=True, fmax_loss=opts.fmax_for_loss, 
        device=device, fine_tuning=opts.fine_tuning, base_mels_path=opts.input_mels_dir
    )

    validset = MelDataset(
        validation_filelist, opts.segment_size, opts.n_fft, opts.num_mels,
        opts.hop_size, opts.win_size, opts.sampling_rate, opts.fmin,
        opts.fmax, True, False, n_cache_reuse=0, fmax_loss=opts.fmax_for_loss,
        device=device, fine_tuning=opts.fine_tuning, base_mels_path=opts.input_mels_dir
    )

    train_loader = DataLoader(
        trainset, num_workers=opts.num_workers, shuffle=False,
        batch_size=opts.batch_size, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        validset, num_workers=opts.num_workers, shuffle=False,
        batch_size=opts.batch_size, pin_memory=True, drop_last=True
    )


    # Define Models
    netG = Generator().to(device)
    netDP = MultiPeriodDiscriminator().to(device)
    netDS = MultiScaleDiscriminator().to(device)


    # Model Parameters
    optim_g = torch.optim.AdamW(netG.parameters(), opts.learning_rate, betas=[opts.adam_b1, opts.adam_b2])
    optim_d = torch.optim.AdamW(
        itertools.chain(netDP.parameters(), netDS.parameters()),
        opts.learning_rate, betas=[opts.adam_b1, opts.adam_b2]
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=opts.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=opts.lr_decay, last_epoch=last_epoch)

    logger.info('Starting training for {} epoches'.format(opts.training_epochs))

    train_step = 0
    eval_step = 0
    best_loss = np.inf

    for epoch in range(max(0, last_epoch), opts.training_epochs):

        netG.train()
        netDP.train()
        netDS.train()

        for batch_idx, batch in enumerate(train_loader):

            start_b = time.time()

            x, y, _, y_mel = batch
            x, y, y_mel = x.cuda(), y.cuda().unsqueeze(1), y_mel.cuda()
            
            # Discriminator Training
            y_g_hat = netG(x)

            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), opts.n_fft, opts.num_mels, opts.sampling_rate,
                opts.hop_size, opts.win_size, opts.fmin, opts.fmax_for_loss
            )

            optim_d.zero_grad()

            # Peroid
            y_df_hat_r, y_df_hat_g, _, _ = netDP(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            
            # Scale
            y_ds_hat_r, y_ds_hat_g, _, _ = netDS(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator Training
            optim_g.zero_grad()

            # L1 Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            # Feature Loss + Generator Loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = netDP(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = netDS(y, y_g_hat)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)

            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()


            # Log
            if batch_idx % opts.log_interval == 0:

                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                msg = 'Epoch : {}, Train Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(
                    epoch, train_step, loss_gen_all, mel_error, time.time() - start_b
                )

                logger.info(msg)
                writer.add_scalar("training/gen_loss_total", loss_gen_all, train_step)
                writer.add_scalar("training/mel_spec_error", mel_error, train_step)

            train_step += 1

        scheduler_g.step()
        scheduler_d.step()

        # Evaluation
        netG.eval()
        val_err_tot = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                start_b = time.time()

                x, y, _, y_mel = batch
                x, y, y_mel = x.cuda(), y.cuda().unsqueeze(1), y_mel.cuda()

                y_g_hat = netG(x)

                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1), opts.n_fft, opts.num_mels, opts.sampling_rate,
                    opts.hop_size, opts.win_size, opts.fmin, opts.fmax_for_loss,
                )

                val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                if batch_idx <= 4:
                    if eval_step == 0:
                        # add ground truth to TB
                        writer.add_audio('gt/y_{}'.format(batch_idx), y[0], eval_step, opts.sampling_rate)
                        writer.add_figure('gt/y_spec_{}'.format(batch_idx), plot_spectrogram(x[0].cpu().numpy()), eval_step)

                    writer.add_audio('generated/y_hat_{}'.format(batch_idx), y_g_hat[0], eval_step, opts.sampling_rate)
                    y_hat_spec = mel_spectrogram(
                        y_g_hat.squeeze(1), opts.n_fft, opts.num_mels, opts.sampling_rate,
                        opts.hop_size, opts.win_size, opts.fmin, opts.fmax
                    )
                    writer.add_figure('generated/y_hat_spec_{}'.format(batch_idx),
                                    plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), eval_step)

        eval_step += 1  

        # Log
        
        val_err = val_err_tot / len(val_loader)
        writer.add_scalar("validation/mel_spec_error", val_err, eval_step)

        msg = 'Eval Epoch : {:d}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(epoch, val_err, time.time() - start_b)
        logger.info(msg)

        # Save results
        if val_err < best_loss:
            best_loss = val_err

            ckpt = {
                'netG': netG.state_dict(),
                'netDP': netDP.state_dict(),
                'netDS': netDS.state_dict(),
                'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
                'best_loss': best_loss,
                'epoch': epoch,
            }

            ckpt_path = os.path.join(opts.model_dir, opts.model_type + '_best_model.pt')
            torch.save(ckpt, ckpt_path)
            logger.info('Model Saved with loss {}'.format(best_loss))

    logger.info('==>Training done!\nBest Loss: %.3f' % (best_loss))


if __name__ == '__main__':
    main()