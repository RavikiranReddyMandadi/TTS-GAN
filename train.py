import argparse
import pprint
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.ljspeech import LJSpeechTextMEL, LJCollate
from utils.text_processing import symbols
from models.flownet import FlowGenerator
from models.losses import duration_loss, mle_loss
from utils.utils import Adam, clip_grad_value_, plot_spectrogram_to_numpy
from utils.logger import create_logger

def parse_args():
    parser = argparse.ArgumentParser('Gan Based TTS')
    parser.add_argument('--model_type', type=str, default='flowgan', help='model name')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log Dir')
    parser.add_argument('--model_dir', type=str, default='weights', help='Weight save dir')
    
    # Data
    parser.add_argument('--metadata_file', type=str, default='data/LJSpeech-1.1/metadata.csv', help='metadata file name')
    parser.add_argument('--data_root', type=str, default='data/LJSpeech-1.1', help='Data root path')
    parser.add_argument('--cmudict_path', type=str, default='data/cmu_dictionary', help='CMU Dictionary File')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--add_noise', type=bool, default=False, help='Add noise')
    parser.add_argument('--filter_length', type=int, default=1024, help='stft filter length')
    parser.add_argument('--hop_length', type=int, default=256, help='Hop length')
    parser.add_argument('--win_length', type=int, default=1024, help='Window length')
    parser.add_argument('--sampling_rate', type=int, default=22050, help='LJ Speech Sampling Rate')
    parser.add_argument('--n_mel_channels', type=int, default=80, help='MEL Channel')
    parser.add_argument('--mel_fmin', type=float, default=0.0, help='MEL Channel')
    parser.add_argument('--mel_fmax', type=float, default=8000.0, help='MEL Channel')

    # Model
    parser.add_argument('--hidden_channels', type=int, default=192, help='MEL Channel')
    parser.add_argument('--filter_channels', type=int, default=768, help='MEL Channel')
    parser.add_argument('--filter_channels_dp', type=int, default=256, help='MEL Channel')
    parser.add_argument('--kernel_size', type=int, default=3, help='MEL Channel')
    parser.add_argument('--p_dropout', type=float, default=0.1, help='MEL Channel')
    parser.add_argument('--n_blocks_dec', type=int, default=12, help='MEL Channel')
    parser.add_argument('--n_layers_enc', type=int, default=6, help='MEL Channel')
    parser.add_argument('--n_heads', type=int, default=2, help='MEL Channel')
    parser.add_argument('--p_dropout_dec', type=float, default=0.05, help='MEL Channel')
    parser.add_argument('--dilation_rate', type=float, default=1.0, help='MEL Channel')
    parser.add_argument('--kernel_size_dec', type=int, default=5, help='MEL Channel')
    parser.add_argument('--n_block_layers', type=int, default=4, help='MEL Channel')
    parser.add_argument('--n_sqz', type=int, default=2, help='MEL Channel')
    parser.add_argument('--prenet', type=bool, default=True, help='MEL Channel')
    parser.add_argument('--mean_only', type=bool, default=True, help='MEL Channel')
    parser.add_argument('--hidden_channels_enc', type=int, default=192, help='MEL Channel')
    parser.add_argument('--hidden_channels_dec', type=int, default=192, help='MEL Channel')
    parser.add_argument('--window_size', type=int, default=4, help='MEL Channel')

    # Training Params
    parser.add_argument('--epochs', type=int, default=1000, help='Training Epoch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=400, help='warmup steps')
    parser.add_argument('--scheduler', type=str, default="noam", help='scheduler type')
    parser.add_argument('--log_interval', type=int, default=50, help='log interval duration')

    return parser.parse_args()

def main():
    opts = parse_args()

    os.makedirs(opts.model_dir, exist_ok=True)

    logger, tb_dir = create_logger(opts)
    writer = SummaryWriter(log_dir=tb_dir)

    logger.info(pprint.pformat(opts))

    # Define dataset
    torch.manual_seed(1)
    data = LJSpeechTextMEL(opts)
    indices = torch.randperm(len(data)).tolist()
    split_idx = int(0.2 * len(data))

    trainset = torch.utils.data.Subset(data, indices[:-split_idx])
    valset = torch.utils.data.Subset(data, indices[-split_idx:])

    collate_fn = LJCollate()
    train_loader = DataLoader(trainset, batch_size=opts.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, batch_size=opts.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True, collate_fn=collate_fn)

    logger.info('Train set length {}\t Eval set length {}'.format(len(trainset), len(valset)))

    # Define Model
    model = FlowGenerator(
        n_vocab=len(symbols),
        out_channels=opts.n_mel_channels,
        hidden_channels=opts.hidden_channels, 
        filter_channels=opts.filter_channels, 
        filter_channels_dp=opts.filter_channels_dp, 
        kernel_size=opts.kernel_size,
        p_dropout=opts.p_dropout,
        n_blocks_dec=opts.n_blocks_dec,
        n_layers_enc=opts.n_layers_enc,
        n_heads=opts.n_heads, 
        p_dropout_dec=opts.p_dropout_dec, 
        dilation_rate=opts.dilation_rate,
        kernel_size_dec=opts.kernel_size_dec,
        n_block_layers=opts.n_block_layers,
        n_sqz=opts.n_sqz,
        prenet=opts.prenet,
        mean_only=opts.mean_only,
        hidden_channels_enc=opts.hidden_channels_enc,
        hidden_channels_dec=opts.hidden_channels_dec,
        window_size=opts.window_size,   
        n_speakers=0, 
        gin_channels=0, 
        n_split=4,
        sigmoid_scale=False,
        block_length=None,
    ).cuda()

    optimizer =Adam(
        model.parameters(),
        scheduler=opts.scheduler,
        dim_model=opts.hidden_channels,
        warmup_steps=opts.warmup_steps,
        lr=opts.learning_rate,
        betas=[0.9, 0.98],
        eps=1e-9,
    )

    logger.info(model)

    # Start training
    logger.info('Starting training for {} epoches'.format(opts.epochs))

    train_step = 0
    eval_step = 0
    best_loss = np.inf

    for epoch in range(0, opts.epochs):

        # Training
        model.train()


        for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):

            # print(x.shape, x_lengths.shape, y.shape, y_lengths.shape)
            x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
            y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)

            optimizer.zero_grad()
            (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(x, x_lengths, y, y_lengths, gen=False)

            l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
            l_length = duration_loss(logw, logw_, x_lengths)

            loss = sum([l_mle, l_length])
            loss.backward()

            # Clip grad
            grad_norm = clip_grad_value_(model.parameters(), 5)
            optimizer.step()

            if batch_idx % opts.log_interval == 0:
                (y_gen, *_), *_ = model(x[:1], x_lengths[:1], gen=True)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()
                ))

                logger.info([x.item() for x in [l_mle, l_length]] + [train_step, optimizer.get_lr()])

                # Scalar
                writer.add_scalar("loss/train/g/total", loss, train_step)
                writer.add_scalar("loss/train/g/mle", l_mle, train_step)
                writer.add_scalar("loss/train/g/duration", l_length, train_step)
                writer.add_scalar("learning_rate", optimizer.get_lr(), train_step)
                writer.add_scalar("grad_norm", grad_norm, train_step)

                # Add Image
                writer.add_image("y_org", plot_spectrogram_to_numpy(y[0].data.cpu().numpy()), train_step, dataformats='HWC')
                writer.add_image("y_gen", plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()), train_step, dataformats='HWC')
                writer.add_image("attn", plot_spectrogram_to_numpy(attn[0,0].data.cpu().numpy()), train_step, dataformats='HWC')

            train_step += 1

        
        # Evaluation
        model.eval()
        loss_tot = 0
        loss_mle = 0
        loss_duration = 0

        with torch.no_grad():
            for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(val_loader):

                x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
                y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)

                (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(x, x_lengths, y, y_lengths, gen=False)
                l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
                l_length = duration_loss(logw, logw_, x_lengths)

                loss = sum([l_mle, l_length])

                loss_tot += loss.item()
                loss_mle += l_mle.item()
                loss_duration += l_length.item()


                if batch_idx % opts.log_interval == 0:
                    logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(x), len(val_loader.dataset),
                        100. * batch_idx / len(val_loader),
                        loss.item(),
                    ))

                    logger.info([x.item() for x in [l_mle, l_length]])

        
        loss_tot /= len(val_loader)
        loss_mle /= len(val_loader)
        loss_duration /= len(val_loader)

        # Scalar
        writer.add_scalar("loss/eval/g/total", loss_tot, eval_step)
        writer.add_scalar("loss/eval/g/mle", loss_mle, eval_step)
        writer.add_scalar("loss/eval/g/duration", loss_duration, eval_step)

        eval_step += 1


        # Save results
        if loss_tot < best_loss:
            best_loss = loss_tot

            ckpt = {
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch,
                'learning_rate': optimizer.get_lr(),
                'eval_loss': best_loss,
            }

            ckpt_path = os.path.join(opts.model_dir, 'best_model.pt')
            torch.save(ckpt, ckpt_path)
            logger.info('Model Saved with loss {}'.format(best_loss))

    logger.info('==>Training done!\nBest Loss: %.3f' % (best_loss))

if __name__ == '__main__':
    main()