import time
import logging
import os

# Basic logging module
def create_logger(cfg):
    # root_output_dir = Path(cfg.log_dir)
    # set up logger
    os.makedirs(cfg.log_dir, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    phase = 'train'
    log_file = '{}_{}_{}.log'.format(time_str, cfg.model_type, phase)
    final_log_file = os.path.join(cfg.log_dir, log_file)
    # root_output_dir / log_file

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = os.path.join(cfg.log_dir, 'tensorboard' + '_' + time_str + '_' + cfg.model_type)
    # Path(cfg.log_dir) / ('tensorboard' + '_' + time_str + '_' + cfg.model_type)

    print('=> creating {}'.format(tensorboard_log_dir))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    # tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(tensorboard_log_dir)