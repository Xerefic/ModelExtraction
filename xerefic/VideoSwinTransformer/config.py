from dataclasses import dataclass, field

@dataclass
class TrainingArgs():

    seed: int = 420
    lr: float = 1.e-3
    batch_size: int = 8
    num_workers: int = 2
    max_epochs: str = 1000

    image_size: int = 224
    temporal_history: int = 16

    train_file: str = '/home/ubuntu/ModelExtraction/data/Moments_in_Time/rawframes/training/'
    valid_file: str = None
    test_file: str = None
    checkpoint: str = None

    project_name: str = None
    wandb_run_name: str = None

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
    transform = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='RandomResizedCrop'),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='ToTensor', keys=['imgs']),
        dict(type='Normalize', **img_norm_cfg),
        ]