work_dir = '/home/faymek/MPCompress/data/test-fc/dinov2_vitg14_voc2012_linear'
data_root = "/home/faymek/MPCompress/data/dataset/VOC2012"
dataset_type = "PascalVOCDataset"

patch_size = 14
crop_size = (518, 518)
stride = (crop_size[0] // 2, crop_size[1] // 2)
# crop_size = (512, 512)
# stride = (341, 341)
data_preprocessor = dict(
    bgr_to_rgb=True,
    # mean=[0, 0, 0,],
    # std=[255, 255, 255,],
    mean=[123.675, 116.28, 103.53,],
    std=[58.395, 57.12, 57.375,],
    pad_val=0,
    seg_pad_val=255,
    # size=crop_size,
    type="SegDataPreProcessor",
)  # only pad when have test_cfg
default_hooks = dict(
    logger=dict(interval=100, log_metric_by_epoch=True, type="LoggerHook"),
    visualization=dict(type="SegVisualizationHook", draw=True, interval=1),
)
default_scope = "mmseg"
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
log_level = "INFO"
log_processor = dict(by_epoch=True)
norm_cfg = dict(requires_grad=True, type="SyncBN")

test_cfg = dict(type="TestLoop")
test_evaluator = dict(
    iou_metrics=[
        "mIoU",
    ],
    type="IoUMetric",
)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='Resize', scale=(99999999, crop_size[0]), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
test_dataloader = dict(
    batch_size=1,  # //4 if tta
    dataset=dict(
        type=dataset_type,
        ann_file="VOC2012_sel20.txt",
        data_prefix=dict(
            img_path="JPEGImages", seg_map_path="SegmentationClass"
        ),
        data_root=data_root,
        pipeline=test_pipeline,
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)


norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="DinoVisionBackbone",
        model_size="large",
        img_size=crop_size[0],
        patch_size=patch_size,
        out_indices=[39],
        final_norm=False,
        checkpoint="/home/faymek/MPCompress/data/models/backbone/dinov2_vitg14_pretrain.pth",
    ),
    decode_head=dict(
        type="PretrainedBNHead",
        in_channels=[1536],
        in_index=[0],   # the first index of the extracted features. Since there is only 1 layer, 0 is used.
        input_transform="resize_concat",
        channels=1536,
        dropout_ratio=0,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        patch_size=patch_size,
        checkpoint="/home/faymek/MPCompress/data/models/seg_head/dinov2_vitg14_voc2012_linear_head.pth",
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=stride),
    # test_cfg=dict(mode="whole"),
    data_preprocessor=data_preprocessor,
)
vis_backends = [
    dict(type="LocalVisBackend", save_dir=work_dir),
]
visualizer = dict(
    name="visualizer", type="SegLocalVisualizer", vis_backends=vis_backends
)
gpu_ids = range(0, 1)