_base_ = ['./mask2former_r50_8xb2-lsj-50e_coco.py']

num_things_classes = 8 # wheelchair and cane
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
image_size = (1024, 1024)


# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'CocoDataset'
data_root = 'data/avaChallenge/'
print('current data root is:', data_root)

# metainfo = {
#     'classes': ('nondisabledped', 'wheelchairped','visuallyimpairedped','rider','fourwheelveh','twowheelveh','cane','wheelchair')
# }



train_dataloader = dict(
    batch_size=1,
    # metainfo = metainfo,
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/train_xworld.json',
        data_prefix=dict(img='train_xworld/'),
        pipeline=train_pipeline))
val_dataloader = dict(
    # metainfo = metainfo,
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/val_xworld.json',
        data_prefix=dict(img='val_xworld/'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'annotations/val_xworld.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args={{_base_.backend_args}})
test_evaluator = val_evaluator

load_from = 'checkpoints\mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth'