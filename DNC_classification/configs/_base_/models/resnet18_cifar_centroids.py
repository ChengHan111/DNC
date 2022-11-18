# model settings # change types here
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SubCentroids_Head_Formal',
        num_classes=10,
        in_channels=512,                                                                                                                                                                                                                 
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

    