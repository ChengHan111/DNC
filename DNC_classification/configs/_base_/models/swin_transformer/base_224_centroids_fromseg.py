# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
custom_imports=dict(imports='mmseg.models', allow_failed_imports=False)
model = dict(
    type='ImageClassifier',
    pretrained=None,
    backbone=dict(
        type='mmseg.SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SubCentroids_Head_Formal',
        num_classes=1000,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))