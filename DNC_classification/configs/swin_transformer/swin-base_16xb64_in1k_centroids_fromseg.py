_base_ = [
    '../_base_/models/swin_transformer/base_224_centroids_fromseg.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py', 
    '../_base_/default_runtime.py'
]
