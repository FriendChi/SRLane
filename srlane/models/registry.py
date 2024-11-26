import torch.nn as nn
from srlane.registry import Registry, build_from_cfg
'''
这是一种常见的模式，用于模块的注册。
这种模式允许开发者将自定义的类或函数注册到一个全局的字典或其他数据结构中，
从而使得这些类或函数可以在其他地方通过某种标识符（如类名）轻松地被引用和实例化。
'''
BACKBONES = Registry("backbones")
HEADS = Registry("heads")
NECKS = Registry("necks")
NETS = Registry("nets")


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbones(cfg):
    return build(cfg.backbone, BACKBONES, default_args=dict(cfg=cfg))


def build_neck(cfg):
    return build(cfg.neck, NECKS, default_args=dict(cfg=cfg))


def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))


def build_net(cfg):
    return build(cfg.net, NETS, default_args=dict(cfg=cfg))
