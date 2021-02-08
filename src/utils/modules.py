import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ofa.utils.layers import set_layer_from_config, ZeroLayer
from ofa.utils import MyModule, MyNetwork, MyGlobalAvgPool2d, min_divisible_value, SEModule
from ofa.utils import get_same_padding, make_divisible, build_activation, init_models

__all__ = ['my_set_layer_from_config',
           'LiteResidualModule']


def my_set_layer_from_config(layer_config):
	if layer_config is None:
		return None
	name2layer = {
		LiteResidualModule.__name__: LiteResidualModule,
	}
	layer_name = layer_config.pop('name')
	if layer_name in name2layer:
		layer = name2layer[layer_name]
		return layer.build_from_config(layer_config)
	else:
		return set_layer_from_config({'name': layer_name, **layer_config})


class LiteResidualModule(MyModule):

	def __init__(self, main_branch, in_features, out_features,
	             expand=1.0, kernel_size=3, act_func='relu', n_groups=2,
	             downsample_ratio=2, upsample_type='linear', stride=1):
		super(LiteResidualModule, self).__init__()

		self.main_branch = main_branch
		pooling = nn.AvgPool1d(downsample_ratio)
		self.lite_residual_config = {
			'in_features': in_features,
			'out_features': out_features,
			'expand': expand,
			'kernel_size': kernel_size,
			'act_func': act_func,
			'groups': n_groups,
			'kernel_size': kernel_size,
			'downsample_ratio': downsample_ratio,
			'upsample_type': upsample_type,
		}

		kernel_size = 1 if downsample_ratio is None else kernel_size

		padding = get_same_padding(kernel_size)

		#pooling = nn.AvgPool1d(downsample_ratio)
		#num_mid = make_divisible(int(in_features * expand), divisor=MyNetwork.CHANNEL_DIVISIBLE)
		num_mid = 1024
		out_features = 1536
		'''

		self.lite_residual = nn.Sequential(OrderedDict({
			'conv1': nn.Conv1d(in_features, num_mid, kernel_size= kernel_size, padding= 2, bias=False),
			'bn1': nn.BatchNorm1d(num_mid),
			'act': build_activation(act_func),
			'conv2': nn.Conv1d(num_mid, out_features, kernel_size= kernel_size, padding= 2, bias=False),
			'final_bn': nn.BatchNorm1d(out_features),
		}))
		'''

		self.lite_residual = nn.Sequential(OrderedDict({
			'conv1': nn.Conv1d(in_features, out_features, kernel_size= kernel_size, padding= 2, bias=False),
			'final_bn': nn.BatchNorm1d(out_features),
		}))
		

		# initialize
		init_models(self.lite_residual)
		self.lite_residual.final_bn.weight.data.zero_()

	def forward(self, x):
		main_x = self.main_branch(x)
		x_re = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1])) #b_size, sentence_length, hidden_size
		lite_residual_x = self.lite_residual(x_re)
		lite_residual_x= lite_residual_x.permute(0,2,1)

		
		if self.lite_residual_config['downsample_ratio'] is not None:
			lite_residual_x = F.upsample(lite_residual_x, main_x.shape[2:],
			                             mode=self.lite_residual_config['upsample_type'])
		return main_x + lite_residual_x

	@property
	def module_str(self):
		return self.main_branch.module_str + ' + LiteResidual(downsample=%s, n_groups=%s, expand=%s, ks=%s)' % (
			self.lite_residual_config['downsample_ratio'], self.lite_residual_config['n_groups'],
			self.lite_residual_config['expand'], self.lite_residual_config['kernel_size'],
		)

	@property
	def config(self):
		return {
			'name': LiteResidualModule.__name__,
			'main': self.main_branch.config,
			'lite_residual': self.lite_residual_config,
		}

	@staticmethod
	def build_from_config(config):
		main_branch = my_set_layer_from_config(config['main'])
		lite_residual_module = LiteResidualModule(
			main_branch, **config['lite_residual']
		)
		return lite_residual_module

	def __repr__(self):
		return '{\n (main branch): ' + self.main_branch.__repr__() + ', ' + \
		       '\n (lite residual): ' + self.lite_residual.__repr__() + '}'

	@staticmethod
	def insert_lite_residual(net, downsample_ratio=2, upsample_type='linear',
	                         expand=1.0, max_kernel_size=5, act_func='relu', n_groups=2,
	                         **kwargs):
		if LiteResidualModule.has_lite_residual_module(net):
			# skip if already has lite residual modules
			return
		from transformers import DistilBertForSequenceClassification, MobileBertForSequenceClassification

		if isinstance(net, DistilBertForSequenceClassification):
			
			block_downsample_ratio = downsample_ratio
			tsf_module = net.distilbert.transformer.layer
			for i, layer_module in enumerate(tsf_module):
				ffn_module=layer_module.ffn
				linear_1= ffn_module.lin1
				linear_2= ffn_module.lin2
				block = linear_1
				if ffn_module.lin1:
					block_downsample_ratio = downsample_ratio
					ffn_module.lin1 = LiteResidualModule(
						block, block.in_features, block.out_features, expand=expand, kernel_size=max_kernel_size,
						act_func=act_func, n_groups=n_groups, downsample_ratio=block_downsample_ratio,
						upsample_type=upsample_type,
					)
		elif isinstance(net, MobileBertForSequenceClassification):
			tsf_module = net.mobilebert.transformer.layer
		else:
			raise NotImplementedError

	@staticmethod
	def has_lite_residual_module(net):
		for m in net.modules():
			if isinstance(m, LiteResidualModule):
				return True
		return False

	@property
	def in_features(self):
		return self.lite_residual_config['in_features']

	@property
	def out_features(self):
		return self.lite_residual_config['out_features']

