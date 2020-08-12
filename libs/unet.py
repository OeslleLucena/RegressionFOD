import torch.nn as nn
import torch


class ConvBnActivation(nn.Module):
	def __init__(self, in_channels, out_channels, activation,
				 kernel_size=3,  stride=1, bias=False):
		super(ConvBnActivation, self).__init__()
		padding = (kernel_size - 1) // 2
		self.conv = nn.Conv3d(in_channels,
							  out_channels,
							  kernel_size,
							  stride=stride,
							  padding=padding,
							  bias=bias)
		self.bnorm = nn.BatchNorm3d(out_channels)
		self.activation = activation

	def forward(self, x):
		x = self.conv(x)
		x = self.bnorm(x)
		x = self.activation(x)
		return x

class Encoder(nn.Module):
	def __init__(self, in_channel, out_channel, activation):
		super(Encoder, self).__init__()
		self.encode = nn.Sequential(
			ConvBnActivation(in_channel,
							 out_channel,
							 activation),
			ConvBnActivation(out_channel,
							 out_channel * 2,
							 activation),
		)
		self.max_pool = nn.MaxPool3d(2)

	def forward(self, x):
		x = self.encode(x)
		x_downsampled = self.max_pool(x)
		return x, x_downsampled


class Decoder(nn.Module):
	def __init__(self, base_filter, activation):
		super(Decoder, self).__init__()
		self.upsampling = nn.Upsample(scale_factor = 2,
									  mode='trilinear')
		self.decode = nn.Sequential(
			ConvBnActivation(base_filter + base_filter * 2,
							 base_filter,
							 activation),

			ConvBnActivation(base_filter,
							 base_filter,
							 activation)
		)

	def forward(self, x, down_tensor):
		x = self.upsampling(x)
		x = torch.cat([x, down_tensor], 1)
		x = self.decode(x)
		return x

class Unet3D(nn.Module):
	def __init__(self, num_input_channels,
				 num_output_channels,
				 depth_of_network = 2,
				 num_base_filters = 32,
				 activation = nn.ReLU(inplace=True)):
		super(Unet3D, self).__init__()
		self.num_input_channels = num_input_channels
		self.num_output_channels = num_output_channels
		self.activation = activation
		self.depth_of_network = depth_of_network
		self.num_base_filters = num_base_filters

		encoder_blocks = nn.ModuleList()

		for ii in range(self.depth_of_network):
			if ii == 0:
				# out_channel =  out_channel * 2
				in_channel = self.num_input_channels
				out_channel = self.num_base_filters
				encode_block = Encoder(in_channel,
									   out_channel,
									   self.activation)
			else:
				in_channel = self.num_base_filters * (2 ** ii)
				out_channel = self.num_base_filters * (2 ** ii)
				encode_block = Encoder(in_channel,
									   out_channel,
									   self.activation)

			encoder_blocks.append(encode_block)
		self.encoder = nn.Sequential(*encoder_blocks)

		# Latent Space
		out_channel = out_channel * 2

		self.latent = nn.Sequential(
			ConvBnActivation(out_channel, out_channel, self.activation),
			ConvBnActivation(out_channel, out_channel * 2, self.activation)
		)

		decoder_blocks = nn.ModuleList()
		for ii in range(self.depth_of_network):
			decoder_block = Decoder(out_channel, self.activation)
			decoder_blocks.append(decoder_block)
			out_channel = out_channel // 2

		self.decoder = nn.Sequential(*decoder_blocks)

		out_channel = out_channel * 2 # TODO: fix this repetition
		self.output_layer = nn.Sequential(
			nn.Conv3d(out_channel,
					  self.num_output_channels,
					  kernel_size=1,
					  bias=True),
		)

	def forward(self, x):
		# Encoder
		out_down_list = []
		for encoder in self.encoder:
			out_down, out = encoder(x)
			out_down_list.append(out_down)
			x = out

		out = self.latent(out)
		ii = -1
		for decoder in self.decoder:
			out = decoder(out, out_down_list[ii])
			ii = -2
		out = self.output_layer(out)
		return out
