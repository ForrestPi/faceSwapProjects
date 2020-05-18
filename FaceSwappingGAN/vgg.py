import torch
import torch.nn as nn


def ConvLayer2(in_channels, out_channels):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(True),
		nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(True),
		nn.MaxPool2d(kernel_size=2, stride=2)
		)

def ConvLayer4(in_channels, out_channels):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(True),
		nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(True),
		nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(True),
		nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(True),
		nn.MaxPool2d(kernel_size=2, stride=2)
		)

class VGG(nn.Module):

	def __init__(self, init_weights=True):
		super(VGG, self).__init__()
		self.layer64 = ConvLayer2(3, 64)
		self.layer128 = ConvLayer2(64, 128)
		self.layer256 = ConvLayer4(128, 256)
		self.layer512 = ConvLayer4(256, 512)
		# self.layer512_2 = ConvLayer4(512, 512)

		# self.linearlayer = nn.Sequential(
		# 	nn.Linear(8192, 4096, bias=False),
		# 	nn.BatchNorm1d(4096),
		# 	nn.ReLU(True),
		# 	nn.Linear(4096, 512)
		# 	)
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.layer64(x)
		fea_1 = x

		x = self.layer128(x)
		fea_2 = x

		x = self.layer256(x)
		fea_3 = x

		x = self.layer512(x)
		fea_4 = x

		# x = self.layer512_2(x)

		# x = x.view(-1, 8192)
		# x = self.linearlayer(x)

		return fea_1, fea_2, fea_3, fea_4

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm1d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
	model = VGG()
	model.eval()
	x = torch.ones(1, 3, 128, 128)
	x = model(x)
	print(x.shape)
	torch.save(model.state_dict(), 'vggface.pth')
