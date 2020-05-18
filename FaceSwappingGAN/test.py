import time
import numpy as np
import argparse
import cv2
import os

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

import model
import dataset
import criterion


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--img_size', type=int, default=128)

	parser.add_argument('--source_path', type=str, default='./')

	parser.add_argument('--target_path', type=str, default='./')
	parser.add_argument('--ckpt_path', type=str, default='./work')
	parser.add_argument('--output_path', type=str, default='test')
	args = parser.parse_args()

	generator = model.Globalgenerator(2, 2, 3, 3, 71, 3)
	print(f'Generator loaded')

	g_ckpt = torch.load(f'{args.ckpt_path}/generator_final.pth.tar')
	generator.load_state_dict(g_ckpt)
	generator = generator.cuda()
	print(f'Checkpoints Loaded: {args.ckpt_path}')

	testset = dataset.Leifeng_movie(
		source_img=args.source_path,
		lm_path=args.target_path,
		image_size=args.img_size
		)

	testloader = Data.DataLoader(
		testset,
		batch_size=1,
		shuffle=False,
		num_workers=1
		)

	num_batches = len(testloader)
	print('Iters: %d'%num_batches)

	generator.eval()
	with torch.no_grad():
		start = time.time()
		for batch_idx, (inputs, target_hmap) in enumerate(testloader):

			inputs = inputs.cuda()
			target_hmap = target_hmap.cuda()

			output, _ = generator(torch.cat((inputs, target_hmap), 1))

			speed = (time.time() - start) / (batch_idx + 1)
			remain_time = (num_batches - batch_idx - 1) * speed / 3600

			if not os.path.isdir('./test_data/' + args.output_path):
				os.mkdir('./test_data/' + args.output_path)

			cv2.imwrite('./test_data/' + args.output_path + '/%04d.jpg'%batch_idx, (output.cpu().detach().numpy()[0, :, :, :].transpose(1,2,0) + 1)*127.5)

			print('Batch: %d/%d Speed: %.2f Remaining time: %.2f hrs'%(
				batch_idx,
				num_batches,
				speed,
				remain_time
				))
