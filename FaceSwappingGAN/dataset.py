import numpy as np
import cv2
from glob import glob
import os
import os.path as path
import random
import pickle

import scipy.stats as st

import torch
import torch.utils.data as Data


def process_pts(line):
	line = line.replace(',', '')
	line = line.split(' ')
	fname = line[0]
	pts = line[1:-3]
	ang = line[-3:]
	ang = [float(i) for i in ang]
	ang = np.float32(ang)
	pts = [float(i) for i in pts]
	pts = np.float32(pts)
	pts = pts.reshape([-1,3])
	return fname, pts, ang

def plot_gaussian_kernel(pos, size=25):
	sigma = (size-1) / 6
	xx = np.linspace(-3,3,size)
	x, y = pos[0], pos[1]
	xbias = (x - (size-1)/2) / sigma
	x = xx + xbias
	ybias = (y - (size-1)/2) / sigma
	y = xx + ybias
	x = st.norm.pdf(x)
	y = st.norm.pdf(y)
	exp = np.outer(y,x)
	hmap = exp / exp.max()
	return hmap

def plot_gaussian(hmap, pos, size, ksize=25):
	x, y = pos[0]/(384/size), pos[1]/(384/size)
	x1 = int(np.floor(x - ksize//2))
	x2 = x1 + ksize
	y1 = int(np.floor(y - ksize//2))
	y2 = y1 + ksize
	x = x - x1 
	y = y - y1 
	kernel = plot_gaussian_kernel([x,y], size=ksize)

	kernel_x1 = kernel_y1 = 0
	kernel_x2 = kernel_y2 = ksize
	if x1<0:
		kernel_x1 = -x1
		x1 = 0

	if y1<0:
		kernel_y1 = -y1 
		y1 = 0

	if y2>size:
		kernel_y2 = ksize - (y2 - size)
		y2 = size

	if x2 > size:
		kernel_x2 = ksize - (x2 - size) 
		x2 = size

	# try:
	hmap[y1:y2, x1:x2] = kernel[kernel_y1:kernel_y2, kernel_x1:kernel_x2]
	# except Exception as e:
	# 	print(e)
	# 	print(y1,y2,x1,x2, kernel_y1,kernel_y2, kernel_x1, kernel_x2)

def get_hmap(pts, size=128):
	hmap = np.zeros([size, size, 68])
	for i in range(len(pts)):
		plot_gaussian(hmap[:,:,i], pts[i], size=size)
	return hmap

# def get_hmap(pts, size=256):
# 	pos = np.dstack(np.mgrid[0:size:1, 0:size:1])
# 	hmap = np.zeros([size, size, 68])
# 	for i, point in enumerate(pts):
# 		p_resize = point / 256 * size
# 		hmap[:, :, i] = st.multivariate_normal(mean=[p_resize[1], p_resize[0]], cov=16).pdf(pos)
# 	return hmap

def Seg2map(seg, size=128, interpolation=cv2.INTER_NEAREST):
	seg_new = np.zeros(seg.shape, dtype='float32')
	seg_new[seg > 7.5] = 1
	seg = np.copy(cv2.resize(seg_new, (size, size), interpolation=interpolation))
	return seg

def Cv2tensor(img):
	img = img.transpose(2, 0, 1)
	img = torch.from_numpy(img.astype(np.float32))
	return img

class Reenactset(Data.Dataset):
	def __init__(self, pkl_path='', img_path='', max_iter=80000, consistency_iter=3, image_size=128):
		super(Reenactset, self).__init__()
		self.img_path = img_path
		self.data = pickle.load(open(pkl_path, 'rb'))
		self.idx = list(self.data.keys())
		self.size = max_iter
		self.image_size = image_size
		self.consistency_iter = consistency_iter
		assert self.consistency_iter > 0

	def __getitem__(self, index):
		ID = random.choice(self.idx)
		samples = random.sample(self.data[ID], self.consistency_iter+1)
		source = samples[0]
		target = samples[1]
		mid_samples = samples[2:]

		source_name, _, _ = process_pts(source)
		target_name, pts, _ = process_pts(target)

		m_pts = []
		for m_s in mid_samples:
			_, m_pt, _ = process_pts(m_s)
			m_pts.append(m_pt)

		# pts = torch.from_numpy(pts[:, 0:2].astype(np.float32))
		# pts = torch.unsqueeze(pts, 0)
		m_hmaps = []
		hmap = Cv2tensor(get_hmap(pts, size=self.image_size))
		for m_pt in m_pts:
			m_hmaps.append(Cv2tensor(get_hmap(m_pt, size=self.image_size)))

		source_file = self.img_path + f'/img/{ID}/{source_name}'
		target_file = self.img_path + f'/img/{ID}/{target_name}'
		target_seg_file = self.img_path + f'/seg/{ID}/seg_{target_name}'

		source_img = cv2.imread(source_file)
		source_img = cv2.resize(source_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
		source_img = source_img / 127.5 - 1
		source_img = Cv2tensor(source_img)

		target_img = cv2.imread(target_file)
		target_img = cv2.resize(target_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
		target_img = target_img / 127.5 - 1
		target_img = Cv2tensor(target_img)

		# source_seg = cv2.imread(self.img_path + f'/seg/{ID}/seg_{source_name}')
		# source_seg = Seg2map(source_seg, size=self.image_size)
		# source_seg = Cv2tensor(source_seg)

		target_seg = cv2.imread(target_seg_file)
		target_seg = Seg2map(target_seg, size=self.image_size)
		target_seg = Cv2tensor(target_seg)

		return source_img, hmap, target_img, target_seg, m_hmaps

	def __len__(self):
		return self.size

class Reenactset_new(Data.Dataset):
	def __init__(self, img_path='', seg_path='', max_iter=80000, consistency_iter=3, image_size=256):
		super(Reenactset_new, self).__init__()
		self.img_path = img_path
		self.seg_path = seg_path
		self.data_list = sorted(glob(self.img_path + '/*.txt'))
		self.size = max_iter
		self.image_size = image_size
		self.consistency_iter = consistency_iter
		assert self.consistency_iter > 0

		for data in self.data_list:
			lineList = [line.rstrip('\n') for line in open(data, 'r')]
			if len(lineList)<(self.consistency_iter+1):
				self.data_list.remove(data)

	def __getitem__(self, index):
		while True:
			try:
				ID = random.choice(self.data_list)
				# ID = '/home/yy/FSGAN/ijbc_clean_no_align_v2/145.txt'
				lineList = [line.rstrip('\n') for line in open(ID, 'r')]
				samples = random.sample(lineList, self.consistency_iter+1)
				source = samples[0]
				target = samples[1]
				mid_samples = samples[2:]

				source_name, _, _ = process_pts(source)
				target_name, pts, _ = process_pts(target)

				source_file = self.img_path + f'/img/{path.basename(path.split(source_name)[0])}/{path.basename(source_name)}'
				target_file = self.img_path + f'/img/{path.basename(path.split(target_name)[0])}/{path.basename(target_name)}'
				target_seg_file = self.seg_path + f'/seg/{path.basename(path.split(target_name)[0])}/seg_{path.basename(target_name)}'[:-4] + '.png'
				if not os.path.isfile(source_file) and os.path.isfile(target_file) and os.path.isfile(target_seg_file):
					continue

				m_pts = []
				for m_s in mid_samples:
					_, m_pt, _ = process_pts(m_s)
					m_pts.append(m_pt)

				# pts = torch.from_numpy(pts[:, 0:2].astype(np.float32))
				# pts = torch.unsqueeze(pts, 0)
				m_hmaps = []
				hmap = Cv2tensor(get_hmap(pts, size=self.image_size))
				for m_pt in m_pts:
					m_hmaps.append(Cv2tensor(get_hmap(m_pt, size=self.image_size)))
			except:
				continue

			source_img = cv2.imread(source_file)
			source_img = cv2.resize(source_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
			source_img = source_img / 127.5 - 1
			source_img = Cv2tensor(source_img)

			target_img = cv2.imread(target_file)
			target_img = cv2.resize(target_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
			target_img = target_img / 127.5 - 1
			target_img = Cv2tensor(target_img)

			# source_seg = cv2.imread(self.img_path + f'/seg/{ID}/seg_{source_name}')
			# source_seg = Seg2map(source_seg, size=self.image_size)
			# source_seg = Cv2tensor(source_seg)

			target_seg = cv2.imread(target_seg_file)
			target_seg = Seg2map(target_seg, size=self.image_size)
			target_seg = Cv2tensor(target_seg)
			break

		return source_img, hmap, target_img, target_seg, m_hmaps

	def __len__(self):
		return self.size

def low_pass(data, cutoff):
	assert cutoff<data.shape[0]//2,'cutoff should be less than half seq'
	print('Apply low pass with stop band:',cutoff)
	for pt in range(data.shape[1]):
		for dim in range(data.shape[2]):
			pts1 = data[:,pt,dim]
			x = np.array(list(range(len(pts1))))

			fft1 = np.fft.fft(pts1)
			fft1_amp = np.fft.fftshift(fft1)
			fft1_amp = np.abs(fft1_amp)

			fft1[cutoff:-cutoff] = 0
			recover = np.fft.ifft(fft1)
			data[:,pt,dim] = recover
	return data
