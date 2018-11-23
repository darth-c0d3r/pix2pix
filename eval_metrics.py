import sys, os
from PIL import Image
import numpy as np
from skimage.measure import compare_ssim as ssim

def RootMeanSquareDeviation(image_out, image_ref):
	assert image_out.shape == image_ref.shape, "Eval error: Image Size mismatch"
	return np.linalg.norm(image_out - image_ref)/(255.0*(image_ref.size**0.5))

def StructuralSimilarityIndex(image_out, image_ref):
	assert image_out.shape == image_ref.shape, "Eval error: Image Size mismatch"
	return ssim(image_out, image_ref, multichannel=True)

root_dir = sys.argv[1]
ssim_list = []
rmsd_list = []
for idx in range(len(os.listdir(root_dir+'/target/'))):
	img1 = np.asarray(Image.open(root_dir+'/output/'+str(idx+1)+'.jpg'))
	img2 = np.asarray(Image.open(root_dir+'/target/'+str(idx+1)+'.jpg'))
	ssim_list.append(StructuralSimilarityIndex(img1,img2))
	rmsd_list.append(RootMeanSquareDeviation(img1,img2))

print('Avg. SSIM: '+ str(np.mean(ssim_list)))
print('Avg. RMSD: '+ str(np.mean(rmsd_list)))