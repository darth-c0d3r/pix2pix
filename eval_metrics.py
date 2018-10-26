import numpy as np
from skimage.measure import compare_ssim as ssim

def RootMeanSquareDifference(image_out, image_ref):
	assert image_out.shape == image_ref.shape, "Eval error: Image Size mismatch"
	return np.linalg.norm(image_out - image_ref)/(255.0*image_ref.size)

def StructuralSimilarityIndex(image_out, image_ref):
	assert image_out.shape == image_ref.shape, "Eval error: Image Size mismatch"
	return ssim(image_out, image_ref, multichannel=True)

