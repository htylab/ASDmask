import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from nilearn import plotting, input_data, image
import nibabel as nib
def get_connindex(rsn10_ff,maskrns10_ff, zthr=4):
    img_data= nib.load(rsn10_ff).get_data()
    maskrsn10 = nib.load(maskrns10_ff).get_data()
    connindex= np.zeros(10)
    temp = ((maskrsn10 > zthr) * img_data).reshape((-1,10))
    for ii in range(10):
        temp1= temp[:,ii]
        connindex[ii] =np.nanmean(temp1[np.nonzero(temp1)])
    print('Connindex:'+' '.join(['%.5f' % ii for ii in connindex.tolist()]))
    return connindex

def seed_to_voxel_corr(func_filename,confound_filename, resultdir = '.',
                       seed_coord = [(0, -52, 18)], output_head = 'DMN',
                       maps_img ='', mask_img = None):
    #print(func_filename)
    #print(confound_filename)
    mask_or_seed = 0
    if maps_img == '':
        mask_or_seed = 1
        seed_masker = input_data.NiftiSpheresMasker(
            seed_coord, radius=8,
            detrend=True, standardize=True, mask_img =mask_img, verbose=0)
            #memory='nilearn_cache', memory_level=1, verbose=0)
    else:
        mask_or_seed = 0
        seed_masker = input_data.NiftiMapsMasker(maps_img=[maps_img],
                                                 standardize=True,
                                                 mask_img=mask_img,
                                                 verbose=0)
                                                 #memory='nilearn_cache',


    seed_time_series = seed_masker.fit_transform(func_filename,
                                                 confounds=[confound_filename])

    brain_masker = input_data.NiftiMasker(
        smoothing_fwhm=6,
        detrend=True, standardize=True,
        memory_level=1, verbose=0)

    brain_time_series = brain_masker.fit_transform(func_filename,
                                                   confounds=[confound_filename])
    '''
    fig = plt.figure(figsize=(6,3), dpi=300)
    plt.plot(seed_time_series)
    plt.title('Seed time series')
    plt.xlabel('Scan number')
    plt.ylabel('Normalized signal')
    plt.tight_layout()
    fig.savefig(join(resultdir,'%s_curve.png' % output_head), bbox_inches='tight')
    '''

    seed_based_correlations = np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0]


    print("seed-based correlation shape: (%s, %s)" % seed_based_correlations.shape)
    print("seed-based correlation: min = %.3f; max = %.3f" % (
        seed_based_correlations.min(), seed_based_correlations.max()))

    seed_based_correlations_fisher_z = np.arctanh(seed_based_correlations)
    print("seed-based correlation Fisher-z transformed: min = %.3f; max = %.3f" % (
        seed_based_correlations_fisher_z.min(),
        seed_based_correlations_fisher_z.max()))

    # Finally, we can tranform the correlation array back to a Nifti image
    # object, that we can save.
    seed_based_corr_img = brain_masker.inverse_transform(seed_based_correlations.T)
    seed_based_corr_img.to_filename(join(resultdir,'%s_z.nii.gz') % output_head)
    '''
    display = plotting.plot_stat_map(seed_based_corr_img, threshold=0.3,
                                     cut_coords=(0,0,0), draw_cross=False)
    if mask_or_seed == 1:
        display.add_markers(marker_coords=seed_coord, marker_color='g',
                            marker_size=20)
    # At last, we save the plot as pdf.
    #display.savefig(join(resultdir,'%s_z.pdf') % output_head)
    display.savefig(join(resultdir,'%s_z.jpg') % output_head)
    plt.close()
    '''
    return join(resultdir,'%s_z.nii.gz') % output_head


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.uint8)
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output


def plotrsn10(nii_ff,workdir,resultjpg_ff, threshold):
    '''
    plotrsn10('PNAS_Smith09_rsn10.nii.gz',r'c:\temp',r'c:\temp\test.jpg')
    '''
    z=[8,-4,-10,30,-34,50,14,22,46,48]
    ii = 0
    images=[]
    fig=plt.figure(figsize=(4,6),dpi=300)
    for img in image.iter_img(nii_ff):
            # img is now an in-memory 3D img
        tempimage= join(workdir,'RSN%02d.jpg' % (ii+1))
        display = plotting.plot_stat_map(img, figure=fig, threshold=threshold, display_mode="z", cut_coords=[(z[ii])],
                               colorbar=False)
        display.annotate(size=30)
        display.savefig(tempimage)
        images.append(tempimage)
        plt.clf()
        ii += 1
    plt.close()
    row1 = concat_n_images(images[0:5])
    row2 = concat_n_images(images[5:10])

    output = np.vstack((row1,255*np.ones((10,row1.shape[1],3),dtype=np.uint8),row2))

    fig=plt.figure(figsize=(output.shape[0]//30,output.shape[1]//30),dpi=100)
    plt.axis('off')
    plt.imshow(output)
    fig.savefig(resultjpg_ff, bbox_inches='tight')
def plotrsn10from20(nii_ff, workdir, resultjpg_ff, threshold):

    index_in_rsn20 = [6,16,17,7,9,2,3,8,13,12]
    z=[8,-4,-10,30,-34,50,14,22,46,48]
    ii = 0
    images=[]
    fig=plt.figure(figsize=(4,6),dpi=300)
    for ii in range(10):
        #img = image.index_img(nii_ff, index_in_rsn20[ii]-1)
        img = image.index_img(nii_ff, index_in_rsn20[ii]-1)
            # img is now an in-memory 3D img
        tempimage= join(workdir,'RSN%02d.jpg' % (ii+1))
        display = plotting.plot_stat_map(img, figure=fig, threshold=threshold, display_mode="z", cut_coords=[(z[ii])],
                               colorbar=False)
        display.annotate(size=30)
        display.savefig(tempimage)
        images.append(tempimage)
        plt.clf()
        ii += 1
    plt.close()
    row1 = concat_n_images(images[0:5])
    row2 = concat_n_images(images[5:10])

    output = np.vstack((row1,255*np.ones((10,row1.shape[1],3),dtype=np.uint8),row2))

    fig=plt.figure(figsize=(output.shape[0]//30,output.shape[1]//30),dpi=100)
    plt.axis('off')
    plt.imshow(output)
    fig.savefig(resultjpg_ff, bbox_inches='tight')
