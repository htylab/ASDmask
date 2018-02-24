from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
from sklearn.covariance import GraphLassoCV
from nilearn.input_data import NiftiMapsMasker

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

def get_all_connectome(fmri_ff, confound_ff, atlas_ff, outputjpg_ff, workpath, labelrange = None, label_or_map = 0):
    metrics = ['correlation', 'partial correlation', 'tangent', 'covariance',
               'precision','sparse inverse covariance']
    images = []
    conn = []
    for ii, metric in enumerate(metrics):
        tempimage= join(workpath,'temp%02d.jpg' % ii)
        conn.append(cal_connectome(fmri_ff, confound_ff, atlas_ff, tempimage,
                              labelrange=labelrange, metric=metric, label_or_map=label_or_map))

        images.append(tempimage)
    row1 = concat_n_images(images[0:3])
    row2 = concat_n_images(images[3:6])
    output = np.vstack((row1, 255*np.ones((10, row1.shape[1],3),dtype=np.uint8), row2))
    fig=plt.figure(figsize=(output.shape[0]//80, output.shape[1]//80), dpi=100)
    plt.axis('off')
    plt.imshow(output)
    fig.savefig(outputjpg_ff, bbox_inches='tight')
    plt.close()
    return conn


def cal_connectome(fmri_ff, confound_ff, atlas_ff, outputjpg_ff,
                   metric='correlation', labelrange=None, label_or_map=0):
    if label_or_map == 0:
        # “correlation”, “partial correlation”, “tangent”, “covariance”, “precision”
        masker = NiftiLabelsMasker(labels_img=atlas_ff, standardize=True,
                                   verbose=0)
    else:
        masker = NiftiMapsMasker(maps_img=atlas_ff, standardize=True,
                                 verbose=0)

    time_series_0 = masker.fit_transform(fmri_ff, confounds=confound_ff)
    if labelrange is None:
        labelrange = np.arange(time_series_0.shape[1])
    time_series = time_series_0[:,labelrange]
    if metric == 'sparse inverse covariance':
        try:
            estimator = GraphLassoCV()
            estimator.fit(time_series)
            correlation_matrix = -estimator.precision_
        except:
            correlation_matrix = np.zeros((time_series.shape[1], time_series.shape[1]))
    else:
        correlation_measure = ConnectivityMeasure(kind=metric)
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    # Plot the correlation matrix

    fig = plt.figure(figsize=(6, 5),dpi=100)
    plt.clf()
    # Mask the main diagonal for visualization:
    np.fill_diagonal(correlation_matrix, 0)

    plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r",
               vmax=0.8, vmin=-0.8)
    plt.gca().yaxis.tick_right()
    plt.axis('off')
    plt.colorbar()
    plt.title(metric.title(), fontsize=12)
    plt.tight_layout()
    fig.savefig(outputjpg_ff, bbox_inches='tight')
    plt.close()
    return correlation_matrix
