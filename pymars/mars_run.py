import os
from os.path import join, isdir, split, dirname, basename, isfile
import subprocess
import nibabel
import numpy as np
import logging
import nilearn
from nilearn import plotting
import shutil
import matplotlib.pyplot as plt
from rmap import seed_to_voxel_corr, plotrsn10, get_connindex

app_option = '0'
def check_indi(workpath):
    file_dict = {'org_workpath': workpath}
    pathkey = {'anat_': 'T1', 'hires_': 'T1', 'rest_': 'rest',
               'fmM_': 'fmM', 'fmP_': 'fmP'}
    INDI = False
    for root, subdirs, files in os.walk(workpath):
        if len(subdirs) > 0 or len(files) == 0:
            continue

        nii_fname = '0'
        for fname in files:
            if '.nii' in fname:
                nii_fname = fname  # found nii file
        if nii_fname == '0':
            continue

        lastdir = os.path.split(root)[-1]
        for key, value in pathkey.items():
            if key in lastdir:
                INDI = True
                file_dict[value + '_file'] = nii_fname
                file_dict[value + '_dir'] = root
                file_dict[value + '_ffname'] = join(root, nii_fname)
                logging.info('Found %s file\t: %s' %
                             (value, file_dict[value + '_ffname']))
    return file_dict, INDI


def safe_mkdir(dirname):
    try:
        os.mkdir(dirname)
    except:
        pass


def systemx(cmd, display=True):
    if display:
        # logging.info("run: " + cmd.replace(workpath,''))
        logging.info("run: " + cmd)
    subprocess.call(cmd, shell=True)


def nii3d_to_jpg(nii_ff, output_ff=''):
    if output_ff == '':
        output_ff = nii_ff + '.jpg'
    if not isfile(nii_ff):
        nii_ff = nii_ff + '.nii'
    if not isfile(nii_ff):
        nii_ff = nii_ff + '.gz'
    try:
        plotting.plot_anat(nii_ff, draw_cross=False, output_file=output_ff)
    except:
        pass


def FSL_T1_prep(mni152_brain_ff, file_dict):
    # return bpmprage_pve_1,bpmprage_pve_2,bpmprage_pve_0
    mT1_dir = join(file_dict['org_workpath'], 'marsT1')
    safe_mkdir(mT1_dir)

    logging.info('Swap T1 Dimension')
    # systemx('fslswapdim %s RL PA IS %s' % (file_dict['T1_ffname'],join(mT1_dir, 'pmprage')))
    systemx('fslreorient2std %s %s' %
            (file_dict['T1_ffname'], join(mT1_dir, 'mprage.nii.gz')))
    systemx('robustfov -i %s -r  %s' %
            (join(mT1_dir, 'mprage.nii.gz'), join(mT1_dir, 'pmprage.nii.gz')))
    systemx('bet %s %s -m -R -B -f 0.2' %
            (join(mT1_dir, 'pmprage.nii.gz'), join(mT1_dir, 'bpmprage')))
    systemx('flirt -in %s -ref %s -omat %s -dof 12 -cost normmi ' %
            (join(mT1_dir, 'bpmprage'),
             mni152_brain_ff,
             join(mT1_dir, 'str2stand.mat')))
    systemx('convert_xfm -omat %s -inverse %s' %
            (join(mT1_dir, 'stand2str.mat'), join(mT1_dir, 'str2stand.mat')))
    systemx('fnirt --in=%s --aff=%s --cout=%s --config=T1_2_MNI152_2mm' %
            (join(mT1_dir, 'pmprage'),
             join(mT1_dir, 'str2stand.mat'),
             join(mT1_dir, 'nl_str2stand.mat')))
    logging.info('segmenting T1 ')
    systemx('fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o %s' %
            join(mT1_dir, 'bpmprage'))
    T1prep_dict = {'T1_ffname': join(mT1_dir, 'pmprage'),
                   'T1bet_ffname': join(mT1_dir, 'bpmprage'),
                   'bmask_ffname': join(mT1_dir, 'bpmprage_mask.nii.gz'),
                   'str2stand': join(mT1_dir, 'str2stand.mat'),
                   'stand2str': join(mT1_dir, 'stand2str.mat'),
                   'nl_str2stand': join(mT1_dir, 'nl_str2stand.mat'),
                   'GM_ffname': join(mT1_dir, 'bpmprage_pve_1.nii.gz'),
                   'WM_ffname': join(mT1_dir, 'bpmprage_pve_2.nii.gz'),
                   'CSF_ffname': join(mT1_dir, 'bpmprage_pve_0.nii.gz'), }
    '''
    T1prep_dict['wT1_ffname'] =  FSL_norm(T1prep_dict['T1_ffname'],T1prep_dict['str2stand'],mni152_brain_ff)
    T1prep_dict['wGM_ffname'] = FSL_norm(T1prep_dict['GM_ffname'],T1prep_dict['str2stand'],mni152_brain_ff)
    T1prep_dict['wWM_ffname'] = FSL_norm(T1prep_dict['WM_ffname'],T1prep_dict['str2stand'],mni152_brain_ff)
    T1prep_dict['wCSF_ffname'] = FSL_norm(T1prep_dict['CSF_ffname'],T1prep_dict['str2stand'],mni152_brain_ff)
    '''
    T1prep_dict['wT1_ffname'] = FSL_fnirt_applywarp(mni152_brain_ff,
                                                    T1prep_dict['T1_ffname'],
                                                    T1prep_dict['nl_str2stand'])
    T1prep_dict['wGM_ffname'] = FSL_fnirt_applywarp(mni152_brain_ff,
                                                    T1prep_dict['GM_ffname'],
                                                    T1prep_dict['nl_str2stand'])
    T1prep_dict['wWM_ffname'] = FSL_fnirt_applywarp(mni152_brain_ff,
                                                    T1prep_dict['WM_ffname'],
                                                    T1prep_dict['nl_str2stand'])
    T1prep_dict['wCSF_ffname'] = FSL_fnirt_applywarp(mni152_brain_ff,
                                                     T1prep_dict['CSF_ffname'],
                                                     T1prep_dict['nl_str2stand'])
    T1prep_dict['wbmask_ffname'] = FSL_fnirt_applywarp(mni152_brain_ff,
                                                       T1prep_dict['bmask_ffname'],
                                                       T1prep_dict['nl_str2stand'])
    T1prep_dict['wWMCSF_ffname'] = join(mT1_dir,'wWMCSF.nii.gz')

    systemx('fslmaths %s -add %s -thr 0.5 -bin %s' % (T1prep_dict['wWM_ffname'],
                                                      T1prep_dict['wCSF_ffname'],
                                                      T1prep_dict['wWMCSF_ffname']))
    systemx('fslmaths %s  -thr 0.5 -bin %s' % (T1prep_dict['wbmask_ffname'],
                                               T1prep_dict['wbmask_ffname']))


    return T1prep_dict

def get_friston24(data):
    data_roll = np.roll(data, 1, axis=0)
    data_roll[0] = 0
    new_data = np.concatenate((data, data ** 2, data_roll, data_roll ** 2), axis=1)
    return new_data

def FSL_EPI_prep(file_dict):
    # motion correction and regressing out motion correction parameters
    #remove first 8 volumes

    # slice timing
    data = nibabel.nifti1.load(file_dict['rest_ffname'])
    TR = data.get_header().get_zooms()[3]
    mrest_dir = join(file_dict['org_workpath'], 'mrest')
    safe_mkdir(mrest_dir)
    systemx('fslroi %s %s 8 -1' % (file_dict['rest_ffname'],join(mrest_dir, 'rest.nii.gz')))
    systemx('slicetimer -i %s --out=%s -r %d --odd' %
            (join(mrest_dir, 'rest.nii.gz'),
             join(mrest_dir, 'srest.nii.gz'),
             TR))
    logging.info('Estimating motion parameter...')
    systemx('mcflirt -in %s -o %s -plots' % (join(mrest_dir, 'srest.nii.gz'),
            join(mrest_dir, 'rsrest.nii.gz')))
    logging.info('Estimating motion parameter...done')
    #logging.info('regressing out motion correction parameters')

    # load mcflirt params
    data = np.genfromtxt(join(mrest_dir,'rsrest.nii.gz.par'))
    f24 = get_friston24(data)
    # load nifti data

    rest_prep_ffname = join(mrest_dir, 'rsrest.nii.gz')
    return rest_prep_ffname, f24, TR


def FSL_EPI2struct(rest_prep_ffname, T1prep_dict):
    logging.info('making transformation matrix with BBR')
    EPI_dir = dirname(rest_prep_ffname)
    systemx('epi_reg --epi=%s --t1=%s --t1brain=%s --wmseg=%s --out=%s' %
            (rest_prep_ffname,
             T1prep_dict['T1_ffname'],
             T1prep_dict['T1bet_ffname'],
             T1prep_dict['WM_ffname'],
             join(EPI_dir, 'epi2str')))
    systemx('rm -f %s' % join(EPI_dir, 'epi2str.nii.gz'))
    # remove large tempfile
    systemx('convert_xfm -omat %s -inverse %s' %
            (join(EPI_dir, 'str2epi.mat'),
             join(EPI_dir, 'epi2str.mat')))
    systemx('convert_xfm -omat %s  -concat %s  %s ' %
            (join(EPI_dir, 'epi2stand.mat'),
             T1prep_dict['str2stand'],
             join(EPI_dir, 'epi2str.mat')))

    epi2std_ff = join(EPI_dir, 'epi2stand.mat')
    epi2struct_ff = join(EPI_dir, 'epi2str.mat')
    # EPI->MNI152(bounding): EPI=>Stand=>Stand(bounding)
    return epi2std_ff, epi2struct_ff


def FSL_norm(img_ff, mat_img2ref, refimg_ff):
    norm_img_ff = join(dirname(img_ff), 'w' + basename(img_ff))
    systemx('flirt -in %s -ref %s -applyxfm -init %s -out %s' %
            (img_ff, refimg_ff, mat_img2ref, norm_img_ff))
    return norm_img_ff


def FSL_fnirt_applywarp(mni152_brain_ff, img_ff, warp_ff, premat_ff=''):
    warped_img_ff = join(dirname(img_ff), 'w' + basename(img_ff))
    cmd = 'applywarp --ref=%s --in=%s --warp=%s --out=%s ' % (mni152_brain_ff,
                                                              img_ff,
                                                              warp_ff,
                                                              warped_img_ff)
    if not premat_ff == '':
        cmd += '--premat=' + premat_ff
    systemx(cmd)
    return warped_img_ff



def get_confound(epi_ff,cfnmask_ff):
    tsout = join(dirname(epi_ff), "ts.txt")
    systemx("fslmeants -i %s -m %s -o %s" % (epi_ff, cfnmask_ff, tsout))
    ts_to_regress = np.loadtxt(tsout, unpack=True)
    systemx("rm -r %s" % tsout)
    return ts_to_regress

# regress out WM/CSF
def regressout_mask(epi_ff, cfnmask_ff):
    # cfnmask_ff: confound mask
    # load nifti data
    # A function modified from Nan-Kuei Chen's script
    data = nibabel.nifti1.load(epi_ff)
    epi_vol = data.get_data()

    tsout = join(dirname(epi_ff), "ts.txt")
    systemx("fslmeants -i %s -m %s -o %s" % (epi_ff, cfnmask_ff, tsout))
    ts_to_regress = np.loadtxt(tsout, unpack=True)
    tdim = epi_vol.shape[3]
    zdim = epi_vol.shape[2]
    systemx("rm -r %s" % tsout)

    X_confound = np.vstack([np.ones(tdim), ts_to_regress]).T

    logging.info('starting linear regression')
    tmp_mean = np.mean(epi_vol, axis=3)
    shape = epi_vol.shape
    data1v = epi_vol.reshape((shape[0]*shape[1], shape[2], shape[3])).transpose((1, 2, 0))
    # data1v is a view in z, t, x*y order
    # go slice-by-slice
    for cntz in range(zdim):
        tmp_data = data1v[cntz]
        # regress wm
        p01 = np.linalg.lstsq(X_confound, tmp_data)[0]
        p001 = np.dot(X_confound, p01)  # product
        tmp02 = tmp_data - p001

    data_mr = data1v.transpose((2, 0, 1)).reshape(shape)
    del data1v
    del epi_vol
    data_mr = data_mr + tmp_mean.reshape(tmp_mean.shape + (1,))
    data_mr -= np.min(data_mr)
    data_mr *= 30000.0 / np.max(data_mr)
    newNii = nibabel.Nifti1Pair(data_mr, None, data.get_header())
    nibabel.save(newNii, epi_ff)


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(message)s',
                    datefmt='%m/%d %I:%M')
logging.info('Analysis Started')
mni152_brain_ff = join(os.environ['FSLDIR'], 'data',
                       'standard', 'MNI152_T1_2mm_brain.nii.gz')
mni152_ff = join(os.environ['FSLDIR'], 'data',
                 'standard', 'MNI152_T1_2mm.nii.gz')
app_dir = os.path.dirname(os.path.abspath(__file__))
if 'workpath' not in locals():  # local testing
    logging.info('Local Testing!!')
    workpath = join(app_dir, 'workdir')
    resultpath = join(app_dir, 'resultdir')
file_dict, INDI = check_indi(workpath)
logging.info('Performing T1 preprocessing')
T1prep_dict = FSL_T1_prep(mni152_brain_ff, file_dict)
rest_prep_ff, friston24, TR = FSL_EPI_prep(file_dict)
epi2std_ff, epi2struct_ff = FSL_EPI2struct(rest_prep_ff, T1prep_dict)
# wEPI_ff = FSL_norm(rest_prep_ff,epi2stdb_ff,mni152_brain_ff)
wEPI_ff = FSL_fnirt_applywarp(mni152_ff, rest_prep_ff,
                              T1prep_dict['nl_str2stand'],
                              premat_ff=epi2struct_ff)
wm_ts = get_confound(wEPI_ff, T1prep_dict['wWM_ffname'])
csf_ts = get_confound(wEPI_ff, T1prep_dict['wCSF_ffname'])
gb_ts = get_confound(wEPI_ff, T1prep_dict['wbmask_ffname'])
constant_ts = np.ones(wm_ts.size)
linear_ts = np.arange(0.0,wm_ts.size)
compcor5 = nilearn.image.high_variance_confounds(wEPI_ff, n_confounds=5,mask_img=T1prep_dict['wWMCSF_ffname'])


confound_ff = join(resultpath, 'confound.csv')
confound_GSR_ff = join(resultpath, 'csf_wm_global.csv')
fristonstr=''
for ii in range(24):fristonstr+='f%s ' % (ii+1)

confounds = np.hstack([np.array([csf_ts,constant_ts, linear_ts, linear_ts**2]).T,
                      friston24, compcor5])
np.savetxt(confound_ff, confounds,
           header='csf constant linear quadratic ' + fristonstr + 'compcor1 compcor2 compcor3 compcor4 compcor5',
           comments='',fmt='%.10f')
confounds = np.hstack([np.array([csf_ts, wm_ts,gb_ts,constant_ts, linear_ts, linear_ts**2]).T,
                      friston24, compcor5])
np.savetxt(confound_GSR_ff, confounds,
           header='csf wm global constant linear quadratic ' + fristonstr + 'compcor1 compcor2 compcor3 compcor4 compcor5',
           comments='',fmt='%.10f')

# > From 2 to 4 mm:
# flirt -in brain_2mm.nii.gz -ref brain_2mm.nii.gz -out brain_4mm.nii.gz -nosearch -applyisoxfm 4
# -nosearch -applyisoxfm 4
#fsl_glm -i wrsrest.nii.gz -d rsn10_0001.nii.gz -o test2.txt --demean

#regressout_mask(wEPI_ff, T1prep_dict['wWM_ffname'])
#regressout_mask(wEPI_ff, T1prep_dict['wCSF_ffname'])
if (app_option=='1'):
    #shutil.copy(wEPI_ff, resultpath)
    systemx('flirt -in %s -ref %s -out %s -nosearch -applyisoxfm 3' % (wEPI_ff, wEPI_ff, join(resultpath,'EPI_MNI3mm.nii.gz')))
    shutil.copy(T1prep_dict['wWM_ffname'], join(resultpath,'WM_2mm.nii.gz'))
    shutil.copy(T1prep_dict['wCSF_ffname'], join(resultpath,'CSF_2mm.nii.gz'))
    shutil.copy(T1prep_dict['wGM_ffname'], join(resultpath,'GM_2mm.nii.gz'))
#nii3d_to_jpg(T1prep_dict['T1_ffname'], join(resultpath, 'Original_T1.jpg'))
nii3d_to_jpg(T1prep_dict['T1bet_ffname'],
             join(resultpath, 'Original_T1brain.jpg'))
#nii3d_to_jpg(T1prep_dict['wT1_ffname'], join(resultpath, 'normalized_T1.jpg'))
# extract first point to save jpg
wEPI3d_ff = join(dirname(wEPI_ff), '3d'+basename(wEPI_ff))
systemx('fslroi %s %s 0 1' % (wEPI_ff, wEPI3d_ff))
nii3d_to_jpg(wEPI3d_ff, join(resultpath, 'normalized_EPI.jpg'))



RSN10dir = join(resultpath,'RSN10')
safe_mkdir(RSN10dir)

shutil.copy(join(app_dir,'atlas','RSN10.jpg'), RSN10dir)
#dual regression
from nilearn.image import load_img, smooth_img

cleaned_wEPI_ff = join(workpath,'clean_wEPI.nii.gz')
nilearn.image.clean_img(smooth_img(wEPI_ff, 6), detrend=True, confounds=[confound_ff]).to_filename(cleaned_wEPI_ff)

rsn10_dr_ff = join(RSN10dir, 'rsn10_dr.nii.gz')
rsn10_drz_ff = join(RSN10dir, 'rsn10_drzstat.nii.gz')
systemx('fsl_glm -i %s -d %s -o %s -m %s --demean' % (cleaned_wEPI_ff,
                                                join(app_dir,'atlas','PNAS_Smith09_rsn10.nii.gz'),
                                                join(RSN10dir, 'RSN10_step1.txt'),
                                                T1prep_dict['wbmask_ffname']))
systemx('fsl_glm -i %s -d %s -o %s --out_z=%s -m %s --demean ' % (cleaned_wEPI_ff,
                                                              join(RSN10dir, 'RSN10_step1.txt'),
                                                              rsn10_dr_ff,
                                                              rsn10_drz_ff ,
                                                              T1prep_dict['wbmask_ffname']))
plotrsn10(rsn10_drz_ff, workpath, join(RSN10dir, 'RSN10DR_dualregress_ztest.jpg'),4)


shutil.copy(T1prep_dict['wbmask_ffname'], join(resultpath,'brainmask.nii.gz'))

os.remove(wEPI_ff)


systemx('date > %s' % join(resultpath, 'MRAPP.DONE'))
logging.info('Finshed!!')
