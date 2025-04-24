from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import json
import re
import scipy
import skimage
from pathlib import Path
from multiprocessing import Pool


def register_local_zstack_from_raw_tif(zstack_path: Union[Path, str],
                                       parallel: bool = True
                                       ) -> Tuple[np.ndarray, list]:
    """ Get registered z-stack, both within and between planes
    From raw tiff stack, meaning that we have to split first
    
    Parameters
    ----------
    local_z_stack : np.ndarray (3D)
        Raw local z-stack, tiff file

    Returns
    -------
    np.ndarray (3D)
        within and between plane registered z-stack
    """
    stack_metadata, _, _ = metadata_from_scanimage_tif(zstack_path)
    num_slices = stack_metadata['num_slices']
    num_volumes = stack_metadata['num_volumes']
    num_channels = stack_metadata['num_channels'] # TODO: need to check its validity in a larger batch of data
    channels_saved = stack_metadata['channels_saved']

    cz_reader = ScanImageTiffReader(str(zstack_path))
    total_num_frames = cz_reader.shape()[0]
    assert total_num_frames == num_slices * num_volumes * num_channels

    data = cz_reader.data()
    if num_channels == 1:
        zstack_reg = _register_stack(data, total_num_frames, num_slices)
    elif num_channels > 0:
        total_num_frames_each_channel = total_num_frames // num_channels
        if parallel:
            with Pool() as pool:
                # Use multiprocessing to register each channel in parallel
                reg_results = pool.starmap(_register_stack,
                                          [(data[ch_ind::num_channels], 
                                            total_num_frames_each_channel, num_slices) 
                                           for ch_ind in range(len(channels_saved))])
            zstack_reg = []
            ch_inds = [rr[1] for rr in reg_results]
            for ch_ind in range(len(ch_inds)):
                matched_ind = ch_inds.index(ch_ind)
                zstack_reg.append(reg_result[matched_ind][0])
            
        else:
            # Sequential processing
            zstack_reg = []
            for ch_ind in range(len(channels_saved)):
                temp_reg, _ = _register_stack(data[ch_ind::num_channels],
                                              total_num_frames_each_channel, num_slices)
                zstack_reg.append(temp_reg)
    else:
        raise ValueError("num_channels should be 1 or more")

    return zstack_reg, channels_saved


def metadata_from_scanimage_tif(stack_path):
    """Extract metadata from ScanImage tiff stack

    Dev notes:
    Seems awkward to parse this way
    Depends on ScanImageTiffReader

    Parameters
    ----------
    stack_path : str
        Path to tiff stack

    Returns
    -------
    dict
        stack_metadata: important metadata extracted from scanimage tiff header
    dict
        si_metadata: all scanimge metadata. Each value still a string, so convert if needed.
    dict
        roi_groups_dict: 
    """
    with ScanImageTiffReader(str(stack_path)) as reader:
        md_string = reader.metadata()

    # split si & roi groups, prep for seprate parse
    s = md_string.split("\n{")
    rg_str = "{" + s[1]
    si_str = s[0]

    # parse 1: extract keys and values, dump, then load again
    si_metadata = _extract_dict_from_si_string(si_str)
    # parse 2: json loads works hurray
    roi_groups_dict = json.loads(rg_str)

    stack_metadata = {}
    stack_metadata['num_slices'] = int(si_metadata['SI.hStackManager.actualNumSlices'])
    stack_metadata['num_volumes'] = int(si_metadata['SI.hStackManager.actualNumVolumes'])
    stack_metadata['frames_per_slice'] = int(si_metadata['SI.hStackManager.framesPerSlice'])
    # stack_metadata['z_steps'] = _str_to_int_list(si_metadata['SI.hStackManager.zs'])
    stack_metadata['z_steps'] = _str_to_float_list(si_metadata['SI.hStackManager.zs'])
    stack_metadata['actuator'] = si_metadata['SI.hStackManager.stackActuator']
    # stack_metadata['num_channels'] = sum(_str_to_bool_list(si_metadata['SI.hPmts.powersOn']))
    channels_saved = [ss for ss in re.split('\[|\]| ', si_metadata['SI.hChannels.channelSave']) if len(ss)>0]
    channels_saved = [int(cs) for cs in channels_saved if str(int(cs)) == cs]
    stack_metadata['num_channels'] = len(channels_saved) # TODO: need to check its validity in a larger batch of data
    stack_metadata['channels_saved'] = channels_saved
    # stack_metadata['z_step_size'] = int(si_metadata['SI.hStackManager.actualStackZStepSize'])
    stack_metadata['z_step_size'] = float(si_metadata['SI.hStackManager.actualStackZStepSize'])

    return stack_metadata, si_metadata, roi_groups_dict


def _extract_dict_from_si_string(string):
    """Parse the 'SI' variables from a scanimage metadata string"""

    lines = string.split('\n')
    data_dict = {}
    for line in lines:
        if line.strip():  # Check if the line is not empty
            key, value = line.split(' = ')
            key = key.strip()
            if value.strip() == 'true':
                value = True
            elif value.strip() == 'false':
                value = False
            else:
                value = value.strip().strip("'")  # Remove leading/trailing whitespace and single quotes
            data_dict[key] = value

    json_data = json.dumps(data_dict, indent=2)
    loaded_data_dict = json.loads(json_data)
    return loaded_data_dict


def _register_stack(stack, total_num_frames, number_of_z_planes, ch_ind):
    mean_local_zstack_reg = []
    for plane_ind in range(number_of_z_planes):
        single_plane_images = stack[range(
            plane_ind, total_num_frames, number_of_z_planes), ...]
        single_plane, shifts = average_reg_plane(single_plane_images)
        mean_local_zstack_reg.append(single_plane)

    # Old Scientifica microscope had flyback and ringing in the first 5 frames
    # TODO: reimplement for old rigs (4/2024)
    # if 'CAM2P' in equipment_name:
    #     mean_local_zstack_reg = mean_local_zstack_reg[5:]
    _zstack_reg, _shifts_between = reg_between_planes(np.array(mean_local_zstack_reg))
    return _zstack_reg, ch_ind


def average_reg_plane(images: np.ndarray) -> Union[np.ndarray, list]:
    """Get mean FOV of a plane after registration.
    Use phase correlation

    Parameters
    ----------
    images : np.ndarray (3D)
        frames from a plane

    Returns
    -------
    np.ndarray (2D)
        mean FOV of a plane after registration.
    """

    # if num_for_ref is None or num_for_ref < 1:
    #   ref_img = np.mean(images, axis=0)
    ref_img, _ = pick_initial_reference(images)
    reg = np.zeros_like(images)
    shift_all = []
    for i in range(images.shape[0]):
        shift, _, _ = skimage.registration.phase_cross_correlation(
            ref_img, images[i, :, :], normalization=None)
        reg[i, :, :] = scipy.ndimage.shift(images[i, :, :], shift)
        shift_all.append(shift)
    return np.mean(reg, axis=0), shift_all


def pick_initial_reference(frames: np.ndarray, num_for_ref: int = 20) -> np.ndarray:
    """ computes the initial reference image

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top 20 correlated pairs is the
    inital reference frame returned

    From suite2p.registration.register

    Parameters
    ----------
    frames : 3D array, int16
        size [frames x Ly x Lx], frames from binary

    Returns
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    """
    nimg, Ly, Lx = frames.shape
    frames = np.reshape(frames, (nimg, -1)).astype('float32')
    frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1))
    cc = np.matmul(frames, frames.T)
    ndiag = np.sqrt(np.diag(cc))
    cc = cc / np.outer(ndiag, ndiag)
    CCsort = -np.sort(-cc, axis=1)
    bestCC = np.mean(CCsort[:, 1:num_for_ref], axis=1)
    imax = np.argmax(bestCC)
    indsort = np.argsort(-cc[imax, :])
    selected_frame_inds = indsort[0:num_for_ref]
    refImg = np.mean(frames[selected_frame_inds, :], axis=0)
    refImg = np.reshape(refImg, (Ly, Lx))
    return refImg, selected_frame_inds


def reg_between_planes(stack_imgs,
                       ref_ind: int = 30,
                       top_ring_buffer: int = 10,
                       window_size: int = 5,
                       use_adapthisteq: bool = True):
    """Register between planes. Each plane with single 2D image
    Use phase correlation.
    Use median filtered images to calculate shift between neighboring planes.
    Resulting image is not filtered.

    Parameters
    ----------
    stack_imgs : np.ndarray (3D)
        images of a stack. Typically z-stack with each plane registered and averaged.
    ref_ind : int, optional
        index of the reference plane, by default 30
    top_ring_buffer : int, optional
        number of top lines to skip due to ringing noise, by default 10
    window_size : int, optional
        window size for rolling, by default 5
    use_adapthisteq : bool, optional
        whether to use adaptive histogram equalization, by default True

    Returns
    -------
    np.ndarray (3D)
        Stack after plane-to-plane registration.
    """
    num_planes = stack_imgs.shape[0]
    reg_stack_imgs = np.zeros_like(stack_imgs)
    reg_stack_imgs[ref_ind, :, :] = stack_imgs[ref_ind, :, :]
    ref_stack_imgs = med_filt_z_stack(stack_imgs)
    if use_adapthisteq:
        for i in range(num_planes):
            plane_img = ref_stack_imgs[i, :, :]
            timg = skimage.exposure.equalize_adapthist(plane_img.astype(np.uint16))
            ref_stack_imgs[i, :, :] = image_normalization(timg, dtype='uint16')

    temp_stack_imgs = np.zeros_like(stack_imgs)

    temp_stack_imgs[ref_ind, :, :] = ref_stack_imgs[ref_ind, :, :]
    shift_all = []
    shift_all.append([0, 0])
    for i in range(ref_ind + 1, num_planes):
        # Calculation valid pixels
        temp_ref = np.mean(
            temp_stack_imgs[max(0, i - window_size):i, :, :], axis=0)
        temp_mov = ref_stack_imgs[i, :, :]
        valid_y, valid_x = calculate_valid_pix(temp_ref, temp_mov)

        temp_ref = temp_ref[valid_y[0] +
                            top_ring_buffer:valid_y[1] + 1, valid_x[0]:valid_x[1] + 1]
        temp_mov = temp_mov[valid_y[0] +
                            top_ring_buffer:valid_y[1] + 1, valid_x[0]:valid_x[1] + 1]

        shift, _, _ = skimage.registration.phase_cross_correlation(
            temp_ref, temp_mov, normalization=None, upsample_factor=10)
        temp_stack_imgs[i, :, :] = scipy.ndimage.shift(
            ref_stack_imgs[i, :, :], shift)
        reg_stack_imgs[i, :, :] = scipy.ndimage.shift(
            stack_imgs[i, :, :], shift)
        shift_all.append(shift)
    if ref_ind > 0:
        for i in range(ref_ind - 1, -1, -1):
            temp_ref = np.mean(
                temp_stack_imgs[i + 1: min(num_planes, i + window_size + 1), :, :], axis=0)
            temp_mov = ref_stack_imgs[i, :, :]
            valid_y, valid_x = calculate_valid_pix(temp_ref, temp_mov)

            temp_ref = temp_ref[valid_y[0] +
                                top_ring_buffer:valid_y[1] + 1, valid_x[0]:valid_x[1] + 1]
            temp_mov = temp_mov[valid_y[0] +
                                top_ring_buffer:valid_y[1] + 1, valid_x[0]:valid_x[1] + 1]

            shift, _, _ = skimage.registration.phase_cross_correlation(
                temp_ref, temp_mov, normalization=None, upsample_factor=10)
            temp_stack_imgs[i, :, :] = scipy.ndimage.shift(
                ref_stack_imgs[i, :, :], shift)
            reg_stack_imgs[i, :, :] = scipy.ndimage.shift(
                stack_imgs[i, :, :], shift)
            shift_all.insert(0, shift)
    return reg_stack_imgs, shift_all


def med_filt_z_stack(zstack, kernel_size=5):
    """Get z-stack with each plane median-filtered

    Parameters
    ----------
    zstack : np.ndarray
        z-stack to apply median filtering
    kernel_size : int, optional
        kernel size for median filtering, by default 5
        It seems only certain odd numbers work, e.g., 3, 5, 11, ...

    Returns
    -------
    np.ndarray
        median-filtered z-stack
    """
    filtered_z_stack = []
    for image in zstack:
        filtered_z_stack.append(cv2.medianBlur(
            image.astype(np.uint16), kernel_size))
    return np.array(filtered_z_stack)


def image_normalization(image: np.ndarray,
                        dtype: str = 'uint16',
                        im_thresh: float = 0):
    """Normalize 2D image and convert to dtype
    Prevent saturation.

    Parameters
    ----------
    image : np.ndarray
        input image (2D)
    dtype : str, optional
        output data type, by default 'uint16'
    im_thresh : float, optional
        threshold when calculating pixel intensity percentile, by default 0
    """
    assert dtype in ['uint8', 'uint16'], "dtype should be either 'uint8' or 'uint16'"

    if dtype == 'uint8':
        dtype = np.uint8
    elif dtype == 'uint16':
        dtype = np.uint16

    clip_image = np.clip(image, np.percentile(
        image[image > im_thresh], 0.2), np.percentile(image[image > im_thresh], 99.8))
    norm_image = (clip_image - np.amin(clip_image)) / \
        (np.amax(clip_image) - np.amin(clip_image)) * 0.9
    norm_image = ((norm_image + 0.05) * np.iinfo(dtype).max * 0.9).astype(dtype)
    return norm_image


def calculate_valid_pix(img1, img2, valid_pix_threshold=1e-3):
    """Calculate valid pixels for registration between two images

    Parameters
    ----------
    img1 : np.ndarray (2D)
        Image 1
    img2 : np.ndarray (2D)
        Image 2
    valid_pix_threshold : float, optional
        threshold for valid pixels, by default 1e-3

    Returns
    -------
    list
        valid y range
    list
        valid x range
    """
    y1, x1 = np.where(img1 > valid_pix_threshold)
    y2, x2 = np.where(img2 > valid_pix_threshold)
    # unravel the indices
    valid_y = [max(min(y1), min(y2)), min(max(y1), max(y2))]
    valid_x = [max(min(x1), min(x2)), min(max(x1), max(x2))]
    return valid_y, valid_x