from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import json
import tifffile
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Union, Tuple, List, Dict
from time import time
from .core import metadata_from_scanimage_tif, _register_stack


def split_data_to_channels(zstack_path: Union[Path, str]) -> Tuple[List[np.ndarray], List[Dict]]:
    """Split a z-stack file into separate channels
        
    Parameters
    ----------
    zstack_path : Union[Path, str]
        Path to the zstack file
    
    Returns
    -------
    List[np.ndarray]
        List of split data
    List[Dict]
        Metadata of the zstack for each channel
    """
    stack_metadata, _, _ = metadata_from_scanimage_tif(zstack_path)
    num_slices = stack_metadata['num_slices']
    num_volumes = stack_metadata['num_volumes']
    num_channels = stack_metadata['num_channels']
    channels_saved = stack_metadata['channels_saved']

    cz_reader = ScanImageTiffReader(str(zstack_path))
    total_num_frames = cz_reader.shape()[0]
    assert total_num_frames == num_slices * num_volumes * num_channels

    data = cz_reader.data()
    zstack_name = Path(zstack_path).stem
    
    split_data = []
    metadata = []
    
    if num_channels >= 1:
        total_num_frames_each_channel = total_num_frames // num_channels
        
        for ch_ind, channel in enumerate(channels_saved):
            channel_data = data[ch_ind::num_channels]
            split_data.append(channel_data)
            metadata.append({
                'zstack_name': zstack_name,
                'channel': channel,
                'num_slices': num_slices,
                'num_volumes': num_volumes,
                'num_channels': num_channels,
                'total_num_frames': total_num_frames,
                'total_num_frames_each_channel': total_num_frames_each_channel,
                'ch_ind': ch_ind
            })
    return split_data, metadata


def process_split_data(split_data: np.ndarray, metadata: Dict, output_dir: str) -> str:
    """Process a split data containing a single channel of a z-stack
    
    Parameters
    ----------
    split_data : np.ndarray
        Split data
    metadata : Dict
        Metadata of the zstack for each channel
    output_dir : str
        Directory to save the processed file
    
    Returns
    -------
    str
        Path to the processed file
    """
    num_slices = metadata['num_slices']
    total_num_frames = metadata['total_num_frames_each_channel']
    assert split_data.shape[0] == total_num_frames
    base_name = metadata['zstack_name']
    ch_ind = metadata['ch_ind']
    channel = metadata['channel']
    
    # Process the channel
    zstack_reg, _ = _register_stack(split_data, total_num_frames, num_slices, ch_ind)
    
    # Save the processed file
    output_path = Path(output_dir) / f"{base_name}_reg_ch_{channel}.tif"
    tifffile.imwrite(output_path, zstack_reg)
    
    return str(output_path)


def register_local_zstack_from_raw_tif(zstack_path: Union[Path, str],
                                       parallel: bool = True
                                       ) -> Tuple[np.ndarray, list]:
    """ Get registered z-stack, both within and between planes
    From raw tiff stack, meaning that we have to split first
    
    Parameters
    ----------
    zstack_path : Union[Path, str]
        Path to the zstack file
    parallel : bool, optional
        Whether to use parallel processing, by default True
    
    Returns
    -------
    Tuple[np.ndarray, list]
        Registered z-stack and list of saved channels
    """
    stack_metadata, _, _ = metadata_from_scanimage_tif(zstack_path)
    num_slices = stack_metadata['num_slices']
    num_volumes = stack_metadata['num_volumes']
    num_channels = stack_metadata['num_channels']
    channels_saved = stack_metadata['channels_saved']

    cz_reader = ScanImageTiffReader(str(zstack_path))
    total_num_frames = cz_reader.shape()[0]
    assert total_num_frames == num_slices * num_volumes * num_channels

    data = cz_reader.data()
    if num_channels == 1:
        zstack_reg, _ = _register_stack(data, total_num_frames, num_slices, 0)
    elif num_channels > 0:
        total_num_frames_each_channel = total_num_frames // num_channels
        if parallel:
            with ProcessPoolExecutor(max_workers=len(channels_saved)) as executor:
                futures = []
                for ch_ind in range(len(channels_saved)):
                    future = executor.submit(_register_stack,
                                          data[ch_ind::num_channels],
                                          total_num_frames_each_channel,
                                          num_slices,
                                          ch_ind)
                    futures.append(future)
                
                reg_results = [f.result() for f in futures]
                zstack_reg = []
                ch_inds = [rr[1] for rr in reg_results]
                for ch_ind in range(len(ch_inds)):
                    matched_ind = ch_inds.index(ch_ind)
                    zstack_reg.append(reg_results[matched_ind][0])
        else:
            # Sequential processing
            zstack_reg = []
            for ch_ind in range(len(channels_saved)):
                temp_reg, _ = _register_stack(data[ch_ind::num_channels],
                                            total_num_frames_each_channel, num_slices, ch_ind)
                zstack_reg.append(temp_reg)
    else:
        raise ValueError("num_channels should be 1 or more")

    return zstack_reg, channels_saved


def process_zstack_paths(zstack_paths, output_dir):
    """Process multiple zstack paths in parallel
    
    Stage 1: Split each z-stack into channels 
    Stage 2: Process each split z-stack in parallel
    
    Parameters
    ----------
    zstack_paths : list
        List of paths to zstack files
    output_dir : str
        Directory to save processed files
    """
    # Create output directory if it doesn't exist
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(zstack_paths)} files to process")
    print("Stage 1: Splitting files into channels...")
    start_time = time()
    with ProcessPoolExecutor() as executor:
        futures = []
        for path in zstack_paths:
            future = executor.submit(split_data_to_channels, path)
            futures.append(future)
                
    results = [f.result() for f in futures]
    split_data = []
    metadata = []
    for result in results:
        for d, m in zip(result[0], result[1]):
            split_data.append(d)
            metadata.append(m)
    print(f"Stage 1: Split into channels in {time() - start_time} seconds")
    # Stage 2: Process each intermediate file in parallel
    print(f"Stage 2: Processing {len(split_data)} volumes in parallel...")
    start_time = time()
    with ProcessPoolExecutor() as executor:
        futures = []
        for d, m in zip(split_data, metadata):
            future = executor.submit(process_split_data, d, m, output_dir)
            futures.append(future)
        
        # Wait for all futures to complete and collect results
        output_files = []
        for future in futures:
            output_files.append(future.result())
    print(f"Stage 2: Processed {len(split_data)} volumes in {time() - start_time} seconds")
    print(f"Processing complete. Results saved to {output_dir}")

    return output_files
