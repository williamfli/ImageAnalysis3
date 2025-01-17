import sys
import glob
import os
import time
import copy
import numpy as np

import tqdm
import pandas as pd 
import skimage



def load_color_usage_byBit (fov_param, _fov_id, sort_by = 'hyb_number'):

    ##########################################
    # Init fov classes to get stored info
    from ..classes.field_of_view import Field_of_View
    fov = Field_of_View (fov_param, _fov_id=_fov_id,
                                              _color_info_kwargs={
                                                  '_color_filename':'Color_Usage',
                                              }, 
                                              _prioritize_saved_attrs=False,
                                              _save_info_to_file=False, # whether overwrite
                                              )
    
    from ..get_img_info import Load_Color_Usage
    _color_dic, _use_dapi, _channels = Load_Color_Usage(fov.analysis_folder,
                                                                color_filename='Color_Usage',
                                                                color_format='csv',
                                                                return_color=True)

    color_usage_df = pd.DataFrame(data = _color_dic.values(), index=_color_dic.keys(), columns=_channels)
    # Exclude non bit channels
    non_bit_cols = []
    for _non_bit_color in ['beads', 'DAPI']:
        non_bit_cols.extend([col for col in color_usage_df if
            color_usage_df[col].astype(str).str.contains(_non_bit_color).any()])
    bit_cols = [col for col in color_usage_df.columns if col not in non_bit_cols]
    for bit_col in bit_cols:
        # assume all bit usage start with a single letter denoting the data type in _allowed_kwds
        color_usage_df[bit_col] = color_usage_df[bit_col].apply(lambda x: int(x[1:]) if (x!='' and x[0]=='c') else -1)
    
    color_usage_df['hyb_number'] = color_usage_df.index
    color_usage_df['hyb_number'] = color_usage_df['hyb_number'].apply(lambda x: int(x.split('H')[1].split('C')[0]))
    color_usage_df = color_usage_df.sort_values(by='hyb_number')

    bit_cols_hyb = bit_cols.copy()
    bit_cols_hyb.append('hyb_number')
    bit_usage_df = color_usage_df[bit_cols_hyb]
    # Melt df to get channel and hyb values for every bit
    bit_df_melt = pd.melt(bit_usage_df, id_vars=['hyb_number'], var_name='channel', value_name='bit')
    bit_df_melt = bit_df_melt.loc[bit_df_melt['bit']!=-1] # eliminate all rows that don't qualify (empty cells and values starting with 'f')
    if sort_by in ['hyb_number', 'bit']:
        bit_df_melt = bit_df_melt.sort_values(by=sort_by)
    else:
        bit_df_melt = bit_df_melt.sort_values(by='hyb_number')
    return bit_df_melt




# function to calculate drift for a single hyb round relative to the prior round
def calculate_drift_from_prior_hyb(_hyb_id, 
                                 fov_annotated_folders, 
                                 fov_name, 
                                 _precise_align,
                                 color_info, 
                                 _drift_channel, 
                                 _drift_ref,
                                 _channels, 
                                 single_im_size, 
                                 num_buffer_frames, 
                                 num_empty_frames, 
                                 illumination_corr,
                                 correction_folder, 
                                 ):
    # No drift for the first cycle
    if _hyb_id == 0:
        return np.array([0,0,0]) 
    # Calculate drift from the previous hyb
    else:
        _Image_dict = {}
        for _idx, _Image_key in enumerate(['_DriftImage', 'RefImage']):
            # get hyb_id for ref and drift accordingly
            # one with smaller _ind is ref; make sure the order is correct
            _ind = _hyb_id - _idx # - zero would be self; -1 would Ref
            _bead_folder = fov_annotated_folders[_ind]
            _bead_filename = os.path.join(_bead_folder, fov_name)
            _drift_channel = _drift_channel
            # get used_channels for this dapi folder:
            _info = color_info[os.path.basename(_bead_folder)]
            _used_channels = []
            for _mk, _ch in zip(_info, _channels):
                if _mk.lower() == 'null':
                    continue
                else:
                    _used_channels.append(_ch)
                    
            # load beads image for both drift and ref
            from ..io_tools.load import correct_fov_image
            # Append beads image for both drift and ref
            _Image_dict[_Image_key] = correct_fov_image(
                                        _bead_filename, 
                                        [_drift_channel],
                                        single_im_size=single_im_size,
                                        all_channels=_used_channels,
                                        num_buffer_frames=num_buffer_frames,
                                        num_empty_frames=num_empty_frames,
                                        drift=None, calculate_drift=False,
                                        drift_channel=_drift_channel,
                                        ref_filename=_drift_ref,
                                        correction_folder=correction_folder,
                                        warp_image=True,  # warp bead ims so drift can be applied to ref channel eg.647
                                        illumination_corr=illumination_corr, # warp/coorect bead ims so drift can be applied to ref channel eg.647
                                        bleed_corr=False, 
                                        chromatic_corr=False, 
                                        z_shift_corr=False,
                                        verbose=True,
                                        )[0][0]
            
        # Calculate the drift
        if _precise_align:
            print ('Use precise aligment function to align')
            from ..correction_tools.alignment import align_image
            # make sure the order is correct; check the source function
            _drift, _drift_flag = align_image(_Image_dict['_DriftImage'], 
                                            _Image_dict['RefImage'],
                                            use_autocorr=True, 
                                            drift_channel=_drift_channel)
        else:
            print ('Use only phase cross correlation to align')
            from skimage.registration import phase_cross_correlation
            _drift, _error, _phasediff = phase_cross_correlation(
                                                                  _Image_dict['RefImage'],
                                                                  _Image_dict['_DriftImage'], 
                                                                  upsample_factor= 100,
                                                                  )
        
        return _drift
            



# function to process each FOV
def calculate_drift_FOV (fov_param, _fov_id, 
                         _precise_align = True,
                         _drift_channel = '488',
                         overwrite_drift = False, 
                         save_drift=True, 
                         parallel=False, 
                         num_threads =10, 
                         _overwrite_FOV = False,    # parameters for init FOV
                         _fit_spots_FOV = True,     # parameters for init FOV
                         _warp_images_FOV = False,  # parameters for init FOV
                         _save_images_FOV = False,  # parameters for init FOV
                         drift_filename = '_Drift_byNextIm.csv',
                          ):

    ##########################################
    # Init fov classes to get stored info
    from ..classes.field_of_view import Field_of_View
    fov = Field_of_View (fov_param, _fov_id=_fov_id,
                                              _color_info_kwargs={
                                                  '_color_filename':'Color_Usage',
                                              }, 
                                              _prioritize_saved_attrs=False,
                                              _save_info_to_file=False, # whether overwrite
                                              )

    #fov.parallel = True
    #fov.combo_ref_id = 0
    #fov._process_image_to_spots('combo', 
                                #_load_common_reference=True, _load_with_multiple=False,
                                #_save_images=_save_images_FOV,
                                #_warp_images=_warp_images_FOV, 
                                #_fit_spots=_fit_spots_FOV,
                                #_overwrite_drift=False, _overwrite_image=_overwrite_FOV,
                                #_overwrite_spot=_overwrite_FOV,
                                #_verbose=True)
    # Get filename and folders
    fov_name = fov.fov_name
    #_folder = fov.annotated_folders[_fov_id]
    #_save_filename = os.path.join(_folder, fov_name)
    save_folder = fov_param['save_folder']
    drift_folder = os.path.join(save_folder, 'Drift')

    if not os.path.exists(drift_folder):
        os.makedirs(drift_folder)
        print(f"Generate drift_folder: {drift_folder}")
    else:
        print(f"Use drift_folder: {drift_folder}")
    
    # determine to proceed or not by overwrite arg
    # deinfe/find the alignment results accordingly
    if not _precise_align:
        drift_filename = drift_filename.replace('Drift', 'RoughDrift')

    _drift_savefile = os.path.join(drift_folder, 
        os.path.basename(fov_name).replace('.dax', drift_filename))
    ##########################################
    # Determine if it has been processed
    if os.path.exists(_drift_savefile) and not overwrite_drift:
        print (f'Field of view exists, skip: {_drift_savefile}')
        _drift_df = pd.read_csv(_drift_savefile, index_col =0 )
    else:
        # Get color and channel information from FOV class
        bit_df_melt = load_color_usage_byBit (fov_param, _fov_id, sort_by = 'hyb_number')
        ##########################################
        # Get shared parameters from fov for drift calculation
        fov_annotated_folders = fov.annotated_folders
        fov_name = fov.fov_name
        color_info = fov.color_dic
        # _drift_channel = fov.drift_channel
        _channels = fov.channels
        # load from Dax file
        if hasattr(fov, '_ref_im'):
            _drift_ref = getattr(fov, '_ref_im')
        else:
            _drift_ref = getattr(fov, 'ref_filename')

        single_im_size=fov.shared_parameters['single_im_size']
        num_buffer_frames=fov.shared_parameters['num_buffer_frames']
        num_empty_frames=fov.shared_parameters['num_empty_frames']
        correction_folder=fov.correction_folder
        illumination_corr=fov.shared_parameters['corr_illumination']
        ##########################################
        # Calculate drift for all bits
        bit_df_melt_byHyb = bit_df_melt.drop_duplicates(subset='hyb_number', keep='first')
        hyb_info = np.unique(bit_df_melt_byHyb['hyb_number']).tolist()
        _time = time.time()
        if parallel:
            mp_args = []
            for _hyb_id in list(range(len(hyb_info)))[:]:  # adjust hyb round fro debugbing
                _arg = (
                         _hyb_id, # hyb id that change
                         fov_annotated_folders, 
                         fov_name, 
                         _precise_align,
                         color_info, 
                         _drift_channel, 
                         _drift_ref,
                         _channels, 
                         single_im_size, 
                         num_buffer_frames, 
                         num_empty_frames, 
                         illumination_corr,
                         correction_folder,
                                          )
            
                mp_args.append(_arg)
            import multiprocessing as mp
            print (f"-- Start multi-processing drift correction with {num_threads} threads", end=' ')
            with mp.Pool(num_threads) as drift_pool:
                drift_list = drift_pool.starmap(calculate_drift_from_prior_hyb, mp_args)
                drift_pool.close()
                drift_pool.join()
                drift_pool.terminate()
            print(f"in {time.time()-_time:.3f}s.")
            # clear
            del(mp_args)
        
        else:
            drift_list = []
            print ('-- Calculate drift for each channel sequentially.')
            for _hyb_id in tqdm.tqdm(list(range(len(hyb_info)))[:], desc='Calculating drift channels'): # adjust hyb round fro debugbing
                _drift = calculate_drift_from_prior_hyb(_hyb_id, 
                                 fov_annotated_folders, 
                                 fov_name, 
                                 _precise_align,
                                 color_info, 
                                 _drift_channel, 
                                 _drift_ref,
                                 _channels, 
                                 single_im_size, 
                                 num_buffer_frames, 
                                 num_empty_frames, 
                                 illumination_corr,
                                 correction_folder,)
                drift_list.append(_drift)

        drift_list = np.array(drift_list)
        cumsum_drifts = np.cumsum(drift_list, axis=0)
        #####################################
        # Convert results to dataframe output
        # Assign zxy in correct order depending on whether drift info is available
        def map_to_hyb_number(hyb_number, cumsum_drifts):
            if hyb_number < len(cumsum_drifts):
                return cumsum_drifts[hyb_number, :]
            else:
                return np.array([0,0,0])
        for _drift_order, _drift_col in enumerate(['drift_z', 'drift_x', 'drift_y']):
            bit_df_melt[_drift_col] = bit_df_melt['hyb_number'].apply(lambda x: 
                                                                map_to_hyb_number(x,cumsum_drifts)[_drift_order])
        _drift_df = bit_df_melt.copy(deep=True)
        _drift_df = _drift_df.reset_index(drop=True)
        print(f"Complete drift calculation.")
        #####################################
        # Save result if needed
        if save_drift:
            print (f'Save drift information: {_drift_savefile}')
            _drift_df.to_csv(_drift_savefile)
        
    return _drift_df




# function to calculate drift for a single hyb round relative to the multiple prior round
def calculate_drift_from_multiple_prior_hyb(_hyb_id, 
                                 fov_annotated_folders, 
                                 fov_name, 
                                 _precise_align,
                                 _prior_num,
                                 color_info, 
                                 _drift_channel, 
                                 _drift_ref,
                                 _channels, 
                                 single_im_size, 
                                 num_buffer_frames, 
                                 num_empty_frames, 
                                 illumination_corr,
                                 correction_folder, 
                                 ):
    # No drift for the first cycle
    if _hyb_id == 0:
        _drifts = {f"{_key}_hyb_before" : np.array([0,0,0]) for _key in range(1,_prior_num+1)} 
        return _drifts
    # Calculate drift from the previous hyb
    else:
        # load beads image for both drift and ref
        from ..io_tools.load import correct_fov_image
        _ImageList = []
        for _ind in range(np.max((0, _hyb_id-_prior_num)), _hyb_id+1):   # start Ref from zero
            _bead_folder = fov_annotated_folders[_ind]
            _bead_filename = os.path.join(_bead_folder, fov_name)
            _drift_channel = _drift_channel  
            # get used_channels for this dapi folder:
            _info = color_info[os.path.basename(_bead_folder)]
            _used_channels = []
            for _mk, _ch in zip(_info, _channels):
                if _mk.lower() == 'null':
                    continue
                else:
                    _used_channels.append(_ch)

            # Append beads image for both drift and ref
            _bead_im = correct_fov_image(
                                        _bead_filename, 
                                        [_drift_channel],
                                        single_im_size=single_im_size,
                                        all_channels=_used_channels,
                                        num_buffer_frames=num_buffer_frames,
                                        num_empty_frames=num_empty_frames,
                                        drift=None, calculate_drift=False,
                                        drift_channel=_drift_channel,
                                        ref_filename=_drift_ref,
                                        correction_folder=correction_folder,
                                        warp_image=True,  # warp bead ims so drift can be applied to ref channel eg.647
                                        illumination_corr=illumination_corr, # warp/coorect bead ims so drift can be applied to ref channel eg.647
                                        bleed_corr=False, 
                                        chromatic_corr=False, 
                                        z_shift_corr=False,
                                        verbose=True,
                                        )[0][0]
            _ImageList.append(_bead_im)
        

        _drifts = {}   
        # Calculate the drift
        if _precise_align:
            print ('Use precise aligment function to align')
            from ..correction_tools.alignment import align_image
            for _idx, _im in enumerate(_ImageList[:-1]):
                # make sure the order is correct; check the source function
                _drift, _drift_flag = align_image(_ImageList[-1], # Drfit is always the last
                                                _im, # Each RefImage by order
                                                use_autocorr=True, 
                                                drift_channel=_drift_channel)
                # append result back ward so the key is the hyb gap between Ref and Drift
                _key = int(len(_ImageList[:-1]) - _idx)
                _drifts[f"{_key}_hyb_before"] = _drift


        else:
            print ('Use only phase cross correlation to align')
            from skimage.registration import phase_cross_correlation
            for _idx, _im in enumerate(_ImageList[:-1]):
                _drift, _error, _phasediff = phase_cross_correlation(
                                                                  _im, # Each RefImage by order
                                                                  _ImageList[-1], # Drfit is always the last
                                                                  upsample_factor= 100,
                                                                  )
                # append result back ward so the key is the hyb gap between Ref and Drift
                _key = int(len(_ImageList[:-1]) - _idx)
                _drifts[f"{_key}_hyb_before"] = _drift

        # Fill empty keys (hybs that do not have the prior gap of hybs: hyb# 2 do not have a prior hyb that is 3 gaps away) with nan
        for _key in range(1,_prior_num+1):
            if f"{_key}_hyb_before" in _drifts.keys():
                continue
            else:
                _drifts[f"{_key}_hyb_before"] = np.array([np.nan,np.nan,np.nan])
        
        return _drifts
    



# function to process each FOV using multiple prior hybs
def calculate_multiple_drift_FOV (fov_param, _fov_id, 
                         _precise_align = True,
                         _prior_num =3,
                         _drift_channel = '488',
                         overwrite_drift = False, 
                         save_drift=True, 
                         parallel=False, 
                         num_threads =10, 
                         _overwrite_FOV = False,    # parameters for init FOV
                         _fit_spots_FOV = True,     # parameters for init FOV
                         _warp_images_FOV = False,  # parameters for init FOV
                         _save_images_FOV = False,  # parameters for init FOV
                         drift_filename = '_Drift_byNextIm.csv',
                          ):

    ##########################################
    # Init fov classes to get stored info
    from ..classes.field_of_view import Field_of_View
    fov = Field_of_View (fov_param, _fov_id=_fov_id,
                                              _color_info_kwargs={
                                                  '_color_filename':'Color_Usage',
                                              }, 
                                              _prioritize_saved_attrs=False,
                                              _save_info_to_file=False, # whether overwrite
                                              )

    #fov.parallel = True
    #fov.combo_ref_id = 0
    #fov._process_image_to_spots('combo', 
                                #_load_common_reference=True, _load_with_multiple=False,
                                #_save_images=_save_images_FOV,
                                #_warp_images=_warp_images_FOV, 
                                #_fit_spots=_fit_spots_FOV,
                                #_overwrite_drift=False, _overwrite_image=_overwrite_FOV,
                                #_overwrite_spot=_overwrite_FOV,
                                #_verbose=True)
    # Get filename and folders
    fov_name = fov.fov_name
    #_folder = fov.annotated_folders[_fov_id]
    #_save_filename = os.path.join(_folder, fov_name)
    save_folder = fov_param['save_folder']
    drift_folder = os.path.join(save_folder, 'Drift')

    if not os.path.exists(drift_folder):
        os.makedirs(drift_folder)
        print(f"Generate drift_folder: {drift_folder}")
    else:
        print(f"Use drift_folder: {drift_folder}")
    
    # determine to proceed or not by overwrite arg
    # deinfe/find the alignment results accordingly
    if not _precise_align:
        drift_filename = drift_filename.replace('Drift', 'RoughDrift')

    _drift_savefile = os.path.join(drift_folder, 
        os.path.basename(fov_name).replace('.dax', drift_filename))
    ##########################################
    # Determine if it has been processed
    if os.path.exists(_drift_savefile) and not overwrite_drift:
        print (f'Field of view exists, skip: {_drift_savefile}')
        _drift_df = pd.read_csv(_drift_savefile, index_col =0 )
    else:
        # Get color and channel information from FOV class
        bit_df_melt = load_color_usage_byBit (fov_param, _fov_id, sort_by = 'hyb_number')
        ##########################################
        # Get shared parameters from fov for drift calculation
        fov_annotated_folders = fov.annotated_folders
        fov_name = fov.fov_name
        color_info = fov.color_dic
        #_drift_channel = fov.drift_channel
        _channels = fov.channels
        # load from Dax file
        if hasattr(fov, '_ref_im'):
            _drift_ref = getattr(fov, '_ref_im')
        else:
            _drift_ref = getattr(fov, 'ref_filename')

        single_im_size=fov.shared_parameters['single_im_size']
        num_buffer_frames=fov.shared_parameters['num_buffer_frames']
        num_empty_frames=fov.shared_parameters['num_empty_frames']
        correction_folder=fov.correction_folder
        illumination_corr=fov.shared_parameters['corr_illumination']
        ##########################################
        # Calculate drift for all bits
        bit_df_melt_byHyb = bit_df_melt.drop_duplicates(subset='hyb_number', keep='first')
        hyb_info = np.unique(bit_df_melt_byHyb['hyb_number']).tolist()
        # start with the immediate prior one 
        _time = time.time()
        if parallel:
            mp_args = []
            for _hyb_id in list(range(len(hyb_info)))[:]:  # adjust hyb round fro debugbing
                _arg = (
                         _hyb_id, # hyb id that change
                         fov_annotated_folders, 
                         fov_name, 
                         _precise_align,
                         _prior_num,
                         color_info, 
                         _drift_channel, 
                         _drift_ref,
                         _channels, 
                         single_im_size, 
                         num_buffer_frames, 
                         num_empty_frames, 
                         illumination_corr,
                         correction_folder,
                                          )
            
                mp_args.append(_arg)
            import multiprocessing as mp
            print (f"-- Start multi-processing drift correction with {num_threads} threads", end=' ')
            with mp.Pool(num_threads) as drift_pool:
                drift_list = drift_pool.starmap(calculate_drift_from_multiple_prior_hyb, mp_args)
                drift_pool.close()
                drift_pool.join()
                drift_pool.terminate()
            print(f"in {time.time()-_time:.3f}s.")
            # clear
            del(mp_args)
        
        else:
            drift_list = []
            print ('-- Calculate drift for each channel sequentially.')
            for _hyb_id in tqdm.tqdm(list(range(len(hyb_info)))[:], desc='Calculating drift channels'): # adjust hyb round fro debugbing
                _drifts = calculate_drift_from_multiple_prior_hyb(_hyb_id, 
                                 fov_annotated_folders, 
                                 fov_name, 
                                 _precise_align,
                                 _prior_num,
                                 color_info, 
                                 _drift_channel, 
                                 _drift_ref,
                                 _channels, 
                                 single_im_size, 
                                 num_buffer_frames, 
                                 num_empty_frames, 
                                 illumination_corr,
                                 correction_folder,)
                drift_list.append(_drifts)

        return drift_list





      
        
# function to load drift corected image for indicated hybs
def load_aligned_bead_images  (_hyb_id, _drift,
                                 load_max_projection,
                                 fov_annotated_folders, 
                                 fov_name, 
                                 color_info, 
                                 _drift_channel, 
                                 _drift_ref,
                                 _channels, 
                                 single_im_size, 
                                 num_buffer_frames, 
                                 num_empty_frames, 
                                 illumination_corr,
                                 correction_folder,):
    from ..io_tools.load import correct_fov_image
    _ind = _hyb_id
    print('Load_aligned_bead_images length fov_annotated_folders:', len(fov_annotated_folders), flush=True)
    print('Load_aligned_bead_images fov_annotated_folders', fov_annotated_folders, flush=True)
    print('Load_aligned_bead_images _ind:', _ind, flush=True)
    _bead_folder = fov_annotated_folders[_ind]
    _bead_filename = os.path.join(_bead_folder, fov_name)
    _drift_channel = _drift_channel
    # get used_channels for this dapi folder:
    _info = color_info[os.path.basename(_bead_folder)]
    _used_channels = []
    for _mk, _ch in zip(_info, _channels):
        if _mk.lower() == 'null':
            continue
        else:
            _used_channels.append(_ch)

    aligned_bead_ims = correct_fov_image(
                                        _bead_filename, 
                                        [_drift_channel],
                                        single_im_size=single_im_size,
                                        all_channels=_used_channels,
                                        num_buffer_frames=num_buffer_frames,
                                        num_empty_frames=num_empty_frames,
                                        drift=_drift, calculate_drift=False,
                                        drift_channel=_drift_channel,
                                        ref_filename=_drift_ref,
                                        correction_folder=correction_folder,
                                        warp_image=True,  # warp bead ims so drift can be applied to ref channel eg.647
                                        illumination_corr=illumination_corr, # warp/coorect bead ims so drift can be applied to ref channel eg.647
                                        bleed_corr=False, 
                                        chromatic_corr=False, 
                                        z_shift_corr=False,
                                        verbose=True,
                                        )[0][0]
    
    if load_max_projection:
        max_aligned_bead_ims_xy = np.max(aligned_bead_ims, axis=0)
        max_aligned_bead_ims_zy = np.max(aligned_bead_ims, axis=1)
        return [max_aligned_bead_ims_xy, max_aligned_bead_ims_zy]
    else:
        return aligned_bead_ims




# function to process each FOV
def aligned_bead_stack_FOV (fov_param, 
                            _fov_id, 
                            hyb_drift_dict,
                            load_max_projection =True,
                         _drift_channel = '488',
                         overwrite_bead_im = False, 
                         save_bead_im=True, 
                         parallel=False, 
                         num_threads =10, 
                         _overwrite_FOV = False,    # parameters for init FOV
                         _fit_spots_FOV = True,     # parameters for init FOV
                         _warp_images_FOV = False,  # parameters for init FOV
                         _save_images_FOV = False,  # parameters for init FOV
                         aligned_bead_filename = '_Aligned_bead_stack.tif',
                          ):

    ##########################################
    # Init fov classes to get stored info
    from ..classes.field_of_view import Field_of_View
    fov = Field_of_View (fov_param, _fov_id=_fov_id,
                                              _color_info_kwargs={
                                                  '_color_filename':'Color_Usage',
                                              }, 
                                              _prioritize_saved_attrs=False,
                                              _save_info_to_file=False, # whether overwrite
                                              )

    # Get filename and folders
    fov_name = fov.fov_name
    #_folder = fov.annotated_folders[_fov_id]
    #_save_filename = os.path.join(_folder, fov_name)
    save_folder = fov_param['save_folder']
    drift_folder = os.path.join(save_folder, 'Drift')

    if not os.path.exists(drift_folder):
        os.makedirs(drift_folder)
        print(f"Generate drift_folder: {drift_folder}")
    else:
        print(f"Use drift_folder: {drift_folder}")
    
    # determine to proceed or not by overwrite arg
    aligned_bead_savefile = os.path.join(drift_folder, 
        os.path.basename(fov_name).replace('.dax', aligned_bead_filename))
    aligned_bead_savefile_xy = aligned_bead_savefile.replace('.tif', '_xy.tif')
    aligned_bead_savefile_zy = aligned_bead_savefile.replace('.tif', '_zy.tif')

    ##########################################
    # Determine if it has been processed
    if (os.path.exists(aligned_bead_savefile_xy) or os.path.exists(aligned_bead_savefile_zy)) and not overwrite_bead_im:
        print (f'Field of view exists, skip: {aligned_bead_savefile}')
        _aligned_bead_stack = skimage.io.imread(aligned_bead_savefile)
    else:
        ##########################################
        # Get shared parameters from fov for drift calculation
        fov_annotated_folders = fov.annotated_folders
        fov_name = fov.fov_name
        color_info = fov.color_dic
        # _drift_channel = fov.drift_channel
        _channels = fov.channels
        # load from Dax file
        if hasattr(fov, '_ref_im'):
            _drift_ref = getattr(fov, '_ref_im')
        else:
            _drift_ref = getattr(fov, 'ref_filename')

        single_im_size=fov.shared_parameters['single_im_size']
        num_buffer_frames=fov.shared_parameters['num_buffer_frames']
        num_empty_frames=fov.shared_parameters['num_empty_frames']
        correction_folder=fov.correction_folder
        illumination_corr=fov.shared_parameters['corr_illumination']
        ##########################################
        # Load bead images for all bits
        _time = time.time()
        if parallel:
            mp_args = []
            for _hyb_id, _drift in hyb_drift_dict.items():  # adjust hyb round for debugbing
                _arg = (
                         _hyb_id, # hyb id that change
                         _drift,  # drfit corresponding to the hyb
                         load_max_projection,
                         fov_annotated_folders, 
                         fov_name, 
                         color_info, 
                         _drift_channel, 
                         _drift_ref,
                         _channels, 
                         single_im_size, 
                         num_buffer_frames, 
                         num_empty_frames, 
                         illumination_corr,
                         correction_folder,
                                          )
            
                mp_args.append(_arg)
            import multiprocessing as mp
            print (f"-- Start multi-processing bead image alignment with {num_threads} threads", end=' ')
            with mp.Pool(num_threads) as align_pool:
                align_bead_list = align_pool.starmap(load_aligned_bead_images, mp_args)
                align_pool.close()
                align_pool.join()
                align_pool.terminate()
            print(f"in {time.time()-_time:.3f}s.")
            # clear
            del(mp_args)
        
        else:
            align_bead_list = []
            print ('-- Align bead images for each channel sequentially.')
            for _hyb_id, _drift in hyb_drift_dict.items():  # adjust hyb round for debugbing
                _aligned_bead_im = load_aligned_bead_images(_hyb_id, # hyb id that change
                                                            _drift,  # drfit corresponding to the hyb
                                                            load_max_projection,
                                                            fov_annotated_folders, 
                                                            fov_name, 
                                                            color_info, 
                                                            _drift_channel, 
                                                            _drift_ref,
                                                            _channels, 
                                                            single_im_size, 
                                                            num_buffer_frames, 
                                                            num_empty_frames, 
                                                            illumination_corr,
                                                            correction_folder,)
                align_bead_list.append(_aligned_bead_im)

        # Assemble max projection images for all hybs
        if load_max_projection:
            _aligned_bead_stack_xy = np.array([_ims[0] for _ims in align_bead_list])
            _aligned_bead_stack_zy = np.array([_ims[1] for _ims in align_bead_list])
        else:
            _aligned_bead_stack_xy = np.array([np.max(_ims,axis=0) for _ims in align_bead_list])
            _aligned_bead_stack_zy = np.array([np.max(_ims,axis=1) for _ims in align_bead_list])

        print(f"Complete bead alignment.")
        #####################################
        # Save result if needed
        if save_bead_im:
            print (f'Save aligned bead stack: {aligned_bead_savefile}')
            skimage.io.imsave(aligned_bead_savefile_xy, _aligned_bead_stack_xy)
            skimage.io.imsave(aligned_bead_savefile_zy, _aligned_bead_stack_zy)
        
    return [_aligned_bead_stack_xy, _aligned_bead_stack_zy]