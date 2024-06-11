#!/usr/bin/python
import ome_types
import numpy as np
import pandas as pd
from pathlib import Path
#from PIL import Image
#import PIL
#from PIL.TiffTags import TAGS
import tifffile as tifff
from bs4 import BeautifulSoup
import os
from uuid import uuid4
import copy
from ome_types import from_tiff,to_xml
from ome_types.model import OME,Image,Instrument,Pixels,TiffData,Channel,Plane
from ome_types.model.tiff_data import UUID
from ome_types.model.simple_types import PixelsID,ImageID,Color,ChannelID
import ome_types.model

import platform
import argparse


#---CLI-BLOCK---#
parser=argparse.ArgumentParser()


parser.add_argument('-i',
                    '--input',
                    nargs='*',
                    required=True,
                    help='Ordered pairs of paths to the Antigen and corresponding Bleaching cycle, e.g [002_AntigenCycle,001_BleachCycle].'
                    )

parser.add_argument('-o',
                    '--output',
                    required=True,
                    help='Directory to save the stack.'
                    )

parser.add_argument('-c',
                    '--cycle',
                    required=True,
                    type=int,
                    help='Number of the antigen cycle.'
                    )

parser.add_argument('-rm',
                    '--ref_marker',
                    required=False,
                    type=str,
                    default='DAPI',
                    help='Name of the reference marker.Default value is DAPI.'
                    )

parser.add_argument('-he',
                    '--hi_exposure_only',
                    action='store_true',
                    help='Activate this flag to extract only the set of images with the highest exposure time.'
                    )


args=parser.parse_args()
#---END_CLI-BLOCK---#



#------FORMAT INPUT BLOCK-------#
device_name='MACSIMA'
antigen_dir=Path(args.input[0])
bleach_dir=Path(args.input[1])
stack_path=Path(args.output)
ref_marker=args.ref_marker
cycle_no=args.cycle

#------ ENDINPUT BLOCK----#

if os.path.exists(stack_path):
    pass
else:
    os.mkdir(stack_path)


#---- HELPER FUNCTIONS ----#

def antigen_cycle_info(antigen_folder=antigen_dir,ref_marker=ref_marker):

    images=list(filter(lambda x: x.endswith('.tif'),os.listdir(antigen_folder)))

    cycle_info={'img_full_path':[],
                'image':images,
                'marker':[],
                'filter':[],
                'rack':[],
                'well':[],
                'roi':[],
                'fov':[],
                'exposure':[]
               }

    for im in images:
        
        #Extraction of marker name info and acquisition info (rack,well,exposure,roi)
        marker_info=im.split('AntigenCycle')[-1].split('__')
        acq_info=im.split('sensor')[-1].split('_')
        #add the information to the cycle_info dictionary
        #img full path
        cycle_info['img_full_path'].append(os.path.join(antigen_folder,im))
        #marker and fluorophore ()
        m=marker_info[0].strip('_')
        if ref_marker in m:
            cycle_info['marker'].append(ref_marker)
            cycle_info['filter'].append(ref_marker)
        else:
            cycle_info['marker'].append(m)
            cycle_info['filter'].append(marker_info[-1].split('bit')[0].split("_")[-2])

        #rack
        cycle_info['rack'].append(acq_info[2].split('-')[-1])
        #well
        cycle_info['well'].append(acq_info[3].split('-')[-1])
        #roi
        cycle_info['roi'].append(acq_info[4].split('-')[-1])
        #fov, i.e. tiles
        cycle_info['fov'].append(acq_info[5].split('-')[-1])
        #exposure
        exposure=cycle_info['exposure'].append(acq_info[6].split('-')[-1].strip('.tif'))        
        
    info=pd.DataFrame(cycle_info)
    

    markers= info['marker'].unique()
    markers_subset=np.setdiff1d(markers,[ref_marker])
    #insert the exposure level colum at the end of the info dataframe
    info.insert(len(cycle_info),'exposure_level',np.zeros(info.shape[0]))
    #label the exposure level of the reference markers with "ref"
    info.loc[ info['marker']==ref_marker, 'exposure_level']='ref'

    for m in markers_subset:
        exposure=info.loc[info['marker']==m]['exposure'].unique()
        val=pd.to_numeric(exposure)
        val=np.sort(val)
        for level,value in enumerate(val):
            info.loc[ (info['marker']==m) & (info['exposure']==str(value)), 'exposure_level']=level+1
    
            
    #change the data type to numeric
    info['rack']=pd.to_numeric(info['rack'],downcast='unsigned')
    info['well']=pd.to_numeric(info['well'],downcast='unsigned')
    info['roi']=pd.to_numeric(info['roi'],downcast='unsigned')
    info['fov']=pd.to_numeric(info['fov'],downcast='unsigned')
    info['exposure']=pd.to_numeric(info['exposure'],downcast='unsigned')
    
    return info


def bleach_cycle_info(bleach_folder=bleach_dir,ref_marker=ref_marker,antigen_cycle_no=cycle_no):

    bleach_cycle=f'{antigen_cycle_no-1:03d}'

    images=list(filter(lambda x: x.endswith('.tif'),os.listdir(bleach_folder)))

    cycle_info={'img_full_path':[],
                'image':images,
                'marker':[],
                'filter':[],
                'rack':[],
                'well':[],
                'roi':[],
                'fov':[],
                'exposure':[]
               }

    for im in images:
        
        marker_info=im.split('BleachCycle')[-1].split('_')
        acq_info=im.split('sensor')[-1].split('_')
        #add the information to the cycle_info dictionary
        #img full path
        cycle_info['img_full_path'].append(os.path.join(bleach_folder,im))
        #marker and fluorophore
        #marker_info=['', 'DAPI', 'V0', 'DAPI', '16bit', 'M-20x-S Fluor full sensor', 'B-1', 'R-1', 'W-2', 'G-1', 'F-10', 'E-16.0.tif']
        m=marker_info[1]
        if m==ref_marker:
            cycle_info['marker'].append(ref_marker)
        else:
            cycle_info['marker'].append('_'.join([bleach_cycle,'bleach',marker_info[3]]))
        cycle_info['filter'].append(marker_info[3])
        #rack
        cycle_info['rack'].append(acq_info[2].split('-')[-1])
        #well
        cycle_info['well'].append(acq_info[3].split('-')[-1])
        #roi
        cycle_info['roi'].append(acq_info[4].split('-')[-1])
        #fov, i.e. tiles
        cycle_info['fov'].append(acq_info[5].split('-')[-1])
        #exposure
        exposure=cycle_info['exposure'].append(acq_info[6].split('-')[-1].strip('.tif'))        
        
    info=pd.DataFrame(cycle_info)
    

    markers= info['filter'].unique()
    markers_subset=np.setdiff1d(markers,[ref_marker])
    info.insert(len(cycle_info),'exposure_level',np.zeros(info.shape[0]))
    info.loc[ info['marker']==ref_marker, 'exposure_level']='ref'
    #info.to_csv(os.path.join(cycles_path,'test_bleach.csv'))


    for m in markers_subset:
        exposure=info.loc[info['filter']==m]['exposure'].unique()
        val=pd.to_numeric(exposure)
        val=np.sort(val)
        for level,value in enumerate(val):
            info.loc[ (info['filter']==m) & (info['exposure']==str(value)), 'exposure_level']=level+1
    
            

    info['rack']=pd.to_numeric(info['rack'],downcast='unsigned')
    info['well']=pd.to_numeric(info['well'],downcast='unsigned')
    info['roi']=pd.to_numeric(info['roi'],downcast='unsigned')
    info['fov']=pd.to_numeric(info['fov'],downcast='unsigned')
    info['exposure']=pd.to_numeric(info['exposure'],downcast='unsigned')
    
    return info

def create_dir(dir_path):
    if os.path.exists(dir_path):
        pass
    else:
        os.mkdir(dir_path)

def create_ome(img_info,xy_tile_positions_units,img_path):
    img_name=img_info['name']
    device_name=img_info['device']
    no_of_channels=img_info['no_channels']
    img_size=img_info['xy_img_size_pix']
    markers=img_info['markers']
    exposures=img_info['exposure_times']
    bit_depth=img_info['bit_depth']
    pixel_size=img_info['pix_size']
    pixel_units=img_info['pix_units']
    sig_bits=img_info['sig_bits']
    if pixel_units=='mm':
        pixel_size=1000*pixel_size
        #pixel_units='um'
    no_of_tiles=len(xy_tile_positions_units)
    tifff.tiffcomment(img_path,'')
    #--Generate tiff_data_blocks--#
    tiff_block=[]
    #UUID=ome_types.model.tiff_data.UUID(file_name=img_name,value=uuid4().urn)
    unique_identifier=UUID(file_name=img_name,value=uuid4().urn)
    for ch in range(0,no_of_channels):
        tiff_block.append(TiffData(first_c=ch,
                                    ifd=ch,
                                    plane_count=1,
                                    uuid=unique_identifier
                                )
                         
                         )
    #--Generate planes block (contains the information of each tile)--#
    plane_block=[]
    #length_units=ome_types.model.simple_types.UnitsLength('µm')
    for ch in range(0,no_of_channels):
        plane_block.append(Plane(the_c=ch,
                                 the_t=0,
                                 the_z=0,
                                 position_x=0,#x=0 is just a place holder
                                 position_y=0,#y=0 is just a place holder
                                 position_z=0,
                                 exposure_time=0
                                 #position_x_unit=pixel_units,
                                 #position_y_unit=pixel_units
                                )
                          )
    #--Generate channels block--#
    chann_block=[]
    for ch in range(0,no_of_channels):
        chann_block.append(Channel(id=ChannelID('Channel:{x}'.format(x=ch)),
                                   color=Color((255,255,255)),
                                   emission_wavelength=1,#place holder
                                   excitation_wavelength=1,#place holder
                               
                                  )
                          )
    #--Generate pixels block--#
    pix_block=[]
    ifd_counter=0
    for t in range(0,no_of_tiles):
        template_plane_block=copy.deepcopy(plane_block)
        template_chann_block=copy.deepcopy(chann_block)
        template_tiffdata_block=copy.deepcopy(tiff_block)
        for ch,mark in enumerate(markers):
            template_plane_block[ch].position_x=xy_tile_positions_units[t][0]
            template_plane_block[ch].position_y=xy_tile_positions_units[t][1]
            template_plane_block[ch].exposure_time=exposures[ch]
            template_chann_block[ch].id='Channel:{y}:{x}:{marker_name}'.format(x=ch,y=100+t,marker_name=mark)
            template_tiffdata_block[ch].ifd=ifd_counter
            ifd_counter+=1
        pix_block.append(Pixels(id=PixelsID('Pixels:{x}'.format(x=t)),
                                dimension_order=ome_types.model.pixels.DimensionOrder('XYCZT'),
                                size_c=no_of_channels,
                                size_t=1,
                                size_x=img_size[0],
                                size_y=img_size[1],
                                size_z=1,
                                type=bit_depth,
                                big_endian=False,
                                channels=template_chann_block,
                                interleaved=False,
                                physical_size_x=pixel_size,
                                #physical_size_x_unit=pixel_units,
                                physical_size_y=pixel_size,
                                #physical_size_y_unit=pixel_units,
                                physical_size_z=1.0,
                                planes=template_plane_block,
                                significant_bits=sig_bits,
                                tiff_data_blocks=template_tiffdata_block
                               )
                        )
    #--Generate image block--#
    img_block=[]
    for t in range(0,no_of_tiles):
        img_block.append(Image(id=ImageID('Image:{x}'.format(x=t)),
                               pixels=pix_block[t]
                              )
                        )
    #--Create the OME object with all prebiously defined blocks--#
    ome_custom=OME()
    ome_custom.creator=" ".join([ome_types.__name__,
                                 ome_types.__version__,
                                 '/ python version-',
                                 platform.python_version()
                                ]
                               )
    ome_custom.images=img_block
    ome_custom.uuid=uuid4().urn
    ome_xml=to_xml(ome_custom)
    tifff.tiffcomment(img_path,ome_xml)
    
def setup_coords(x,y,pix_units):
    if pix_units=='mm':
        x_norm=1000*(np.array(x)-np.min(x))#/pixel_size
        y_norm=1000*(np.array(y)-np.min(y))#/pixel_size
    x_norm=np.rint(x_norm).astype('int')
    y_norm=np.rint(y_norm).astype('int')
    #invert y
    y_norm=np.max(y_norm)-y_norm
    xy_tile_positions=[(i, j) for i,j in zip(x_norm,y_norm)]
    return xy_tile_positions

def tile_position(metadata_string):
    #meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
    #ome=BeautifulSoup(meta_dict['ImageDescription'][0],'xml')
    ome=BeautifulSoup(metadata_string,'xml')
    x=float(ome.StageLabel["X"])
    y=float(ome.StageLabel["Y"])
    stage_units=ome.StageLabel["XUnit"]
    pixel_size=float(ome.Pixels['PhysicalSizeX'])
    pixel_units=ome.Pixels['PhysicalSizeXUnit']
    bit_depth=ome.Pixels['Type']
    significantBits=int(ome.Pixels['SignificantBits'])
    tile_info={'x':x,
               'y':y,
               'stage_units':stage_units,
               'pixel_size':pixel_size,
               'pixel_units':pixel_units,
               'bit_depth':bit_depth,
               'sig_bits':significantBits
              }
    return tile_info

def create_stack(info,exp_level=1,
                 antigen_cycle_no=cycle_no,
                 cycle_folder='',
                 isbleach=False,
                 offset=0,
                 device=device_name,
                 ref_marker=ref_marker,
                 results_path=stack_path):
    
    racks=info['rack'].unique()
    wells=info['well'].unique()
    antigen_cycle=f'{antigen_cycle_no:03d}'
    if isbleach:
        cycle_prefix='Bleach'
    else:
        cycle_prefix='Antigen'

    workdir=Path(cycle_folder) 

    with tifff.TiffFile(info['img_full_path'][0]) as tif:
        img_ref=tif.pages[0].asarray()
    width=img_ref.shape[1]
    height=img_ref.shape[0]
    dtype_ref=img_ref.dtype
    ref_marker= list(info.loc[ info['exposure_level']=='ref', 'marker'])[0]
    markers= info['marker'].unique()
    markers_subset=np.setdiff1d(markers,[ref_marker])
    sorted_markers=np.insert(markers_subset,0,ref_marker)
    sorted_filters=[]
    for m in sorted_markers:
            sorted_filters.append(info.loc[info['marker']==m,'filter'].tolist()[0])
    if isbleach:
        target_name='filters'
        target='_'.join(sorted_filters)
    else:
        target_name='markers'
        target='_'.join(sorted_markers)

    
    for r in racks:

        output_levels=[]
        rack_no='rack_{n}'.format(n=f'{r:02d}')
        output_levels.append(rack_no)

        for w in wells:

            well_no='well_{n}'.format(n=f'{w:02d}')
            output_levels.append(well_no)
            groupA=info.loc[(info['rack']==r) & (info['well']==w)]
            rois=groupA['roi'].unique()
            
            for roi in rois:

                roi_no='roi_{n}'.format(n=f'{roi:02d}')
                exp_level_no='exp_level_{n}'.format(n=f'{exp_level:02d}')
                output_levels.append(roi_no)
                output_levels.append(exp_level_no)


                counter=0
                groupB=groupA.loc[groupA['roi']==roi]
                A=groupB.loc[(groupB['exposure_level']=='ref')]
                stack_size_z=A.shape[0]*len(markers)
                fov_id=groupB.loc[groupB['marker']==ref_marker,'fov'].unique()
                fov_id=np.sort(fov_id)
                
                stack_name='cycle-{C}-{prefix}-exp-{E}-rack-{R}-well-{W}-roi-{ROI}-{T}-{M}.{img_format}'.format(C=f'{(offset+antigen_cycle_no):03d}',
                                                                                                       prefix=cycle_prefix,
                                                                                                       E=exp_level,
                                                                                                       R=r,
                                                                                                       W=w,
                                                                                                       ROI=roi,
                                                                                                       T=target_name,#markers or filters
                                                                                                       M=target,
                                                                                                       img_format='ome.tiff'
                                                                                                      )


                X=[]
                Y=[]
                stack=np.zeros((stack_size_z,height,width),dtype=dtype_ref)
                
                exposure_per_marker=[]
                exp=groupB.loc[(groupB['marker']==ref_marker) & (groupB['fov']==1) & (groupB['exposure_level']=='ref'),'exposure'].tolist()[0]
                exposure_per_marker.append(exp)
                for s in markers_subset:
                    exp=groupB.loc[(groupB['marker']==s) & (groupB['fov']==1) & (groupB['exposure_level']==exp_level),'exposure'].tolist()[0]
                    exposure_per_marker.append(exp)
                
                for tile in fov_id:
                    
                    img_ref=groupB.loc[(groupB['marker']==ref_marker) & (groupB['fov']==tile) ,'img_full_path'].tolist()
                    
                
                    if len(img_ref)>0:

                        with tifff.TiffFile(img_ref[0]) as tif:
                            img=tif.pages[0].asarray()
                            metadata=tif.ome_metadata

                        stack[counter,:,:]=img
                        tile_data=tile_position(metadata)
                        X.append(tile_data['x'])
                        Y.append(tile_data['y'])
                        counter+=1
                        
                        for m in markers_subset:
                            img_marker=groupB.loc[(groupB['marker']==m) & (groupB['fov']==tile) & (groupB['exposure_level']==exp_level),'img_full_path'].tolist()[0]
                            img=tifff.imread(img_marker)
                            stack[counter,:,:]=img
                            counter+=1
                

                output_folders_path=stack_path / '--'.join(output_levels) / 'raw'

                if os.path.exists(output_folders_path):
                    pass
                else:
                    os.makedirs(output_folders_path)

                final_stack_path=os.path.join(output_folders_path,stack_name)
                tifff.imwrite(final_stack_path, stack, photometric='minisblack')  
                img_info={'name':stack_name,
                          'device':device_name,
                          'no_channels':len(markers),
                          'markers':sorted_markers,
                          'filters':sorted_filters,
                          'exposure_times':exposure_per_marker,
                          'xy_img_size_pix':(width,height),
                          'pix_size':tile_data['pixel_size'],
                          'pix_units':tile_data['pixel_units'],
                          'bit_depth':tile_data['bit_depth'],
                          'sig_bits':tile_data['sig_bits']
                          }
                create_ome(img_info,setup_coords(X,Y,img_info['pix_units']),img_path=final_stack_path)
    
    return img_info


def main(): 
    
    antigen_info=antigen_cycle_info(antigen_folder=antigen_dir)
    bleach_info=bleach_cycle_info(bleach_folder=bleach_dir)
    exp=antigen_info['exposure_level'].unique()
    exp=exp[exp!='ref']
    exp.sort()

    if args.hi_exposure_only:
        exp=[max(exp)]
    else:
         pass
        
            
    for e in exp:
        antigen_stack_info=create_stack(antigen_info,antigen_cycle_no=cycle_no,exp_level=e)
        bleach_stack_info=create_stack(bleach_info,antigen_cycle_no=cycle_no,isbleach=True,exp_level=e)

            

if __name__=='__main__':
    main()
            
            
    
    