import PIL.Image
import numpy as np
import cv2
import json
import os
import argparse
import shutil

# Set environment variables for GDAL and PROJ libraries
os.environ['GDAL_DATA'] = '/home/miniconda3/share'
os.environ['PROJ_LIB'] = '/home/miniconda3/share/proj'
os.environ['PROJ_DATA'] = '/home/miniconda3/share/proj'

class IMAGE:
    pass

class TILE:
    pass

class BBOX:
    pass


# Function to add padding to an image
def padding(source,padding_top,padding_bottom,padding_left,padding_right,color):
    """
    Add padding to the source image.
    Args:
        source: Input image as a NumPy array.
        padding_top, padding_bottom, padding_left, padding_right: Padding dimensions.
        color: Color to fill the padding areas.
    Returns:
        Padded image.
    """

    h,w,channel = source.shape
    new_h = h + padding_top+padding_bottom
    new_w = w + padding_left + padding_right
    print([h,w])
    print([new_h,new_w])
    # create new image of desired size and color (blue) for padding
    assert len(color) == channel
    result = np.full((new_h,new_w, channel), color, dtype=np.uint8)

    # compute center offset


    # copy img image into center of result image
    result[padding_top:padding_top+h, 
    padding_left:padding_left+w] = source

    # view result
    return result


# Class to split an image into uniform tiles and handle related operations
class ImageMask2Tiles():
    def __init__(self,imgShape,tileSize,isPadding = False, paddingColor = None):
        """
        Initialize the tiling process.
        Args:
            imgShape: Shape of the input image.
            tileSize: Desired size of the tiles.
            isPadding: Boolean indicating whether to add padding.
            paddingColor: Color to use for padding if enabled.
        """

        self.imgShape = imgShape
        self.tileSize = tileSize
        self.horizontal_tile_number = int(imgShape[1]/tileSize[1])
        self.vertical_tile_number = int(imgShape[0]/tileSize[0])
        self.new_horizontal = self.horizontal_tile_number*tileSize[1]
        self.new_vertical = self.vertical_tile_number*tileSize[0]

        horizontal_extra = imgShape[1]-self.new_horizontal
        vertical_extra = imgShape[0]-self.new_vertical
        
        self.isPadding = isPadding
        self.paddingColor = paddingColor
        self.padding_top = 0
        self.padding_left = 0
        self.padding_bottom = 0
        self.padding_right = 0
        
        if vertical_extra != 0:
            self.padding_bottom = (self.vertical_tile_number+1)*tileSize[0]-imgShape[0]
            if isPadding:
                self.new_vertical = (self.vertical_tile_number+1)*tileSize[0]
                self.vertical_tile_number += 1
                vertical_extra = 0
                
        if horizontal_extra != 0:
            self.padding_right = (self.horizontal_tile_number+1)*tileSize[1]-imgShape[1]
            if isPadding:
                self.new_horizontal = (self.horizontal_tile_number+1)*tileSize[1]
                self.horizontal_tile_number += 1                
                horizontal_extra = 0
                
                

                
        if isPadding:
            print("padding:")
            print([self.padding_top,self.padding_bottom,self.padding_left,self.padding_right])
        print((self.horizontal_tile_number,self.vertical_tile_number))

        self.horizontal_offset = int(horizontal_extra/2)
        self.vertical_offset = int(vertical_extra/2)
        self.horizontal_end = self.horizontal_offset+self.new_horizontal
        self.vertical_end = self.vertical_offset+self.new_vertical
        if len(imgShape) == 3:
            self.new_imgShape = (self.new_vertical,self.new_horizontal,imgShape[2])
        else:
            self.new_imgShape = (self.new_vertical,self.new_horizontal)
        print("new image shape:")
        print(self.new_imgShape)
    
    def assemblyImg(self,result_tiles):
        max_x = self.horizontal_tile_number
        max_y = self.vertical_tile_number
        originalShape = self.new_imgShape
        assembledImg = np.zeros(originalShape)
        x_pointer = 0
        y_pointer = 0
    
        print(result_tiles.keys())
        for x in range(max_x):
            y_pointer = 0
            for y in range(max_y):
                tile = result_tiles[(x,y)]
                h_t,w_t = tile.shape[0],tile.shape[1]
                assembledImg[y_pointer:y_pointer+h_t,x_pointer:x_pointer+w_t] = tile
                y_pointer = y_pointer+h_t
            x_pointer = x_pointer+w_t

        assert (x_pointer - originalShape[1]) <=1
        assert (y_pointer - originalShape[0]) <=1
        if self.isPadding:
            assembledImg = assembledImg[self.padding_top:self.imgShape[0],self.padding_left:self.imgShape[1],:]
        return assembledImg



            
    

    def getTiles(self,im):
       
        h,w = (self.new_imgShape[0],self.new_imgShape[1])

        tiles = {}
        last_x = 0
        last_y = 0

        M = self.tileSize[1]
        N = self.tileSize[0]
        idx_x = 0
        for x in range(0,w,M):
            last_x = x
            idx_y = 0
            for y in range(0,h,N):
                tiles[(idx_x,idx_y)] = im[y:y+N,x:x+M]
                last_y = y
                idx_y += 1
                
            idx_x += 1

        return tiles
   
    def processImg(self,image):
        h,w,c = image.shape
        if not (self.imgShape == image.shape):
            print("input shape:")
            print(image.shape)
            print("expected:")
            print(self.imgShape)
            assert (self.imgShape == image.shape)
            
            
        if self.isPadding:
            image = padding(image,self.padding_top,self.padding_bottom,self.padding_left,self.padding_right,self.paddingColor)
            print("padded image size:")
            print(image.shape)
        croped_image = image[self.vertical_offset:self.vertical_end, self.horizontal_offset:self.horizontal_end]
        tiles = self.getTiles(croped_image)
        return tiles



if __name__=="__main__":


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('img_dir', type=str)
    parser.add_argument('img_out_dir', type=str)

    args = parser.parse_args()

    img_dir = args.img_dir + '/'
    img_out_dir = args.img_out_dir + '/'

    if os.path.exists(img_out_dir):
        shutil.rmtree(img_out_dir)
    os.mkdir(img_out_dir)


    prefix = ''
    for fp in os.listdir(img_dir):
        fn = fp[:-4]
        assert fn+'.png' in os.listdir(img_dir)
        img_fp = img_dir+fn+'.png'
        print(img_fp)
        
        
        img_shape = cv2.imread(img_fp).shape
        img_tiler = ImageMask2Tiles(img_shape,(512,512),isPadding = True, paddingColor = [255,255,255]) #Tile size is 512*512. Can be changed.
    
        img = cv2.imread(img_fp)
        tiles = img_tiler.processImg(img)
        tiles_keys = tiles.keys()
        for key in tiles_keys:
            fp_write = img_out_dir+fn+'_'+str(key[0])+'_'+str(key[1])+'.jpg'
            print(fp_write)
            cv2.imwrite(fp_write, tiles[key])
        assembled_img = img_tiler.assemblyImg(tiles)
        #cv2.imwrite(img_out_dir+fn+'_assembled'+'.jpg', assembled_img)

    
