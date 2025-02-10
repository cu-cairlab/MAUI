import argparse
import pandas as pd
import math
import utm
import pyproj
import geopandas as gpd
import numpy as np
from shapely import wkt

# Helper function 1
# Convert lat/lon coordinates to UTM

def convertUTM(utme,utmn,lat_check,lon_check,zone,T,lat,lon):
    x=0
    y=len(lat)
    while x < y:
            utme[x], utmn[x], T[x], zone[x]= utm.from_latlon(lat[x],lon[x])
            lat_check[x], lon_check[x]= utm.to_latlon(utme[x], utmn[x], T[x], zone[x])
            if lat_check[x] < lat[x]+1 or lat_check[x] > lat-1 and lon_check[x] > lon[x]-1 or lon_check[x] < lon[x]+1 :
                x=x+1
                continue
            else:
                print('Conversion Failed')
    return utme,utmn,lat_check,lon_check

# Helper function 2
# Calculate panel vertex coordinates  

def rectangle_coordinates(row,utme,utmn,top_lefte,bot_lefte,top_leftn,bot_leftn,w,panel_num):
    i=panel_num
    j=0
    z=1
    r=0
    # row = row.astype(int) #updated
    while r < max(row):
            delta_utme =abs(utme[j] - utme[i])
            delta_utmn = abs(utmn[j]-utmn[i])
            pythag = math.sqrt(delta_utme**2 +delta_utmn**2)
            angle = math.asin(delta_utmn/pythag)
            y_ang = 90-math.degrees(angle)
            d_utme=abs(utme[j] - utme[z])
            d_utmn= abs(utmn[j]-utmn[z])
            pythag = math.sqrt(d_utme**2 +d_utmn**2)
            y = (w/2)*(math.sin(math.radians(y_ang)))
            x = (w/2)*math.sin(angle)
            while j <= i:
                top_lefte[j] = utme[j] + x
                bot_lefte[j] = utme[j] - x
                top_leftn[j] = utmn[j] + y
                bot_leftn[j] = utmn[j] - y
                j=j+1
            i=i+panel_num+1
            z=z+panel_num+1
            r=r+1
    return top_lefte, bot_lefte, top_leftn, bot_leftn

# Helper function 3
# Convert UTM coordinates back to latitude/longitude  

def toLatLon(utmn, top_lefte, top_leftn, bot_lefte, bot_leftn, T,zone):
    rectlon_bot = [0]*len(utmn)#_conv)
    rectlat_top = [0]*len(utmn)#_conv)
    rectlat_bot = [0]*len(utmn)#_conv)
    rectlon_top = [0]*len(utmn)#_conv)
    
    y = len(utmn)#_conv) 
    x=0
    j=0
    while x < y:
        rectlat_top[x], rectlon_top[x]= utm.to_latlon(top_lefte[j], top_leftn[j], T[j], zone[j])
        rectlat_bot[x], rectlon_bot[x]= utm.to_latlon(bot_lefte[j], bot_leftn[j], T[j], zone[j])
        x=x+1
        j=j+1
    
    return rectlon_bot, rectlon_top, rectlat_bot, rectlat_top

# Helper function 4
# get coordinates of four corners of panel polygon

def fourcourners(rectlon,rectlon_top, rectlon_bot, rectlat, rectlat_top, rectlat_bot, final_panel, row, panel_id):
    x=0
    y=1
    b=0
    while y <= len(rectlon):
        rectlon[x] = rectlon_top[b]
        rectlon[y] = rectlon_bot[b]
        rectlat[x] = rectlat_top[b]
        rectlat[y] = rectlat_bot[b]
        final_panel[x] = panel_id[b+1]
        final_panel[y] = panel_id[b+1]
        
        x=2+x
        y=y+2
        b=b+1
        
        rectlon[x] = rectlon_top[b]
        rectlon[y] = rectlon_bot[b]
        rectlat[x] = rectlat_top[b]
        rectlat[y] = rectlat_bot[b]
        final_panel[x] = panel_id[b]
        final_panel[y] = panel_id[b]
        
        if b +1 >= len(rectlon_top) or y>=len(rectlon) or x>=len(rectlon):
            break
        
        else:
            rectlon[x] = rectlon_top[b]
            rectlon[y] = rectlon_bot[b]
            rectlat[x] = rectlat_top[b]
            rectlat[y] = rectlat_bot[b]
            final_panel[x] = panel_id[b]
            final_panel[y] = panel_id[b]
            
            x=x+2
            y=y+2
    
    return final_panel, rectlon, rectlat

# Main function - GEOJSON version
# Takes points for trellis posts and returns panel geometries

def posts_to_panels_geojson(row_num, panel_num, panel_width, lat_col, lon_col, csv):
    '''
    Takes a .csv file with the lat/lon coordinates of vineyard trellis posts
    collected with Swift Duro RTK unit
    and returns rectangular panel geometries bewteen points.

    ** The file must have columns labeled "Row" and "Post" to identify the points **
    
    
    Inputs:
    row_num (int): number of rows
    panel_num(int): number of panels per row
    panel_width(int): width of each panel in meters
    lat_col(str): column header of the column containing latitude coordinates
    lon_col(str): column header of the column containing longitude coordinates
    csv(str): file path to csv with lat/long coordinates of trellis posts
    
    Output:
    a .geojson file called "posts_to_panels.geojson" saved to the current working directory
    '''
    
    posts = pd.read_csv(csv)
    posts['Panel_ID']= "" # creates new column for unique panel IDs

    for i in np.arange(0,len(posts)):
        panel_id_x = posts['Row'][i]
        panel_id_y = posts['Post'][i]
        posts.at[i, 'Panel_ID']= (panel_id_x, panel_id_y)
        
    c=posts.columns
    row=posts.Row

    post=posts.Post
    panel_id = posts.Panel_ID

    # lat = posts['Lat[deg]']
    # lon = posts['Lon[deg]']

    lat = posts[lat_col]
    lon = posts[lon_col]
    
    utme = [0]*len(row)
    utmn = [0]*len(row)
    lat_check = [0]*len(row)
    lon_check = [0]*len(row)
    zone = [0]*len(row)
    T = ['']*len(row) 

    utme_conv,utmn_conv,lat_check_conv,lon_check_conv = convertUTM(utme,utmn,lat_check,lon_check,zone,T,lat,lon)
    
    top_lefte = [0]*len(row)
    top_leftn = [0]*len(row)
    bot_lefte = [0]*len(row)
    bot_leftn = [0]*len(row)
    
    tle_coord, ble_coord, tln_coord, bln_coord = rectangle_coordinates(row, utme_conv, utmn_conv, top_lefte, bot_lefte,top_leftn, bot_leftn, panel_width, panel_num)
    
    df_rect = pd.DataFrame({'Post':post,
                        'Panel_ID':panel_id,
                        'Post East':utme_conv, 
                        'Post North':utmn_conv, 
                        'Top East':tle_coord, 
                        'Top North':tln_coord, 
                        'Bottom East':ble_coord, 
                        'Bottom North':bln_coord})
    
    rectlon_bot, rectlon_top, rectlat_bot, rectlat_top = toLatLon(utmn_conv, 
                                                                  tle_coord, 
                                                                  tln_coord, 
                                                                  ble_coord, 
                                                                  bln_coord, 
                                                                  T,
                                                                  zone)
    
    rectlon = [0]*(((len(utmn_conv)*4)))
    rectlat = [0]*(((len(utme_conv))*4))
    final_panel = ['']*(((len(utme_conv))*4))
    
    final_panel, rectlon, rectlat = fourcourners(rectlon, 
                                                 rectlon_top, 
                                                 rectlon_bot,
                                                 rectlat,
                                                 rectlat_top, 
                                                 rectlat_bot,
                                                 final_panel,
                                                 row, 
                                                 panel_id)
    
    df = pd.DataFrame({'Panel': final_panel, 
                   'Rectangle_Longitude': rectlon, 
                   'Rectangle_Latitude':rectlat})
    
    # Update projection
    crs=pyproj.CRS.from_user_input(4326)
    gdf = gpd.GeoDataFrame(df, geometry= gpd.points_from_xy(df.Rectangle_Longitude, df.Rectangle_Latitude, crs=crs))
    gdf = gdf[['Panel','geometry']]
    p = gdf.dissolve(by='Panel')
    poly=p.convex_hull
    df_poly = pd.DataFrame(poly)
    wow = poly.geometry.to_wkt()
    df = pd.DataFrame([wow]).T
    
    df.reset_index(inplace=True)
    
    idx=len(df)-1
    
    df=df.drop(index=[idx], axis=0)
    
    for i in np.arange(0,idx):
        x,y = df['Panel'][i]
        if y==1:
            df = df.drop(index=[i], axis=0)
    

    df.iloc[:, -1] = gpd.GeoSeries.from_wkt(df.iloc[:, -1])
    panels_gdf = gpd.GeoDataFrame(df, geometry=df.iloc[:, -1])
    panels_gdf = panels_gdf.rename(columns={"0":"geometry"})
    panels_gdf.drop(panels_gdf.columns[1], axis=1, inplace=True)
    panels_gdf["Panel"]=panels_gdf["Panel"].astype(str) 
    # print(panels_gdf)
    panels_gdf.to_file('posts_to_panels.geojson', driver="GeoJSON")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take a .csv file with the lat/lon coordinates of vineyard trellis posts and return rectangular panel geometries bewteen points.')
    parser.add_argument('row_num', type=int, help='number of rows')
    parser.add_argument('panel_num', type=int, help='number of panels per row')
    parser.add_argument('panel_width', type=float, help='panel width in meters')
    parser.add_argument('lat_col', type=str, help='column header of the column containing latitude coordinates')
    parser.add_argument('lon_col', type=str, help='column header of the column containing longitude coordinates')
    parser.add_argument('csv', type=str, help='path to .csv file with the post coordinates')

    args = parser.parse_args()

    posts_to_panels_geojson(args.row_num, args.panel_num, args.panel_width, args.lat_col, args.lon_col, args.csv)