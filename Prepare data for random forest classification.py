#!/usr/bin/env python
# coding: utf-8

# # Prepare data for Random Forest Classification
# 
# Chris Harding  (charding@iastate.edu), Aug. 6 2019
# 
# 
# - This script will prepare GIS data for a random forest classification. 
# - It needs a polygon layer of the quadrat (or other boundaries) and one or more rasters covered by the polygon layer,
# - The rasters can be multi-band images.
# - In a ArcGIS context, each polygon acts as a zone that covers a set of raster cells (pixels)
# - For each zone (polygon) statistics, such as mean, std.dev., median, etc. can be calculated from the set of cells it covers. Zonal statistis are calculated for each band of the raster
# - The result is stored in a zonal statistics table, in which each row corresponds to a zone, with a column for each type of statistic, repeated by band.
# - Each zone has the same object id as the polygon it corresponds to.
# - Using this id as key the zonal statistics tables will be joined to the polygon layer, adding the zonal statistics (per band) to the polygon's attribute table.
# - The random forst classification is based on the polgon layer's attributes, including the addeed zonal statistics. The rasters are not necessary for the classification.
# 
# 
# 
# ### Notes on the code
# - I assume you have installed the ESRI arcpy package so it can be imported.
# - The input polygon layer can be inside a geoDB or in a shapefile
# - Raster data can be either files (geotiffs, etc.) with a folder a workspace or inside a geoDB as workspace.
# - For this example, all polygons and rasters are kept inside a geoDB, which is set as workspace. As examples, I have provided a single geotiff and a single shapefile. To use these set the workspace to the current folder.
# - To keep the dataset small, I have cropped the original geotiffs to just cover the polygons (quadrant outlines), this is shown by a leading `cr_`, e.g. `cr_T20160831_161520_0e20_3B_AnalyticMS_SR` is the cropped version of the image from Aug.31, 2016 (`20160831`).
# 
# - When a arcpy geoproessing function is used, it is wrapped in try-except so that, on an error, a hopefully diagnostically helpful error message can be displayed (the exeption object as text).
# - On success, the start/end time and any result messages (from `arcpy.GetMessages()`) are displayed. 
# 

# In[1]:


import arcpy, os

# If True, will overwrite existing layers
arcpy.env.overwriteOutput = True


# In[2]:


# We need to use Spatial Analyst tools, check out a license for it
if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
        print("Got license for Spatial Analyst")
else:
    print("Error checking out Spatial Analyst license")


# In[3]:


# make sure all your data (rasters, shapefiles, geodb) are in the
# same folder as this notbook!
cwd = os.getcwd()
print("Current folder is", cwd)


# In[4]:


# make a file geoDB for temp files in the current folder
tmp_gdb = "tmp.gdb"

# try to delete it
try:
    arcpy.Delete_management(tmp_gdb)
except:
    pass # tmp exists already, that's OK ...

# make new tmp geoDB, or re-use if it already exists
try:
    arcpy.CreateFileGDB_management(cwd,tmp_gdb )
except Exception as e:
    print(e)
    print("Delete", tmp_gdb, "manually and re-run this cell")
else:
    arcpy.env.scratchWorkspace = cwd + "\\" + tmp_gdb
    print("temp workspace is", arcpy.env.scratchWorkspace)


# In[5]:


# Set a geoDB (in current folder) as workspace
arcpy.env.workspace = "SDS_detection_ArcGISPro_project.gdb"

# For shapefiles and raster files, set the workspace environment to current folder
#arcpy.env.workspace = cwd
print("workspace is", arcpy.env.workspace)


# ### Polygon layer selection

# In[6]:


# list all polygon layers in the workspace
print("workspace contains these polygon layers:")
walk = arcpy.da.Walk(arcpy.env.workspace, datatype="FeatureClass", 
                     type="Polygon")

for dirpath, dirnames, filenames in walk:
    if filenames != []:
        print(os.path.basename(dirpath))
        for f in filenames:
            print("\t",f)
        print()


# In[7]:


# Select the polygon layer inside the workspace to be processed
poly = r"Soybean_Quadrats_2018"
#poly = r"Soybean_Quadrats_2017"
#poly = "Soybean_Quadrats_2016.shp"

# List all fields (attributes) it contains
try:
    poly_desc = arcpy.Describe(poly)
except Exception as e:
    print("Error:", e)
else:
    print("Polgon layer", poly_desc.name, "has these fields:")
    
    # loaded OK, list all fields
    fields = arcpy.ListFields(poly)
    for field in fields:
        print("\t", field.name, " type:", field.type)


# ### Raster layer selection

# In[8]:


# list all rasters in the workspace
print("workspace contains these rasters:")
walk = arcpy.da.Walk(arcpy.env.workspace,datatype="RasterDataset")
for dirpath, dirnames, filenames in walk:
    if filenames != []:
        print(os.path.basename(dirpath))
        for f in filenames:
            print("\t",f)
        print()


# In[9]:


# define the raster and see if it loads


#r = "cr_T20160720_161306_0e0e_3B_AnalyticMS_SR"
#r = "cr_T20160821_161512_0e26_3B_AnalyticMS_SR"
#r = "cr_T20160831_161520_0e20_3B_AnalyticMS_SR"
#poly = r"Soybean_Quadrats_2016"

#r = "cr_T20170720_161916_101e_3B_AnalyticMS_SR"
#poly = r"Soybean_Quadrats_2017"

#r = "cr_T20180716_163325_101b_3B_AnalyticMS_SR" 
#r = "cr_T20180822_161812_0f46_3B_AnalyticMS_SR"
#r = "cr_T20180724_163356_1004_3B_AnalyticMS_SR"
#r = "cr_T20180829_163535_1021_3B_AnalyticMS_SR"
r = "cr_T20180831_163515_1006_3B_AnalyticMS_SR"
poly = r"Soybean_Quadrats_2018"


print("Using", poly, "as polgons for zonal stats")


ras = arcpy.Raster(r)

try:
    ras_desc = arcpy.Describe(ras)
except Exception as e:
    print(e)
else:
    # loaded OK, show number of bands
    print("Using raster", ras_desc.name, " with has", ras_desc.BandCount, "bands")


# ### Create table(s) with zonal statistics
# - for each band in the selected raster, calculates (zonal) statistics of all raster cells covered by a polygon (zone).
# - see the [ZonalStatisticsAsTable](https://pro.arcgis.com/en/pro-app/tool-reference/spatial-analyst/zonal-statistics-as-table.htm) tool for details (incl. valid values for stats_type)
# - `poly` defines the geometry of the zones (i.e., the polygon layer selected earlier)
# - `key` must be the field that can be used as key to join the table to poly later (see the list of fields in poly shown earlier)
# - `stats_type` defines which stats will be calculated, e.g. `MEAN_STD` (see tool documentation for what other types are valid)
# 

# In[10]:


# field that will be used as key for the final join
key = "Quadrat"

# type of stats to be calculated (must match statistics_type in tool!)
#stats_type = "MEAN_STD"
#stats_type = "MEAN"
stats_type = "ALL"


# In[11]:


# replace . in the raster name as that seems to give problems
safe_raster_name = ras_desc.name.replace(".","_")
print("raster name as used for zstat tables:", safe_raster_name)


# In[12]:


# Make a list of the names of all raster bands and their tables
zonal_stats_band_raster = []
zonal_stats_band_table = []
for b in range(ras_desc.BandCount):
    zonal_stats_band_raster.append(ras_desc.name+r"\Band_"+str(b+1))
    zonal_stats_band_table.append("zstats_band_" + str(b+1) + "_" +
                                  safe_raster_name)
#print(zonal_stats_band_raster)
#print(zonal_stats_band_table)


# In[13]:


# zip both lists together so we have a list of tuples, each tuple with a band and a table
zlist = list(zip(zonal_stats_band_raster, zonal_stats_band_table))
#print(zlist)


# In[14]:


# use the ZonalStatisticsAsTable tool to perform the Zonal statistics for each band 

for band,table in zlist: # get the band (raster) and the table that stores the results
    print()
    print(band)
    print(table)
    
    # put table into tmp.gdb
    table = arcpy.env.scratchWorkspace + "\\" + table

    try:
        arcpy.sa.ZonalStatisticsAsTable(poly, 
                                        key,
                                        band, 
                                        table,
                                        "DATA", # ignore NoData cells
                                        stats_type)
    except Exception as e:
        print(e)
    else:
        print(arcpy.GetMessages())


# ### Remove un-needed fields from bands 2,3, etc.
# - each table (for each band) currently has the fields key, AREA and COUNT, all have exactly the same values
# - we'll later join all the tables into a single table, we can remove them from all bands except band 1

# In[15]:


for table in zonal_stats_band_table[1:]:
    print("\nremoving fields from", table)
    table = arcpy.env.scratchWorkspace + "\\" + table
    
    try:
        arcpy.DeleteField_management(table, 
                             [key, "AREA", "COUNT"])
    except Exception as e:
        print(e)
    else:
        print(arcpy.GetMessages())


# ### Rename stats fields
# - each stats field will get the band appended (e.g. `MEAN_1`)
# - as there's no rename, I make a new field (same type), copy it and delete the old field

# In[16]:


for band, table in enumerate(zonal_stats_band_table):
    print("\nrenaming fields in", table)
    table = arcpy.env.scratchWorkspace + "\\" + table
    band += 1
    
    fields = arcpy.ListFields(table)
    for f in fields:
        
          
        # skip these fields
        if f.name in ["OBJECTID", key, "COUNT", "AREA"]:
            continue
        
        new_name = f.name + "_" + str(band)
        #print(" ",f.name, f.type, new_name)
          
        try:
            arcpy.AddField_management(table, 
                                      new_name,
                                      f.type)
        except Exception as e:
            print(e)
        else:
            print(arcpy.GetMessages())
            
            
            
        # copy values into new field new_name = f.name
        # this uses python, so the f.name must be wrapped into !
        try:    
            arcpy.CalculateField_management(
                                table, 
                                new_name,
                                "!" + f.name  + "!")
        except Exception as e:
            print(e)
        else:
            print(arcpy.GetMessages())
        
  
        # delete old field
        try:
            arcpy.DeleteField_management(table, f.name)
        except Exception as e:
            print(e)
        else:
            print(arcpy.GetMessages())
    
        print()


# ### Copy band 1 zstat table
# - the band 1 table will be the master zstat table to which the other bands will be joined

# In[17]:


zstat_band1_table = zonal_stats_band_table[0]
zstat_master = "zstat_master_" + safe_raster_name 
print("Copying", zstat_band1_table, "to", zstat_master)

zstat_band1_table = arcpy.env.scratchWorkspace + "//" + zstat_band1_table
zstat_master = arcpy.env.scratchWorkspace + "//" + zstat_master

try:
    arcpy.Copy_management (zstat_band1_table, zstat_master)
except Exception as e:
    print(e)
else:
    print(arcpy.GetMessages())


# ### Join band 2, 3, etc. zstats to master zstat table 
# - key (join_field) is `OBJECTID`
# - all fields are added to the master on the disk (unlike with AddField)

# In[18]:


for table in zonal_stats_band_table[1:]:
    print("\njoining renamed fields from", table, "to master zstat table")
    table = arcpy.env.scratchWorkspace + "\\" + table
    
    try:
        arcpy.JoinField_management(zstat_master, 
                                 "OBJECTID", 
                                 table, 
                                 "OBJECTID")
    except Exception as e:
        print(e)
    else:
        print(arcpy.GetMessages())


# ### Make a copy of the polygon layer
# - make a copy of the polygon layer before we join the zstats to it, incase the original is still needed
# - be aware that this can fail if the copy already exists, despite overwriteOutput = True
# - if this happens, delete the copy manually

# In[19]:


zstat_band1_table = zonal_stats_band_table[0]
poly_zstats = poly + "_zstats_" + safe_raster_name
print("Copying", poly, "to", poly_zstats)
try:
    arcpy.Copy_management (poly, poly_zstats)
except Exception as e:
    print(e)
else:
    print(arcpy.GetMessages())


# ### Join the master zstats table to the copied polygon layer
# - `key` is used as key field for joining
# - result is a polygon layer with new fields, one for each type of zonal stats
# - each zonal stats field has the band it was calculated from appended, e.g.  MEAN_1,  MEAN_3, etc.

# In[20]:


print("joining master zstats to polygon layer", poly_zstats)

try:
    arcpy.JoinField_management(poly_zstats, 
                             key, 
                             zstat_master, 
                             key)
except Exception as e:
    print(e)
else:
    # Show name and type of all fields
    print("Joined layer has these fields:")
    for field in arcpy.ListFields(poly_zstats):
        print("\t", field.name, " type:", field.type)
    print(arcpy.GetMessages())


# In[21]:


# cleanup: delete the temp geoDB
# try to delete it
#try:
#    arcpy.Delete_management(tmp_gdb)
#except Exception as e:
#    print(e)
#else:
#    print("deleted temp workspace")


# In[22]:


print("Done")


# In[ ]:





# In[ ]:





# In[ ]:




