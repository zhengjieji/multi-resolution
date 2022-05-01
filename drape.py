from auvlib.data_tools import std_data, xtf_data, all_data, csv_data
from auvlib.bathy_maps import draw_map, mesh_map, map_draper

import os, math, random
import numpy as np
import matplotlib.pyplot as plt


def drape(cereal_path, all_path, xtf_path):
    std_mbes_pings = std_data.mbes_ping.read_data(cereal_path)
    print(len(std_mbes_pings))

    n = 4 # folders = [2,3,4,5,6,12,13,14,15]

    xtf_pings = []
    all_pings = []
    all_entries = []
	
    all_folder_path = all_path + str(n) + "_all"
    xtf_folder_path = xtf_path + str(n) + "_xtf"

    xtf_pings = xtf_data.xtf_sss_ping.parse_folder(xtf_folder_path)
    all_pings = all_data.all_mbes_ping.parse_folder(all_folder_path)
    all_entries = all_data.all_nav_entry.parse_folder(all_folder_path)

    sound_speeds = all_data.convert_sound_speeds(all_pings)

    # bathy map
    d = draw_map.BathyMapImage(std_mbes_pings, 500, 500)
    d.draw_height_map(std_mbes_pings)
    d.draw_track(std_mbes_pings)
    im = d.make_image()
    plt.imshow(im)
    # plt.show()

    V, F, bounds = mesh_map.mesh_from_pings(std_mbes_pings, 1) # change for multi resolution
    # mesh_map.show_mesh(V, F)

    Vb, Fb, Cb = map_draper.get_vehicle_mesh()
    draper = map_draper.MapDraper(V, F, xtf_pings, bounds, sound_speeds)
    draper.set_vehicle_mesh(Vb, Fb, Cb)
    draper.set_ray_tracing_enabled(False) # no need to account for refraction as we are close to ground
    # draper.set_tracing_map_size(300.)
    draper.set_close_when_done(True) # do not close the viewer when done draping
    draper.set_intensity_multiplier(2.) # allows modifying the intesities of sidescan on mesh
    draper.show() # show the viewer and drape
    map_images = draper.get_images() # the resulting map image data structure
    print("Got map images: ", len(map_images))
    if len(map_images) >= 0:
        map_draper.write_data(map_images, str(n) + ".cereal") # save for later

    
if __name__ == "__main__":
    drape("area1.cereal", "./Area1/", "./Area1/")
