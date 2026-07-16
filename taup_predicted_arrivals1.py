from obspy import UTCDateTime
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
import numpy as np

def predict_arrivals (eq_loc, invh):
    
    """ Function to predict theoretical travel time using TauP Obspy, and automatic phase detections

    Perameter
    _________

    eq_loc  = earthquake location (latitude, longitude, depth)
    invh    = inventory of stations

    Return
    ______

    arrivaltime = a dictionary first p-phase or P-phase at each station
    
    """

    arrivaltime=dict()
    
    #1D Velocity model for TauP ray-tracing: 1066a, 1066b, ak135, ak135f, iasp91, prem, etc. 
    model = TauPyModel(model="ak135")

    for nw in invh:
        for stn in nw:
            # grab out the longitude and latitude
            lon=stn.longitude
            lat=stn.latitude
        
            #stations and event distance in degrees
            # check that the units here are really km
            # lon and lat got flipped !!!!
            sta_eq_dist = locations2degrees(lat,lon , eq_loc [1], eq_loc [0])  #lat1, long1, lat2, long2
            
            #theoretical travel time for this station
            arrivals = model.get_travel_times(source_depth_in_km= eq_loc [2],
                                              distance_in_degree= sta_eq_dist,
                                              phase_list=["p", 'P', 'Pg','Pn'])

            # get all the arrival times for this station
            atime = [arv.time for arv in arrivals]
            
            # note the minimum arrival time
            tmin=np.min(atime)
            
            # save to a dictionary
            idi='.'.join([nw.code,stn.code])
            arrivaltime[idi]=tmin


    return arrivaltime