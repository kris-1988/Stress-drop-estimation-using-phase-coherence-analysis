readme_fileformat.txt

File format for all synthetic data, same format. 

One file per earthquake. All earthquakes are co-located. 
Station information is given in the station metadata file.

File name is: Author_event_n_M_n
where n is the event number and Author is the submitting author for that set of synthetics. 
 
Line 1 is a header 
%#fpoints, #stations, eventnumber, log_M0, depth_km, spectral_type(1=disp; 2=vel; 3=acc), log10_spectra(1=log10; 2=linear), wave_type(1=P, 2=S)
	With 
	- #fpoints= Number of frequency points
	- #stations= number of stations
	- event number = unique event number
	- log_M0= event log10 seismic moment in Nm
	- depth_km= event depth in km 
	- spectral_type = 1 for displacement; 2 for velocity; 3 for acceleration
	- log10_spectra = 1 if the spectra are in log10 and 2 if the spectra are in real (linear)
	- wave_type = 1 if P-wave spectra, 2 if S-wave spectra
Line 2 is a header describing Lines 3 to end (freq: frequency column; site_01, site_02, etc...)
Lines 3 to end have columns: 
	Col 1: frequency points
	Col 2 - N+1: Fourier spectra at site 1 to N=1 for each frequency point