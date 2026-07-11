# Stress drop estimation using phase coherence analysis.

This repository contains a Jupyter notebook-Python-based data processing pipeline to estimate earthquake stress drop using the **inter-station phase coherence method** based on the methodology outlined by **Hawthonre (2019)** [(**GJI**)](https://doi.org/10.1093/gji/ggy429).

Stress drop represents the change in average shear stress on a fault plane during an earthquake rupture. This workflow isolates the earthquake source properties from path effects by implementing an Empirical Green's Function (EGF) approach using a collocated target and reference earthquake pair. 

This repository contains four main function:
1. signal_preprocessing.py
   
   This function is a standard data-cleaning pipeline for earthquake data (seismograms) using the ObsPy library. It includes demean, tapering,       filtering, remove instrument response, and merge and fill the gaps.
3. taup_predicted_arivals1.py
4. cross_correlation_function.py
   This Python function, calc_xc, calculates the Normalized Cross-Correlation between two seismic waveforms (tr1 and tr2) around specific            arrival time marks (pick1 and pick2).
5. phscoh.py

**Pipeline workflow**

The core analysis is driven sequentially inside the Jupyter Notebook:

## 0. Import the libraries
This notebook needs some core libraries such as:
1. ObsPy
2. numpy
3. matplotlib
4. pygmt
5. pandas

## 1. Data preparation and pre-processing: target earthquake
This step is to process the target earthquake: the bigger one, the one we estimate the rupture diameter.

## 2. Data preparation and pre-processing: reference earthquake
This step is to process the reference earthquake: the smaller earthquake that we use to remove the Green's function.

## 3. Plot pair of earthquake and the stations
The spatial visualisation of the target earthquake, reference earthquake, and the seismic network used in the analysis.

## 4. Onset pick time correction
This section is to refine the pick time of onset detection between target and reference seismogram by implementing cross-correlation pic correction technique.

## 5. Final signal selection
In this section, the final seismogram for phase-coherence analysis are selected by imposing SNR and cross-correlation factor criteria.

## 6. Phase Coherence
In this section, interstation phase coherence is calculated.

## 7. Stress drop estimation

