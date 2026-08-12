%% read synthetic data
% % % % % % % 
% written by A. Baltay July 20, 2026 %
% % % % % % %

%% set path, metadata, author
clear all
%close all 

%set your path 
metapath='/Users/abaltay/Desktop/Synthetics July 2026 for distro/';
cd(metapath)

event_info=readtable('catalog_info.txt', 'NumHeaderLines',3); 
sta_info=readtable('site_info.txt', 'NumHeaderLines',3); 

epi_dist=sta_info.Var2; 

% chose which set to analyze: 
author='Bindi';  authordir=fullfile(metapath, 'Bindi_synthetic'); 
%author='Baltay'; authordir=fullfile(metapath, 'Baltay_synthetic'); 
%author='Ji'; authordir=fullfile(metapath, 'Ji_synthetic');
%author='Shearer'; authordir=fullfile(metapath, 'Shearer_synthetic');

cd(authordir)

%% fix catalog magnitude info: 
% Read in catalog M, dist info
lmo=event_info.Var2;

mags=Mo2Mw(10.^lmo);
N=length(mags); 

event_info.Var3=mags; 

T=table(event_info); 

%% 
files=dir([author, '*event*M*.csv']);
Nevents=length(files);


for i=1:Nevents
    % Read header
    fid = fopen(files(i).name,'r');
    header = fgetl(fid);
    fclose(fid);

    hdr = sscanf(header,'%f,').';

    % Read numeric data
    A = readmatrix(files(i).name,'NumHeaderLines',1);

    nfreq(i)=hdr(1); Nstas(i)=hdr(2); EVID(i)=hdr(3); 
    lmo(i)=hdr(4); Mw(i)=(lmo(i)-9.05)/1.5; 
    depth(i)=hdr(5); 
    spectral_type(i)=hdr(6); %spectral_type = 1 for displacement; 2 for velocity; 3 for acceleration
    logspectra(i)=hdr(7);    %log10_spectra = 1 if the spectra are in log10 and 2 if the spectra are in real (linear)
	wave(i)=hdr(8);          %wave_type = 1 if P-wave spectra, 2 if S-wave spectra

    freq{i} = A(:,1);

    % check if log or linear
    if logspectra(i)==2;    FAS{i} = A(:,2:end);
    elseif logspectra(i)==1; FAS{i}=10.^A(:,2:end); end
end

%FAS has size the number of events Nevents and each FAS{i} has size=freq_length x Nstas

%% plot example

% for plotting Shearer in log-space, make the 0-frequency .1 
if strcmp(author,'Shearer')
    for i=1:Nevents
    freq{i}(1)=0.1; 
    end
end

plotnums=[1,50]; % choose event numbers to plot

figure
for j=1:length(plotnums)
    nexttile
    i=plotnums(j); 
    loglog(freq{i}, FAS{i})
    legend(num2str(epi_dist))

    if spectral_type(i)==1; tags='displacement'; elseif spectral_type(i)==2; tags='velocity'; elseif spectral_type(i)==3; tags='acceleration'; end
    if wave(i)==1; tag_wave='P-wave'; elseif wave(i)==2; tag_wave='S-wave'; end

    xlabel('freq [hz]'); ylabel([tag_wave, ' ', tags, ' ','FAS amplitude'])
    title([author, ' Magnitude ', num2str(Mw(i))])
end

