%% NoGo Task EEG: Análisis en el dominio de la Frecuencia. Tiempo-Frecuencia (TFRs)

% En esta práctica realizaremos en análisis de la señal limpia y
% preprocesada en el dominio de la frecuencia. Veremos dos formas de hacer 
% los análisis de tiempo-frecuencia: ventana fija y ventana variable.
% Visualizaremos los datos tanto a nivel de un electrodo como en la
% topografía de una ventana de tiempo y frecuencia específica.

% Realizaremos la corrección de la línea base usando el cambio relativo.

% Y, posteriormente, calcularemos el Gran Promedio (grand average) utilizando
% los datos de diversos sujetos y calcularemos la estadística.

%                          Master U. en Neurociencia Cognitiva y Neuropsicología
%                          
%% Preparación general
% clear workspace
clearvars
close all
clc

% definimos la localización de las carpetas
proyect_dir = 'C:\Users\navar\Desktop\master urjc\ELECTROMAGNÉTICAS\practica 3\TFR (1)\TFR'
code_dir = [proyect_dir '/scripts/'];
raw_dir  = [proyect_dir '/rawdata/'];
save_dir = [proyect_dir '/preprocesamiento/nogo/'];

% definimos el número de los participantes
subj_num = {1 2 3};

cd(code_dir);

% añadir fieldtrip al path
addpath('C:\Users\navar\Desktop\master urjc\ELECTROMAGNÉTICAS\fieldtrip-20240110\fieldtrip-20240110'); % ESTUDIANTES: ajustar el path de cada ordenador

ft_defaults

%% 0 - Preparamos el layout
% Como en esta práctica vamos a visualizar los datos en el espacio de
% sensores. Necesitamos cargar la disposición (layout) de cada sensor en el
% casco.

% cargamos
load([code_dir, 'acticap64_custom.mat']); % usamos como base el de 64 para construir el layout
load([code_dir, 'EEG32-acticap_custom-label.mat']); % pero solo usaremos 32

%% 1 - localizamos los datos

isubject = 1; % ESTUDIANTES: elegimos el sujeto que queremos inspeccionar

% busca la carpeta donde están los datos
input_filename  = 'datos_nogo_trials_limpios'; 
filename        = ['sub-',sprintf('%02.0f',subj_num{isubject}), '/sub-',sprintf('%02.0f',subj_num{isubject}),'_', input_filename];

load([save_dir, filename])

%% %%%%%%%%%%%%%%%%%% Repaso del Preprocesamiento %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2 - Aplicamos preprocesamiento óptimo para TFRs

% 2A - Redefinimos los trials (necesitamos trials tan largos para tener suficiente resolución espectral y evitar el efecto borde)
cfg = [];
cfg.trials    = 'all';
cfg.channel   = 'all';
cfg.toilim    = [-1 1.5-(1/datos_limpios.fsample)]; % en segundos
datos_redef   = ft_redefinetrial(cfg,datos_limpios);

% 2B - Filtrado de la señal (paso bajo).
cfg = [];
cfg.trials    = 'all';
cfg.channel   = 'all';
cfg.lpfilter  = 'yes';  
cfg.lpfreq    = 40; %Hz


% 2D - Corrección de la tendencia lineal
cfg.detrend         = 'yes';

% 2E - Ejecutar todo (llamamos a la función)
datos_preproc  = ft_preprocessing (cfg,datos_redef);

%% 3 - Preparamos los datos de comportamiento (beh) para poder separar los ensayos

tmp_tags = cell2mat(datos_preproc.trialinfo);

tags = [];
tags.TrialType     = [tmp_tags.TrialType];

% 3A -Seleccionamos los ensayos de las condiciones que nos interesan

conditions = {'go' 'nogo'};

selections = {...
    'ismember(tags.TrialType,{''trial-go''})'
    'ismember(tags.TrialType,{''trial-nogo''})'
    };

trials_go   = eval(selections{1});
trials_nogo = eval(selections{2});

%% %%%%%%%%%%%%%%%%%% Análisis de Tiempo-Frecuencia %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 4 - Análisis de Tiempo-Frecuencia: ventana fija
%hacemos las ventanas para analizar el tiempo frecuencia con ventana fija
%(0.5 segundos) vamos seleccionando que ocurre en cada una de estas
%ventanas, tanto para la condición go como la no go 

cfg              = [];
cfg.method       = 'mtmconvol'; 
cfg.output       = 'pow';	
cfg.foi          = 2:2:40;                        % como usamos una ventana fija (t_ftimwin) nuestra resolución espectral es 1/05 = 2Hz	     
cfg.taper        = 'hanning';                     % tipo de ventana
cfg.toi          = -1:0.02:1.5;                   % deslizaremos la ventana en pasos de 20ms desde -1 a 1.5s
cfg.t_ftimwin    = ones(1,length(cfg.foi)) * 0.5; % ventana fija de 0.5s

% condición Go
cfg.trials       = find(trials_go);
TFR_go           = ft_freqanalysis(cfg, datos_preproc);

% condición No-Go
cfg.trials       = find(trials_nogo);
TFR_nogo         = ft_freqanalysis(cfg, datos_preproc);


TFR_go.freq == 10 % ESTUDIANTES: indicamos frecuencia en Hz y miramos el índice (1)
cfg.t_ftimwin

%% 5 - Representamos los resultados el TFR de un solo electrodo
%represntamos, en un solo canal, la actviidad de ese sujeto para la
%condicion go y para la condición no go

cfg          = [];
cfg.colorbar = 'yes';
cfg.zlim     = 'maxmin'; % [-1 1]
cfg.layout   = lay;
cfg.channel  = 'CZ'; %el canal que quiero que me represente
cfg.colormap = 'jet';
cfg.baselinetype = 'relchange';  % tipo de corrección: relative, absolute, relchange
cfg.baseline     = [-.5 -.1];    % ventana para la corrección de línea base

figure(1);
ft_singleplotTFR(cfg, TFR_go) 

xlabel ('Tiempo (s)');
ylabel ('Frecuencia (Hz)');
xline(0, 'k')
title('Go')

figure(2);
ft_singleplotTFR(cfg, TFR_nogo)
xlabel ('Tiempo (s)');
ylabel ('Frecuencia (Hz)');
xline(0, 'k')
title('No-Go')

% 5A - Observa el efecto borde por usar una ventana Hanning de 0.5: https://www.fieldtriptoolbox.org/faq/why_does_my_tfr_contain_nans/

%% 6 - Multiplot de la condición NoGo/Go

cfg              = [];
cfg.layout       = lay;
cfg.interactive  = 'yes';
cfg.baselinetype = 'relchange';  % tipo de corrección: relative, absolute, relchange (a efectos de visualización)
cfg.baseline     = [-.5 -.1];    % ventana para la corrección de línea base
cfg.colormap     = 'jet';
ft_multiplotTFR(cfg, TFR_nogo);
ft_multiplotTFR(cfg, TFR_go)

%% 7 - Representamos la topografía de un momento temporal y una banda de frecuencia de interés

cfg              = [];
cfg.layout       = lay;
cfg.baselinetype = 'relchange';  
cfg.baseline     = [-.5 -.1];   
cfg.xlim         = [0 .4]; % ESTUDIANTES: indicamos tiempo de interés en segundos
cfg.ylim         = [8 12];        % ESTUDIANTES: indicamos frecuencia en Hz
cfg.colormap     = 'jet';
figure(1)
ft_topoplotTFR(cfg, TFR_nogo)

%% 8 - Corrección de línea base

cfg = [];
cfg.baseline     = [-.5 -.1];
cfg.baselinetype = 'relchange'; % absolute, relative, dB...
cfg.parameter    = 'powspctrm';

TFR_go_blc   = ft_freqbaseline(cfg, TFR_go);
TFR_nogo_blc = ft_freqbaseline(cfg, TFR_nogo);

%% 9 - Diferencia entre condiciones

diff = TFR_nogo_blc;
diff.powspctrm = TFR_nogo_blc.powspctrm - TFR_go_blc.powspctrm;

cfg          = [];
cfg.colorbar = 'yes';
cfg.zlim     = 'maxmin'; % [-1 1]
cfg.layout   = lay;
cfg.channel  = 'CZ'; %electrodo
cfg.colormap = 'jet';
cfg.colorbar = 'yes';

figure(1);
ft_singleplotTFR(cfg, diff)

xlabel ('Tiempo (s)');
ylabel ('Frecuencia (Hz)');
xline(0, 'k')
title('Efecto NoGo - Go')

%% 10 - Análisis de Tiempo-Frecuencia: longitud de la ventana dependiente de la frecuencia 

cfg            = [];
cfg.method       = 'mtmconvol';
cfg.taper        = 'hanning';
cfg.foi          = 2:2:40;
cfg.t_ftimwin    = 7./cfg.foi;  % 7 ciclos por ventana de tiempo: 1000 ms para 7 Hz (1/7 x 7 cycles); 700 ms para 10 Hz (1/10 x 7 cycles) and 350 ms para 20 Hz (1/20 x 7 cycles).
cfg.toi          = -1:0.02:1.5;

% condición Go
cfg.trials       = find(trials_go);
TFR_go           = ft_freqanalysis(cfg, datos_preproc);

% condición No-Go
cfg.trials       = find(trials_nogo);
TFR_nogo         = ft_freqanalysis(cfg, datos_preproc);



TFR_go.freq == 10 % indicar en Hz
cfg.t_ftimwin

%% %%%%%%%%%%%%%%%%%% Gran Promedio de los TFRs %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 11 - Preprocesamiento (automático)

input_filename = 'datos_nogo_trials_limpios';
datos_preproc = {};

for isubject = 1: length(subj_num)

    % carga los datos
    filename = ['sub-',sprintf('%02.0f',subj_num{isubject}),'/' 'sub-',sprintf('%02.0f',subj_num{isubject}),'_',input_filename];
    load([save_dir,filename]);
    
    % Redefinimos los trials (necesitamos trials tan largos para tener suficiente resolución espectral y evitar el efecto borde)
    cfg = [];
    cfg.trials    = 'all';
    cfg.channel   = 'all';
    cfg.toilim    = [-1 1.5-(1/datos_limpios.fsample)]; % en segundos
    datos_redef   = ft_redefinetrial(cfg,datos_limpios);

    % Filtrado de la señal (paso bajo).
    cfg = [];
    cfg.trials    = 'all';
    cfg.channel   = 'all';
    cfg.lpfilter  = 'yes';  
    cfg.lpfreq    = 40; %Hz

    cfg.detrend   = 'yes';


    datos_preproc{isubject}  = ft_preprocessing (cfg,datos_redef);
    
    sprintf(['Sujeto ' num2str(subj_num{isubject}) ' preprocesado correctamente'])
    
end

%% 12 - Calcula el TFR de cada condición para cada sujeto  (automático)

clear TFR_go TFR_nogo tmp_tags tags TFR_go_blc TFR_nogo_blc

for isubject = 1: length(subj_num)
    
    clear trials_go trials_nogo
   
    % busca el tipo de ensayo en el comportamiento (beh)
    tmp_tags = cell2mat(datos_preproc{isubject}.trialinfo);

    tags = [];
    tags.Target        = [tmp_tags.Target];
    tags.ResponseType  = [tmp_tags.ResponseType];
    tags.TrialType     = [tmp_tags.TrialType];
    tags.RT            = [tmp_tags.RT];
    tags.Accuracy      = [tmp_tags.Accuracy];
    
    trials_go   = eval(selections{1});
    trials_nogo = eval(selections{2});
    
    cfg              = [];
    cfg.method       = 'mtmconvol'; 
    cfg.output       = 'pow';	
    cfg.foi          = 2:2:40;                        % como usamos una ventana fija (t_ftimwin) nuestra resolución espectral es 1/05 = 2Hz	     
    cfg.taper        = 'hanning';                     % tipo de ventana
    cfg.toi          = -1:0.02:1.5;                   % deslizaremos la ventana en pasos de 20ms desde -1 a 1.5s
    cfg.t_ftimwin    = ones(1,length(cfg.foi)) * 0.5; % ventana fija de 0.5s

    % condición Go
    cfg.trials       = find(trials_go);
    TFR_go{isubject} = ft_freqanalysis(cfg, datos_preproc{isubject});

    % condición No-Go
    cfg.trials       = find(trials_nogo);
    TFR_nogo{isubject} = ft_freqanalysis(cfg, datos_preproc{isubject});
    
    % corrección de línea base
    cfg = [];
    cfg.baseline     = [-.5 -.1];
    cfg.baselinetype = 'relchange';
    cfg.parameter    = 'powspctrm';

    TFR_go_blc{isubject}   = ft_freqbaseline(cfg, TFR_go{isubject});
    TFR_nogo_blc{isubject} = ft_freqbaseline(cfg, TFR_nogo{isubject});
    
end

%% 13 - Calcula en Gran Promedio (ft_freqgrandaverage) (automático)

cfg   = [];
cfg.parameter = 'powspctrm';
GA_TFR_go     = ft_freqgrandaverage(cfg, TFR_go_blc{:});

cfg     = [];  
cfg.parameter = 'powspctrm';
GA_TFR_nogo   = ft_freqgrandaverage(cfg, TFR_nogo_blc{:});

%% 14 - Diferencia entre condiciones

diff = GA_TFR_go;
diff.powspctrm = GA_TFR_nogo.powspctrm - GA_TFR_go.powspctrm;

cfg          = [];
cfg.colorbar = 'yes';
cfg.zlim     = 'maxmin'; % [-1 1]
cfg.layout   = lay;
cfg.channel  = 'CZ';
cfg.colormap = 'jet';
cfg.colorbar = 'yes';

figure(1);
ft_singleplotTFR(cfg, diff)

xlabel ('Tiempo (s)');
ylabel ('Frecuencia (Hz)');
xline(0, 'k')
title('Efecto NoGo - Go')

%% %%%%%%%%%%%%%%%%%% Estadística %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 15- Estadística

cfg = [];
cfg.frequency   = [4 8];
cfg.latency     = [0.1 0.5];
cfg.channel     = 'all';
cfg.avgovertime = 'yes';
cfg.avgoverfreq = 'yes';
cfg.method      = 'stats';
cfg.method      = 'analytic';
cfg.statistic   = 'ft_statfun_depsamplesT';
cfg.alpha       = 0.05;
cfg.ivar        = 1;
cfg.uvar        = 2;
cfg.tail        = 0;

Nsub = 3;
cfg.design(1,1:2*Nsub)  = [ones(1,Nsub) 2*ones(1,Nsub)];
cfg.design(2,1:2*Nsub)  = [1:Nsub 1:Nsub];
cfg.ivar                = 1; 
cfg.uvar                = 2; 

Fieldtripstats  = ft_freqstatistics(cfg, TFR_nogo_blc{:}, TFR_go_blc{:});

Fieldtripstats.prob'

labels{Fieldtripstats.prob<0.05}
