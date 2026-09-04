%% Progetto IMAD 2026 - MORELLO MATTEO
%% Creazione della funzione per la valutazione del modello proposto

function yhat = predict_load(X)
% PREDICT_LOAD Prevede il carico elettrico usando il Master Model.
% INPUT: X, tabella MATLAB contenente TIMESTAMP e le stazioni w1...w25
% OUTPUT: yhat, vettore colonna delle previsioni

    % 1. CARICAMENTO PERSISTENTE DEI PARAMETRI
    % Evita di ricaricare il file dal disco a ogni chiamata
    persistent master_model knot_temp k_opt
    if isempty(master_model)
        % Carica il modello salvato
        dati = load('modello_master.mat');
        master_model = dati.master_model;
        knot_temp = dati.knot_temp;
        k_opt = dati.k_opt;
    end

    % 2. PRE-PROCESSING DEI DATI
    % Estrae tutte le colonne che iniziano con 'w' e ne fa la media
    colonne_w = startsWith(X.Properties.VariableNames, 'w');
    W_matrix = X{:, colonne_w};
    x_test = mean(W_matrix, 2);
    
    % Estrae l'ora dal TIMESTAMP (gestisce sia formati datetime che numerici)
    if isdatetime(X.TIMESTAMP)
        hours_test = hour(X.TIMESTAMP) + minute(X.TIMESTAMP)/60;
    else
        hours_test = X.TIMESTAMP; 
    end

    % 3. COSTRUZIONE MATRICE DI TEST (Replica esatta del Punto 7)
    
    % A. Componenti Spline Non Lineare (Termodinamica)
    T_f_test = max(0, knot_temp - x_test); % Lineare Inverno
    T_c_test = max(0, x_test - knot_temp); % Lineare Estate
    T_f2_test = T_f_test.^2;               % Quadratica Inverno (Arco)
    T_c2_test = T_c_test.^2;               % Quadratica Estate (Arco)
    X_temp_test = [T_f_test, T_c_test, T_f2_test, T_c2_test];
    
    % B. Componenti Armoniche (Routine Umana)
    X_time_test = [];
    for j = 1:k_opt
        X_time_test = [X_time_test, sin(2*pi*j*hours_test/24), cos(2*pi*j*hours_test/24)];
    end
    
    % C. Interazioni (Rami termici x TUTTE le armoniche)
    X_int_test = [];
    for c = 1:size(X_time_test, 2)
        X_int_test = [X_int_test, T_f_test .* X_time_test(:, c), T_c_test .* X_time_test(:, c)];
    end
    
    % Matrice Master Finale
    X_test_master = [X_temp_test, X_time_test, X_int_test];

    % 4. GENERAZIONE PREVISIONE
    yhat = predict(master_model, X_test_master);
    
end