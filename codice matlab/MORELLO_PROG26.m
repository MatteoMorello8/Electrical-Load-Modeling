close all
clear
clc
warning('off', 'stats:LinearModel:RankDefDesignMat');
%% Progetto IMAD 2026 - MORELLO MATTEO

%% 1. Fare il grafico del carico nel tempo, della temperatura media nel tempo e del carico in
% funzione della temperatura media;

% Carico del dataset
data = readtable(['C:\Users\Utente\Desktop\matlab\IMAD\IMAD B\Progetto25_26\dataset\L1_train.csv']);
% Inizializzazione dei dati per rappresentarli
matrixTemp = data{:, 3:27};
mean_w = mean(matrixTemp, 2);
data.mean_w = mean_w;
figure('Name','Rappresentazione Grafica della Relazione tra dati','NumberTitle','off');
x_temp = linspace(1,length(data.TIMESTAMP), length(data.TIMESTAMP));
% Figura 1 - Andamento del carico nel tempo
subplot(3,1,1);
scatter(data.TIMESTAMP,data.LOAD);
xlabel('Ora del giorno (TIMESTAMP)');
ylabel('Carico Elettrico (LOAD)');
title('Andamento del Carico Elettrico nel Tempo');
grid on;
% Figura 2 - Andamento della temperatura media nel tempo
subplot(3,1,2);
scatter(data.TIMESTAMP,data.mean_w);
xlabel('Ora del giorno (TIMESTAMP)');
ylabel('Temperatura Media (°C)');
title('Andamento della Temperatura Media nel Tempo');
grid on;
% Figura 3 - Carico in funzione della temperatura media
subplot(3,1,3);
scatter(data.mean_w, data.LOAD);
xlabel('Temperatura Media (°F)');
ylabel('Carico Elettrico (LOAD)');
title('Carico in funzione della temperatura media');
grid on;

% ---> ESPORTAZIONE FIGURA 1 (Scatterplot) <---
exportgraphics(gcf, 'immagini/punto1_scatter_relazioni.png', 'Resolution', 300);


figure('Name','Rappresentazione Grafica della Relazione tra dati','NumberTitle','off');
% Figura 1 - Andamento del carico nel tempo
subplot(2,1,1);
plot(x_temp,data.LOAD);
xlabel('TIMESTAMP');
ylabel('Carico Elettrico (LOAD)');
title('Andamento del Carico Elettrico nel Tempo');
grid on;
xlim([0,length(data.TIMESTAMP)]);
% Figura 2 - Andamento della temperatura media nel tempo
subplot(2,1,2);
plot(x_temp,data.mean_w);
xlabel('TIMESTAMP');
ylabel('Temperatura Media (°F)');
title('Andamento della Temperatura Media nei 10 Anni');
grid on;
xlim([0,length(data.TIMESTAMP)]);

% ---> ESPORTAZIONE FIGURA 2 (Line plot 10 anni) <---
exportgraphics(gcf, 'immagini/punto1_andamento_temporale.png', 'Resolution', 300);

%% 2. Calcolare una matrice di correlazione tra le variabili di temperatura
% LOAD
% 1. Estraggo il target (LOAD) e i predittori (le 25 temperature)
% Metto LOAD come prima colonna per comodità visiva
load = data.LOAD;
temperature = data{:, 3:27};
matrix_data = [load, temperature];  
% 2. Calcolo la matrice di correlazione di Pearson
% 'Rows', 'complete' dice a MATLAB di calcolare la correlazione usando 
% SOLO le righe in cui non ci sono valori NaN
R = corr(matrix_data, 'Rows', 'complete');
% 3. Visualizzo la matrice usando una Heatmap
figure('Name','Rappresentazione Grafica della Matrice di Correlazione','NumberTitle','off');
% Rinomino le etichette degli assi per il grafico
namesVar = ['LOAD', data.Properties.VariableNames(3:27)];
% Creo la heatmap
h = heatmap(namesVar, namesVar, R);
h.Title = 'Matrice di Correlazione tra LOAD e Temperature (w1-w25)';
h.Colormap = parula; % Mappa di colori standard di MATLAB

% ---> ESPORTAZIONE HEATMAP <---
exportgraphics(gcf, 'immagini/punto2_matrice_correlazione.pdf', 'ContentType', 'vector');


%% 3. Fittare modelli polinomiali ai minimi quadrati di LOAD in funzione di w(t), confrontando
% diversi gradi tramite cross-validataon, test F, AIC, FPE o MDL;
% 1. Pulizia dei dati (rimuovo le righe con LOAD = NaN)
idx_valid = ~isnan(data.LOAD);
y_valid = data.LOAD(idx_valid);
x_valid = data.mean_w(idx_valid);
t_valid = data.TIMESTAMP(idx_valid);
% 2. Split Casuale (Random Cross-Validation: 80% Train, 20% Test)
N_total = length(y_valid);
rng(42); % Fisso il seed
idx_random = randperm(N_total); 
index_split = floor(0.8 * N_total); 
idx_train = idx_random(1:index_split);
idx_test  = idx_random(index_split+1:end);
x_train = x_valid(idx_train);
y_train = y_valid(idx_train);
t_train = t_valid(idx_train);
N_train = length(y_train); 
x_test = x_valid(idx_test);
y_test = y_valid(idx_test);
t_test = t_valid(idx_test);
% 3. Inizializzazione delle variabili per i risultati
degreesTest = 1:5;
AIC_values = zeros(length(degreesTest), 1);
FPE_values = zeros(length(degreesTest), 1);
RMSE_test  = zeros(length(degreesTest), 1);
disp('--- VALUTAZIONE UNIFICATA MODELLI POLINOMIALI ---');
disp('Addestramento (Random CV) su 80% dei dati storici. Validazione su 20%.');
% 4. Ciclo di addestramento e valutazione
for d = degreesTest
    formula = sprintf('poly%d', d); 
    model = fitlm(x_train, y_train, formula);
    
    AIC_values(d) = model.ModelCriterion.AIC;
    
    k = d + 1; 
    SSR = model.SSE; 
    FPE_values(d) = (SSR / N_train) * ((N_train + k) / (N_train - k));
    
    y_pred_future = predict(model, x_test);
    RMSE_test(d) = sqrt(mean((y_test - y_pred_future).^2));
end
disp(' ');
finalTable = table(degreesTest', AIC_values, FPE_values, RMSE_test, ...
    'VariableNames', {'Grado', 'AIC_Teorico', 'FPE_Teorico', 'RMSE_Empirico_Test'});
disp(finalTable);
[~, best_idx] = min(RMSE_test);
final_degree = degreesTest(best_idx);
fprintf('\n=> CONCLUSIONE:\n');
fprintf('Secondo la validazione empirica, il modello polinomiale migliore è il Grado %d.\n', final_degree);
% -------------------------------------------------------------------------
% --- GRAFICO DEL MIGLIOR MODELLO POLINOMIALE ---
% -------------------------------------------------------------------------
disp('Generazione grafico del miglior modello polinomiale...');
% Ri-addestro il modello vincente su TUTTI i dati per tracciare una curva completa
final_formula = sprintf('poly%d', final_degree);
winning_model = fitlm(x_valid, y_valid, final_formula);
figure('Name', sprintf('Miglior Polinomio: Grado %d', final_degree), 'NumberTitle', 'off');
% Nuvola di punti dei dati reali
scatter(x_valid, y_valid, 10, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
% Creazione di un asse continuo per far venire la curva fluida
x_axis_plot = linspace(min(x_valid), max(x_valid), 200)';
y_winning_curve = predict(winning_model, x_axis_plot);
% Disegno la curva del modello vincente
plot(x_axis_plot, y_winning_curve, 'r-', 'LineWidth', 3);
xlabel('Temperatura Media (°F)', 'FontWeight', 'bold');
ylabel('Carico Elettrico (LOAD)', 'FontWeight', 'bold');
title(sprintf('Modello Polinomiale Vincente (Grado %d) vs Dati Reali', final_degree), 'FontSize', 12);
legend('Dati Reali (Osservazioni)', sprintf('Modello Polinomiale (Grado %d)', final_degree), 'Location', 'north');
grid on;
hold off;

% ---> ESPORTAZIONE GRAFICO MIGLIOR POLINOMIO <---
exportgraphics(gcf, 'immagini/punto3_miglior_polinomio.png', 'Resolution', 300);

% -------------------------------------------------------------------------
% CONFRONTO POLINOMI E SPLINE (LIN E NON-LIN)
% -------------------------------------------------------------------------
disp(' ');
disp('--- CREAZIONE MODELLI AVANZATI E SPLINE ---');
model_deg4 = fitlm(x_train, y_train, 'poly4');
model_deg5 = fitlm(x_train, y_train, 'poly5');
x_axis = linspace(min(x_valid), max(x_valid), 200)';
% Trovo il Nodo sul Grado 5
y_poly5_curve = predict(model_deg5, x_axis);
[~, idx_min_curve] = min(y_poly5_curve);
knot_temp = x_axis(idx_min_curve);
fprintf('Punto di minimo (Knot) per la Spline trovato a: %.2f °F\n', knot_temp);
% --- A. Calcolo Spline Lineare ---
H_train = [ones(N_train, 1), max(0, knot_temp - x_train), max(0, x_train - knot_temp)]; 
spline_weights = lscov(H_train, y_train);
H_test = [ones(length(x_test), 1), max(0, knot_temp - x_test), max(0, x_test - knot_temp)];
RMSE_spline = sqrt(mean((y_test - (H_test * spline_weights)).^2));
fprintf('RMSE Spline Lineare: %.2f\n', RMSE_spline);
% --- B. Calcolo Spline Non Lineare (Quadratica) ---
H_train_NL = [ones(N_train, 1), ...
              max(0, knot_temp - x_train), max(0, x_train - knot_temp), ...
              (max(0, knot_temp - x_train)).^2, (max(0, x_train - knot_temp)).^2];
beta_NL = lscov(H_train_NL, y_train);
H_test_NL = [ones(length(x_test), 1), ...
             max(0, knot_temp - x_test), max(0, x_test - knot_temp), ...
             (max(0, knot_temp - x_test)).^2, (max(0, x_test - knot_temp)).^2];
RMSE_snl = sqrt(mean((y_test - (H_test_NL * beta_NL)).^2));
fprintf('RMSE Spline NON Lineare (Quadratica): %.2f\n', RMSE_snl);
% -------------------------------------------------------------------------
% --- GRAFICO 1: LE CURVE SOVRAPPOSTE ---
% -------------------------------------------------------------------------
figure('Name', 'Confronto Curve: Poly4, Poly5, Spline Lin e Spline Non Lin', 'NumberTitle', 'off');
scatter(x_valid, y_valid, 10, [0.8 0.8 0.8], 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
% Previsioni per l'asse fluido
y_curve_poly4 = predict(model_deg4, x_axis);
y_curve_poly5 = predict(model_deg5, x_axis);
H_axis = [ones(length(x_axis), 1), max(0, knot_temp - x_axis), max(0, x_axis - knot_temp)];
y_curve_spline = H_axis * spline_weights;
H_axis_NL = [ones(length(x_axis), 1), ...
             max(0, knot_temp - x_axis), max(0, x_axis - knot_temp), ...
             (max(0, knot_temp - x_axis)).^2, (max(0, x_axis - knot_temp)).^2];
y_curve_snl = H_axis_NL * beta_NL;
% Tracciato curve
plot(x_axis, y_curve_poly4, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Grado 4 (Quartico)');
plot(x_axis, y_curve_poly5, 'm-', 'LineWidth', 1.5, 'DisplayName', 'Grado 5 (Quintico)');
plot(x_axis, y_curve_spline, 'b--', 'LineWidth', 2, 'DisplayName', 'Spline Lineare');
plot(x_axis, y_curve_snl, 'r-', 'LineWidth', 3, 'DisplayName', 'Spline Non Lineare (Quadratica)');
xline(knot_temp, 'k--', 'LineWidth', 1, 'DisplayName', 'Nodo (Minimo)');
xlabel('Temperatura Media (°F)', 'FontWeight', 'bold');
ylabel('Carico Elettrico (LOAD)', 'FontWeight', 'bold');
title('Evoluzione del Modello Termico', 'FontSize', 12);
legend('Location', 'north');
grid on;
hold off;

% ---> ESPORTAZIONE GRAFICO CONFRONTO CURVE <---
exportgraphics(gcf, 'immagini/punto3_confronto_curve.png', 'Resolution', 300);

% -------------------------------------------------------------------------
% --- GRAFICO 2: GOODNESS OF FIT (PREDETTI vs OSSERVAZIONI REALI) ---
% -------------------------------------------------------------------------
disp(' ');
disp('--- GENERAZIONE GRAFICI GOODNESS OF FIT ---');
% 1. Estrazione dei valori predetti dai 4 modelli
pred_p4 = predict(model_deg4, x_train);
pred_p5 = predict(model_deg5, x_train);
pred_sl = H_train * spline_weights; 
pred_snl = H_train_NL * beta_NL; 
% Trovare i limiti globali per garantire la stessa scala su tutti i grafici
min_val = min([y_train; pred_p4; pred_p5; pred_sl; pred_snl]);
max_val = max([y_train; pred_p4; pred_p5; pred_sl; pred_snl]);
% 2. Creazione della figura
figure('Name', 'Goodness of Fit: Predetti vs Osservazioni Reali', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);
% --- Grado 4 ---
subplot(2,2,1); 
scatter(pred_p4, y_train, 5, [0.4 0.6 0.8], 'filled', 'MarkerFaceAlpha', 0.2); hold on;
% Retta Affine (blu)
p4_fit = polyfit(pred_p4, y_train, 1); 
plot([min_val max_val], polyval(p4_fit, [min_val max_val]), 'b-', 'LineWidth', 1.5); 
% Bisettrice (rossa tratteggiata, in primo piano)
plot([min_val max_val], [min_val max_val], 'r--', 'LineWidth', 2.5); 
hold off;
title('GoF: Polinomio Grado 4'); 
xlabel('Valori Predetti'); ylabel('Osservazioni Reali (LOAD)'); 
axis square; xlim([min_val max_val]); ylim([min_val max_val]); grid on;
% --- Grado 5 ---
subplot(2,2,2); 
scatter(pred_p5, y_train, 5, [0.4 0.8 0.6], 'filled', 'MarkerFaceAlpha', 0.2); hold on;
% Retta Affine (blu)
p5_fit = polyfit(pred_p5, y_train, 1); 
plot([min_val max_val], polyval(p5_fit, [min_val max_val]), 'b-', 'LineWidth', 1.5); 
% Bisettrice (rossa tratteggiata, in primo piano)
plot([min_val max_val], [min_val max_val], 'r--', 'LineWidth', 2.5); 
hold off;
title('GoF: Polinomio Grado 5'); 
xlabel('Valori Predetti'); ylabel('Osservazioni Reali (LOAD)'); 
axis square; xlim([min_val max_val]); ylim([min_val max_val]); grid on;
% --- Spline Lineare ---
subplot(2,2,3); 
scatter(pred_sl, y_train, 5, [0.8 0.4 0.4], 'filled', 'MarkerFaceAlpha', 0.2); hold on;
% Retta Affine (blu)
sl_fit = polyfit(pred_sl, y_train, 1); 
plot([min_val max_val], polyval(sl_fit, [min_val max_val]), 'b-', 'LineWidth', 1.5); 
% Bisettrice (rossa tratteggiata, in primo piano)
plot([min_val max_val], [min_val max_val], 'r--', 'LineWidth', 2.5); 
hold off;
title('GoF: Spline Lineare'); 
xlabel('Valori Predetti'); ylabel('Osservazioni Reali (LOAD)'); 
axis square; xlim([min_val max_val]); ylim([min_val max_val]); grid on;
% --- Spline Non Lineare ---
subplot(2,2,4); 
scatter(pred_snl, y_train, 5, [0.8 0.6 0.2], 'filled', 'MarkerFaceAlpha', 0.2); hold on;
% Retta Affine (blu)
snl_fit = polyfit(pred_snl, y_train, 1); 
plot([min_val max_val], polyval(snl_fit, [min_val max_val]), 'b-', 'LineWidth', 1.5); 
% Bisettrice (rossa tratteggiata, in primo piano)
plot([min_val max_val], [min_val max_val], 'r--', 'LineWidth', 2.5); 
hold off;
title('GoF: Spline Non Lineare'); 
xlabel('Valori Predetti'); ylabel('Osservazioni Reali (LOAD)'); 
axis square; xlim([min_val max_val]); ylim([min_val max_val]); grid on;

% ---> ESPORTAZIONE GRAFICO GOODNESS OF FIT <---
exportgraphics(gcf, 'immagini/punto3_gof.png', 'Resolution', 300);

% -------------------------------------------------------------------------
% --- GRAFICO 3: ANALISI DEI RESIDUI (SUL MODELLO MIGLIORE) ---
% -------------------------------------------------------------------------
disp('--- GENERAZIONE GRAFICI DEI RESIDUI ---');
% Calcolo dei residui per il modello scelto (Spline Non Lineare)
residuals_snl = y_train - pred_snl;
hour_train = mod(t_train, 24); 
figure('Name', 'Analisi dei Residui: Spline Non Lineare', 'NumberTitle', 'off', 'Position', [150, 150, 1200, 400]);
% 1. Residui vs Temperatura
subplot(1,3,1);
scatter(x_train, residuals_snl, 5, [0.3 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.3);
yline(0, 'r-', 'LineWidth', 2); % Linea dello zero
title('Residui vs Temperatura Media');
xlabel('Temperatura Media (°F)'); ylabel('Residuo (Reale - Predetto)');
grid on;
% 2. Residui vs Ora del Giorno
subplot(1,3,2);
scatter(hour_train, residuals_snl, 5, [0.3 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.3);
yline(0, 'r-', 'LineWidth', 2); % Linea dello zero
title('Residui vs Ora del Giorno');
xlabel('Ora del Giorno (0 - 24)'); ylabel('Residuo');
xlim([0 24]); grid on;
% 3. Distribuzione dei Residui (Istogramma)
subplot(1,3,3);
histogram(residuals_snl, 50, 'Normalization', 'pdf', 'FaceColor', [0.6 0.6 0.6]);
title('Distribuzione dei Residui');
xlabel('Errore Residuo'); ylabel('Densità di Probabilità');
grid on;

% ---> ESPORTAZIONE GRAFICO RESIDUI <---
exportgraphics(gcf, 'immagini/punto3_residui.png', 'Resolution', 300);

% -------------------------------------------------------------------------
% --- AGGIUNTA: GRAFICO SINGOLO SPLINE LINEARE ---
% -------------------------------------------------------------------------
figure('Name', 'Dettaglio: Spline Lineare', 'NumberTitle', 'off');
scatter(x_valid, y_valid, 10, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(x_axis, y_curve_spline, 'b-', 'LineWidth', 2.5);
xline(knot_temp, 'k--', 'LineWidth', 1.5);
xlabel('Temperatura Media (°F)');
ylabel('Carico Elettrico (LOAD)');
title(sprintf('Modello Spline Lineare (Nodo a %.2f °F)', knot_temp));
legend('Dati Reali', 'Spline Lineare', 'Nodo (Knot)', 'Location', 'north');
grid on;
hold off;

% ---> ESPORTAZIONE GRAFICO DETTAGLIO SPLINE LINEARE <---
exportgraphics(gcf, 'immagini/punto3_dettaglio_spline.png', 'Resolution', 300);


%% 4. Fittare un modello ai minimi quadrati usando tutte le 25 variabili di temperatura e discutere
% eventuali problemi di collinearita' e di non significativita' di alcuni coefficienti;

disp(' ');
disp('===============================================================');
disp('--- PUNTO 4: SELEZIONE DELLE VARIABILI E GESTIONE COLLINEARITA'' ---');

% 1. ESTRAZIONE DELLE 25 TEMPERATURE
% Uso gli indici 'idx_train' e 'idx_test' calcolati al Punto 3
% Forziamo la conversione in 'double' per evitare la creazione di variabili dummy
X_train_25 = double(data{idx_train, 3:27}); 
y_train_25 = double(y_train);
X_test_25 = double(data{idx_test, 3:27});
y_test_25 = double(y_test);

% -------------------------------------------------------------------------
% --- 2. SUBSAMPLING (Ottimizzazione del Costo Computazionale) ---
% -------------------------------------------------------------------------
% Estrazione di un sottoinsieme per ridurre i tempi di calcolo e la memoria
% richiesta dagli algoritmi iterativi (Stepwise) e di Cross-Validation.
N_train_total = length(y_train_25);
N_sub = floor(0.10 * N_train_total);

rng(42); % Fisso il seed per la riproducibilità
idx_sub = randperm(N_train_total, N_sub);

% Matrici
X_sub = X_train_25(idx_sub, :);
y_sub = y_train_25(idx_sub);

disp('--- RIDUZIONE DEL COSTO COMPUTAZIONALE (SUBSAMPLING) ---');
fprintf('Al fine di abbattere il costo computazionale e i tempi di calcolo,\n');
fprintf('è stato applicato un campionamento casuale (subsampling) al training set.\n');
fprintf('  -> Numero iniziale di osservazioni: %d\n', N_train_total);
fprintf('  -> Numero finale di osservazioni utilizzate: %d (10%%)\n\n', N_sub);

% Forzo la matrice e imposto a MATLAB che ci sono ZERO categorie
X_mat = double(X_sub);
y_mat = double(y_sub);
no_categories = false(1, size(X_mat, 2));

% -------------------------------------------------------------------------
% --- 3. DIMOSTRAZIONE MULTICOLLINEARITA' (Modello OLS Completo) ---
% -------------------------------------------------------------------------
disp('--- ANALISI OLS: IL PROBLEMA DELLA MULTICOLLINEARITA'' ---');
disp('Fit di un modello lineare (Minimi Quadrati Ordinari) con tutte le 25 stazioni.');

% Fit del modello base
model_ols = fitlm(X_mat, y_mat, 'linear', 'CategoricalVars', no_categories);

% Estrazione dei p-value (escludendo l'intercetta che è il primo elemento)
p_values = model_ols.Coefficients.pValue(2:end);
num_non_sig = sum(p_values > 0.05); % Conto quanti hanno p-value > 5%

% Sintesi numerica e la tabella dei coefficienti
fprintf('=> SINTESI: %d coefficienti su 25 (%.1f%%) risultano NON significativi (p-value > 0.05).\n', ...
    num_non_sig, (num_non_sig/25)*100);
disp('Questo conferma l''elevata ridondanza e l''instabilità del modello globale.');
disp('Di seguito un estratto dei coefficienti OLS (Stima, Errore Standard, p-Value):');
disp(model_ols.Coefficients);
disp('---------------------------------------------------------------');


% -------------------------------------------------------------------------
% --- 4. ESECUZIONE DEGLI ALGORITMI DI SELEZIONE ---
% -------------------------------------------------------------------------
% A. FORWARD STEPWISE
disp('1/4 - Calcolo Stepwise FORWARD...');
model_fwd = stepwiselm(X_mat, y_mat, 'constant', 'Upper', 'linear', ...
    'CategoricalVars', no_categories, 'Verbose', 0);
vars_fwd = model_fwd.NumCoefficients - 1;
rmse_fwd = sqrt(mean((y_test_25 - predict(model_fwd, X_test_25)).^2));

% B. BACKWARD STEPWISE
disp('2/4 - Calcolo Stepwise BACKWARD...');
model_bwd = stepwiselm(X_mat, y_mat, 'linear', 'Lower', 'constant', 'Upper', 'linear', ...
    'CategoricalVars', no_categories, 'Verbose', 0);
vars_bwd = model_bwd.NumCoefficients - 1;
rmse_bwd = sqrt(mean((y_test_25 - predict(model_bwd, X_test_25)).^2));

% C. HYBRID STEPWISE 
disp('3/4 - Calcolo Stepwise HYBRID...');
model_hyb = stepwiselm(X_mat, y_mat, 'linear', 'Upper', 'linear', ...
    'CategoricalVars', no_categories, 'Verbose', 0);
vars_hyb = model_hyb.NumCoefficients - 1;
rmse_hyb = sqrt(mean((y_test_25 - predict(model_hyb, X_test_25)).^2));

% D. REGOLARIZZAZIONE LASSO (Evoluzione con Basi Spline)
disp('4/4 - Calcolo Regolarizzazione LASSO (con Basi Spline)...');

station_names = data.Properties.VariableNames(3:27);

% Matrice di Sensitività a 50 variabili (25 Freddo + 25 Caldo)
X_sub_spline = zeros(size(X_mat, 1), 50);
X_test_spline = zeros(size(X_test_25, 1), 50);
station_names_spline = cell(1, 50);

for j = 1:25
    % Ramo Invernale (Freddo)
    X_sub_spline(:, j) = max(0, knot_temp - X_mat(:, j));
    X_test_spline(:, j) = max(0, knot_temp - X_test_25(:, j));
    station_names_spline{j} = sprintf('%s_Freddo', station_names{j});
    
    % Ramo Estivo (Caldo)
    X_sub_spline(:, j+25) = max(0, X_mat(:, j) - knot_temp);
    X_test_spline(:, j+25) = max(0, X_test_25(:, j) - knot_temp);
    station_names_spline{j+25} = sprintf('%s_Caldo', station_names{j});
end

% LASSO con Standardizzazione esplicita dei regressori
[B_lasso, FitInfo] = lasso(X_sub_spline, y_mat, 'CV', 5, 'Standardize', true); 

% Estrazione Modello Min MSE (Minimo Errore)
idx_opt = FitInfo.IndexMinMSE; 
coef_lasso = B_lasso(:, idx_opt); 
intercept_lasso = FitInfo.Intercept(idx_opt);
vars_lasso = sum(coef_lasso ~= 0); 
y_pred_lasso = X_test_spline * coef_lasso + intercept_lasso;
rmse_lasso = sqrt(mean((y_test_25 - y_pred_lasso).^2));

% Estrazione Modello 1-SE (Più Parsimonioso)
idx_1se = FitInfo.Index1SE; 
coef_lasso_1se = B_lasso(:, idx_1se);
vars_lasso_1se = sum(coef_lasso_1se ~= 0); 

disp('Elaborazione completata.');

% -------------------------------------------------------------------------
% --- TABELLA RIASSUNTIVA DEL CONFRONTO ---
% -------------------------------------------------------------------------
Methods = {'Forward (Lin)'; 'Backward (Lin)'; 'Hybrid (Lin)'; 'LASSO (Spline MinMSE)'};
Selected_Variables = [vars_fwd; vars_bwd; vars_hyb; vars_lasso];
RMSE_Validation_Set = [rmse_fwd; rmse_bwd; rmse_hyb; rmse_lasso];

Comparison_Table_P4 = table(Selected_Variables, RMSE_Validation_Set, 'RowNames', Methods);
disp(' ');
disp('--- RISULTATI CONFRONTATI (Validati sul Validation Set) ---');
disp(Comparison_Table_P4);
fprintf('Nota: La regola di 1-Standard-Error del LASSO avrebbe selezionato %d variabili.\n', vars_lasso_1se);
disp('===============================================================');

% -------------------------------------------------------------------------
% --- GRAFICI DEL LASSO ---
% -------------------------------------------------------------------------
disp('Generazione grafici LASSO...');

% 1. Grafico Cross-Validation
figure_lasso_cv = figure; 
lassoPlot(B_lasso, FitInfo, 'PlotType', 'CV');
set(gcf, 'Name', 'Analisi LASSO: Cross Validation', 'NumberTitle', 'off', 'Position', [100, 100, 600, 450]);
title('Ricerca del \lambda ottimale tramite Cross-Validation');
grid on;

% ---> ESPORTAZIONE GRAFICO LASSO CROSS-VALIDATION <---
exportgraphics(gcf, 'immagini/punto4_lasso_cv.pdf', 'ContentType', 'vector');


% 2. Grafico Shrinkage
figure_lasso_shrink = figure; 
lassoPlot(B_lasso, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log');
set(gcf, 'Name', 'Analisi LASSO: Shrinkage', 'NumberTitle', 'off', 'Position', [750, 100, 600, 450]);
title('LASSO: Restringimento dei coefficienti (Shrinkage)');
% Chiarimento della direzione di lettura dell'asse X
set(gca, 'XDir', 'reverse'); 
xlabel('Iperparametro \lambda (Penalizzazione crescente verso sinistra \leftarrow)');
grid on;

% ---> ESPORTAZIONE GRAFICO LASSO SHRINKAGE <---
exportgraphics(gcf, 'immagini/punto4_lasso_shrinkage.pdf', 'ContentType', 'vector');

% -------------------------------------------------------------------------
% --- DETTAGLIO: QUALI STAZIONI SONO STATE SELEZIONATE? ---
% -------------------------------------------------------------------------
disp(' ');
disp('--- DETTAGLIO STAZIONI METEO SELEZIONATE ---');

% 1. Estrazione per FORWARD
idx_fwd = str2double(regexprep(model_fwd.PredictorNames, 'x', ''));
fprintf('\n=> FORWARD (Lineare) ha scelto %d stazioni:\n', length(idx_fwd));
disp(strjoin(station_names(idx_fwd), ', '));

% 2. Estrazione per BACKWARD
idx_bwd = str2double(regexprep(model_bwd.PredictorNames, 'x', ''));
fprintf('\n=> BACKWARD (Lineare) ha scelto %d stazioni:\n', length(idx_bwd));
disp(strjoin(station_names(idx_bwd), ', '));

% 3. Estrazione per HYBRID
idx_hyb = str2double(regexprep(model_hyb.PredictorNames, 'x', ''));
fprintf('\n=> HYBRID (Lineare) ha scelto %d stazioni:\n', length(idx_hyb));
disp(strjoin(station_names(idx_hyb), ', '));

% 4. Estrazione per LASSO (Ora sceglie tra Freddo e Caldo)
idx_lasso = find(coef_lasso ~= 0);
fprintf('\n=> LASSO (Spline) ha scelto %d rami termici:\n', length(idx_lasso));
disp(strjoin(station_names_spline(idx_lasso), ', '));

disp('===============================================================');

%% 5. Fittare un modello tra LOAD e l'ora del giorno come unico regressore;

disp(' ');
disp('===============================================================');
disp('--- PUNTO 5: MODELLO CON L''ORA DEL GIORNO E ARMONICHE ---');

% 1. SINCRONIZZAZIONE INDICI E CONVERSIONE ORA
t_train_sync = t_valid(idx_train);
t_test_sync  = t_valid(idx_test);

if isdatetime(t_valid)
    hours_train = hour(t_train_sync) + minute(t_train_sync)/60;
    hours_test  = hour(t_test_sync)  + minute(t_test_sync)/60;
    hours_valid = hour(t_valid) + minute(t_valid)/60;
else
    hours_train = t_train_sync; 
    hours_test  = t_test_sync; 
    hours_valid = t_valid;
end

% -------------------------------------------------------------------------
% --- A. MODELLI POLINOMIALI SUL TEMPO ---
% -------------------------------------------------------------------------
disp('1. Ricerca del miglior polinomio per l''ora del giorno...');
time_degrees = 1:5;
RMSE_time_test = zeros(length(time_degrees), 1);
for d = time_degrees
    formula_t = sprintf('poly%d', d);
    model_t = fitlm(hours_train, y_train, formula_t);
    y_pred_t = predict(model_t, hours_test);
    RMSE_time_test(d) = sqrt(mean((y_test - y_pred_t).^2));
end
[min_rmse_t, best_idx_t] = min(RMSE_time_test);
final_degree_t = time_degrees(best_idx_t);
fprintf('=> Polinomio vincente: Grado %d (RMSE = %.2f)\n', final_degree_t, min_rmse_t);

% -------------------------------------------------------------------------
% --- B. RICERCA DEL NUMERO OTTIMO DI ARMONICHE (Seno/Coseno) ---
% -------------------------------------------------------------------------
disp('2. Ricerca del numero ottimo di Armoniche (k)...');
K_max = 8;
AIC_arm = zeros(K_max, 1);
for k = 1:K_max
    H_train_arm = ones(length(hours_train), 1);
    for j = 1:k
        H_train_arm = [H_train_arm, sin(2*pi*j*hours_train/24), cos(2*pi*j*hours_train/24)];
    end
    beta_arm = lscov(H_train_arm, y_train);
    SSR = sum((y_train - H_train_arm*beta_arm).^2);
    AIC_arm(k) = length(y_train) * log(SSR/length(y_train)) + 2 * (1 + 2*k);
end
[~, k_opt] = min(AIC_arm);
fprintf('=> Numero ottimo di Armoniche scelto (AIC): k = %d\n', k_opt);

% NOTA METODOLOGICA: Trade-off AIC vs Parsimonia
disp('   [NOTA]: L''AIC seleziona matematicamente k=7, ma un''analisi visiva');
disp('   del grafico mostra che k=2 o k=3 catturano gia'' il 90% della struttura');
disp('   giornaliera. L''uso di k=7 modella principalmente micro-dinamiche.');

t_fluid = linspace(0, 24, 500)';
H_valid_arm = ones(length(hours_valid), 1);
H_axis_arm = ones(500, 1);
for j = 1:k_opt
    H_valid_arm = [H_valid_arm, sin(2*pi*j*hours_valid/24), cos(2*pi*j*hours_valid/24)];
    H_axis_arm  = [H_axis_arm,  sin(2*pi*j*t_fluid/24),  cos(2*pi*j*t_fluid/24)];
end
beta_opt_arm = lscov(H_valid_arm, y_valid); 
y_curve_arm = H_axis_arm * beta_opt_arm;

% -------------------------------------------------------------------------
% --- C. SPLINE ORARIA SUI VERTICI ---
% -------------------------------------------------------------------------
disp('3. Costruzione Spline Oraria sui vertici...');
hour_knots = [6, 12, 18]; 
H_train_spline_hour = [ones(length(hours_train), 1), hours_train];
H_axis_spline = [ones(500, 1), t_fluid];
for i = 1:length(hour_knots)
    H_train_spline_hour = [H_train_spline_hour, max(0, hours_train - hour_knots(i))];
    H_axis_spline = [H_axis_spline, max(0, t_fluid - hour_knots(i))];
end
beta_spline_hour = lscov(H_train_spline_hour, y_train);
y_curve_spline = H_axis_spline * beta_spline_hour;

% NOTA METODOLOGICA: Limite della Periodicità
disp('   [NOTA]: La Spline lineare applicata all''orario (senza vincoli ai bordi)');
disp('   non garantisce che f(0) == f(24). Condivide quindi lo stesso difetto');
disp('   concettuale del polinomio: il salto a mezzanotte.');

% -------------------------------------------------------------------------
% --- GRAFICI PUNTO 5 ---
% -------------------------------------------------------------------------
disp('Generazione grafici del Punto 5...');

% GRAFICO 1: Analisi Routine (AIC e Forme)
figure('Name', 'Analisi Routine: Armoniche vs Spline', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 450]);

subplot(1,2,1);
plot(1:K_max, AIC_arm, '-bo', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xline(k_opt, 'r--', 'LineWidth', 2, 'DisplayName', 'Ottimo');
xlabel('Numero di Armoniche (k)', 'FontWeight', 'bold'); ylabel('AIC');
title('Ottimizzazione Armoniche (AIC)'); grid on; legend;

subplot(1,2,2);
plot(t_fluid, y_curve_arm, 'r-', 'LineWidth', 3, 'DisplayName', sprintf('Armonica (k=%d)', k_opt));
hold on;
plot(t_fluid, y_curve_spline, 'b--', 'LineWidth', 3, 'DisplayName', 'Spline Oraria');
xline(hour_knots, 'k:', 'LineWidth', 1, 'HandleVisibility', 'off'); 
xlabel('Ora del Giorno (0-24)', 'FontWeight', 'bold'); ylabel('Carico Predetto');
title('Confronto Forme: Onde vs Vertici');
xlim([0 24]); xticks(0:4:24); 
% Forzo l'asse Y a partire da 0
ylim_current = ylim;
ylim([0, ylim_current(2)]);
grid on; legend('Location', 'northwest'); hold off;

% ---> ESPORTAZIONE GRAFICO ARMONICHE VS SPLINE <---
exportgraphics(gcf, 'immagini/punto5_armoniche_vs_spline.pdf', 'ContentType', 'vector');


% -------------------------------------------------------------------------
% GRAFICO 2A: SCATTERPLOT CON SOLO POLINOMIO
% -------------------------------------------------------------------------
% Addestro esplicitamente il miglior polinomio trovato
best_poly_model = fitlm(hours_valid, y_valid, sprintf('poly%d', final_degree_t));
y_line_poly = predict(best_poly_model, t_fluid);

figure('Name', 'Analisi della Routine: Solo Polinomio', 'NumberTitle', 'off', 'Position', [150, 150, 900, 500]);
scatter(hours_valid, y_valid, 10, x_valid, 'filled', 'MarkerFaceAlpha', 0.5);
hold on; colormap(turbo); cb = colorbar;
ylabel(cb, 'Temperatura Media (°F)', 'FontWeight', 'bold');
% Linea Nera CONTINUA ('k-'), senza l'armonica rossa
plot(t_fluid, y_line_poly, 'k-', 'LineWidth', 3, 'DisplayName', sprintf('Polinomio (Grado %d)', final_degree_t));
xlabel('Ora del giorno (0-24)', 'FontWeight', 'bold');
ylabel('Carico Elettrico (LOAD)', 'FontWeight', 'bold');
title(sprintf('Analisi Routine: Modello Polinomiale (Grado %d)', final_degree_t), 'FontSize', 12);
xlim([0 24]); xticks(0:4:24);
legend('Dati Reali (Colorati per Temp)', sprintf('Polinomio (Grado %d)', final_degree_t), 'Location', 'northwest');
grid on; hold off;

% ---> ESPORTAZIONE GRAFICO SOLO POLINOMIO <---
exportgraphics(gcf, 'immagini/punto5_solo_polinomio.png', 'Resolution', 300);


% -------------------------------------------------------------------------
% GRAFICO 2B: CONFRONTO POLINOMIO VS ARMONICA (La Soluzione)
% -------------------------------------------------------------------------
figure('Name', 'Analisi della Routine: Polinomio vs Ciclo Naturale', 'NumberTitle', 'off', 'Position', [200, 200, 900, 500]);
scatter(hours_valid, y_valid, 10, x_valid, 'filled', 'MarkerFaceAlpha', 0.5);
hold on; colormap(turbo); cb = colorbar;
ylabel(cb, 'Temperatura Media (°F)', 'FontWeight', 'bold');
% Linee di confronto: Polinomio TRATTEGGIATO ('k--') e Armonica ROSSA ('r-')
plot(t_fluid, y_line_poly, 'k--', 'LineWidth', 3, 'DisplayName', sprintf('Polinomio (Grado %d)', final_degree_t));
plot(t_fluid, y_curve_arm, 'r-', 'LineWidth', 4, 'DisplayName', sprintf('Modello Armonico (k=%d)', k_opt));
xlabel('Ora del giorno (0-24)', 'FontWeight', 'bold');
ylabel('Carico Elettrico (LOAD)', 'FontWeight', 'bold');
title(sprintf('Analisi Routine: Rigidità (Polinomio G%d) vs Ciclo Naturale (Armonica)', final_degree_t), 'FontSize', 12);
xlim([0 24]); xticks(0:4:24);
legend('Dati Reali (Colorati per Temp)', sprintf('Polinomio (Grado %d)', final_degree_t), sprintf('Modello Armonico (k=%d)', k_opt), 'Location', 'northwest');
grid on; hold off;

% ---> ESPORTAZIONE GRAFICO POLINOMIO VS ARMONICA <---
exportgraphics(gcf, 'immagini/punto5_polinomio_vs_armonica.png', 'Resolution', 300);


% -------------------------------------------------------------------------
% --- PUNTO 7: MODELLO ADDITIVO 3D (Spline NL Temp + Spline Ora) ---
% -------------------------------------------------------------------------
disp('Creazione Modello Additivo 3D e Superficie (Punto 7)...');

H_additive_train = [ones(length(y_train), 1), ...
                    max(0, knot_temp - x_train), max(0, x_train - knot_temp), ... 
                    (max(0, knot_temp - x_train)).^2, (max(0, x_train - knot_temp)).^2, ... 
                    hours_train]; 
for i = 1:length(hour_knots)
    H_additive_train = [H_additive_train, max(0, hours_train - hour_knots(i))];
end
beta_additive = lscov(H_additive_train, y_train);

% GRAFICO 3: SUPERFICIE 3D
figure('Name', 'Superficie Additiva (Spline NL Temp + Spline Ora)', 'NumberTitle', 'off');
[X_grid, T_grid] = meshgrid(linspace(min(x_valid), max(x_valid), 50), ...
                            linspace(0, 24, 50));
H_grid = [ones(numel(X_grid), 1), ...
          max(0, knot_temp - X_grid(:)), max(0, X_grid(:) - knot_temp), ...
          (max(0, knot_temp - X_grid(:))).^2, (max(0, X_grid(:) - knot_temp)).^2, ...
          T_grid(:)];
for i = 1:length(hour_knots)
    H_grid = [H_grid, max(0, T_grid(:) - hour_knots(i))];
end
Z_grid = reshape(H_grid * beta_additive, size(X_grid));

surf(X_grid, T_grid, Z_grid, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
colormap(turbo); colorbar; hold on;
scatter3(x_valid(1:10:end), hours_valid(1:10:end), y_valid(1:10:end), ...
         5, 'k', 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Temperatura (°F)', 'FontWeight', 'bold');
ylabel('Ora del Giorno (0-24)', 'FontWeight', 'bold');
zlabel('Carico Elettrico (LOAD)', 'FontWeight', 'bold');
title('Modello 3D Additivo: Spline Non Lineare (Temp) + Spline Oraria (Tempo)', 'FontSize', 12);
view(-45, 30); grid on; hold off;

% ---> ESPORTAZIONE GRAFICO SUPERFICIE 3D <---
exportgraphics(gcf, 'immagini/punto6_superficie_additiva.png', 'Resolution', 300);

disp('Elaborazione Punto 5 completata con successo!');

%% 6. Fittare un modello tra LOAD, w(t)(media) e l'ora del giorno;
disp(' ');
disp('===============================================================');
disp('--- PUNTO 6: IL MODELLO FINALE (TEMPERATURA + TIMESTAMP) ---');

% Costruisco le matrici dei predittori unendo i polinomi vincenti
% final_degree e' il grado migliore per la temperatura (dal Punto 3)
% final_degree_t e' il grado migliore per il tempo (dal Punto 5)
X_train_comb = [];
X_test_comb = [];
X_valid_comb = []; % Per il grafico finale su tutti i dati

% 1. Aggiungo le potenze della temperatura
for i = 1:final_degree
    X_train_comb = [X_train_comb, x_train.^i];
    X_test_comb  = [X_test_comb, x_test.^i];
    X_valid_comb = [X_valid_comb, x_valid.^i];
end

% 2. Aggiungo le potenze del tempo (TIMESTAMP)
for i = 1:final_degree_t
    X_train_comb = [X_train_comb, t_train.^i];
    X_test_comb  = [X_test_comb, t_test.^i];
    X_valid_comb = [X_valid_comb, t_valid.^i];
end

% 3. Addestramento del modello combinato sul Train Set
combined_model = fitlm(X_train_comb, y_train);

% 4. Validazione sul ValidationSet e calcolo RMSE
y_pred_comb = predict(combined_model, X_test_comb);
RMSE_comb = sqrt(mean((y_test - y_pred_comb).^2));

% Addestro il modello finale su tutti i dati per il grafico
final_comb_model = fitlm(X_valid_comb, y_valid);

% =========================================================================
% MODELLO ARMONICO CON INTERAZIONE
% =========================================================================

% 1. Definisco la frequenza angolare (omega) per un ciclo di 24 ore
omega = (2 * pi) / 24; 

% Inizializzo le nuove matrici dei predittori (vuote)
X_train_harm = [];
X_test_harm = [];
X_valid_harm = []; % Per il grafico finale

% 2. COMPONENTE TERMICA: Aggiungo i polinomi della temperatura
for i = 1:final_degree
    X_train_harm = [X_train_harm, x_train.^i];
    X_test_harm  = [X_test_harm, x_test.^i];
    X_valid_harm = [X_valid_harm, x_valid.^i];
end

% Creo vettori temporanei per le sinusoidi per comodità di lettura
sin_train = sin(omega * t_train);
cos_train = cos(omega * t_train);
sin_test  = sin(omega * t_test);
cos_test  = cos(omega * t_test);
sin_valid = sin(omega * t_valid);
cos_valid = cos(omega * t_valid);

% 3. COMPONENTE CICLICA: Aggiungo le armoniche dell'orario
X_train_harm = [X_train_harm, sin_train, cos_train];
X_test_harm  = [X_test_harm, sin_test, cos_test];
X_valid_harm = [X_valid_harm, sin_valid, cos_valid];

% 4. L'INTERAZIONE: Moltiplico la Temperatura (grado 1) per le sinusoidi
% Questo permette alla temperatura di avere un peso elastico in base all'ora
X_train_harm = [X_train_harm, x_train .* sin_train, x_train .* cos_train];
X_test_harm  = [X_test_harm, x_test .* sin_test, x_test .* cos_test];
X_valid_harm = [X_valid_harm, x_valid .* sin_valid, x_valid .* cos_valid];

% 5. Addestro il modello armonico definitivo
harmonic_model = fitlm(X_train_harm, y_train);

% 6. Validazione sul ValidationSet e calcolo dell'Errore
y_pred_harm = predict(harmonic_model, X_test_harm);
RMSE_harm = sqrt(mean((y_test - y_pred_harm).^2));

% =========================================================================
%  MODELLO 3D ADDITIVO (SPLINE NL + SPLINE ORARIA)
% =========================================================================
% Assumo knot_temp dal Punto 3 (es. 59.65). 
% Per la spline oraria uso i nodi ore 6, 12, 18.
knots_time = [6, 12, 18];

% A. Componente Termica: Spline Non Lineare (Quadrato)
T_f_tr = max(0, knot_temp - x_train);
T_c_tr = max(0, x_train - knot_temp);
T_f2_tr = T_f_tr.^2;
T_c2_tr = T_c_tr.^2;
X_spline_temp_tr = [T_f_tr, T_c_tr, T_f2_tr, T_c2_tr];

T_f_te = max(0, knot_temp - x_test);
T_c_te = max(0, x_test - knot_temp);
T_f2_te = T_f_te.^2;
T_c2_te = T_c_te.^2;
X_spline_temp_te = [T_f_te, T_c_te, T_f2_te, T_c2_te];

% B. Componente Temporale: Spline Lineare Oraria
X_spline_time_tr = t_train;
X_spline_time_te = t_test;
for k_t = knots_time
    X_spline_time_tr = [X_spline_time_tr, max(0, t_train - k_t)];
    X_spline_time_te = [X_spline_time_te, max(0, t_test - k_t)];
end

% C. Matrice Additiva Finale
X_train_add_spline = [X_spline_temp_tr, X_spline_time_tr];
X_test_add_spline  = [X_spline_temp_te, X_spline_time_te];

% D. Addestramento e calcolo RMSE
add_spline_model = fitlm(X_train_add_spline, y_train);
y_pred_add_spline = predict(add_spline_model, X_test_add_spline);
RMSE_add_spline = sqrt(mean((y_test - y_pred_add_spline).^2));


% =========================================================================
% --- STAMPA RIASSUNTIVA DI TUTTI GLI RMSE DEL PUNTO 6 ---
% =========================================================================
disp(' ');
disp('=> RIEPILOGO PERFORMANCE (Validation Set):');
fprintf('1. Modello Base Termico (Polinomio Grado %d):          %.2f\n', final_degree, RMSE_test(final_degree));
fprintf('2. Modello Base Temporale (Polinomio Grado %d):        %.2f\n', final_degree_t, min_rmse_t);
fprintf('3. Modello Multivariato Additivo (Temp + Tempo):     %.2f\n', RMSE_comb);
fprintf('4. Modello Armonico con Interazione (Temp x Tempo):  %.2f\n', RMSE_harm);
fprintf('5. Modello 3D Additivo (Spline NL + Spline Oraria):  %.2f\n', RMSE_add_spline);
disp('===============================================================');

% =========================================================================
% --- SEZIONE GRAFICI ---
% =========================================================================

figure('Name', 'Modello 3D: LOAD = f(Temp, Tempo)', 'NumberTitle', 'off');
% Scatter 3D dei dati reali
scatter3(t_valid, x_valid, y_valid, 10, y_valid, 'filled', 'MarkerFaceAlpha', 0.2);
colormap(turbo);
hold on;

% Creo una griglia 2D (Mesh) per la superficie del modello
t_grid = linspace(min(t_valid), max(t_valid), 50)';
x_grid = linspace(min(x_valid), max(x_valid), 50)';
[T_mesh, X_mesh] = meshgrid(t_grid, x_grid);

% Trasformo le matrici in vettori colonna per la funzione predict
T_col = T_mesh(:);
X_col = X_mesh(:);

% Ricostruisco i predittori per la griglia tridimensionale
Grid_Predictors = [];
for i = 1:final_degree
    Grid_Predictors = [Grid_Predictors, X_col.^i];
end
for i = 1:final_degree_t
    Grid_Predictors = [Grid_Predictors, T_col.^i];
end

% Calcolo le previsioni per l'intera griglia
Y_col_pred = predict(final_comb_model, Grid_Predictors);

% Riformatto le previsioni a forma di superficie (matrice)
Y_mesh = reshape(Y_col_pred, size(T_mesh));

% Disegno la superficie semitrasparente
surf(T_mesh, X_mesh, Y_mesh, 'FaceAlpha', 0.7, 'EdgeColor', 'none');

% Etichette e visuale
xlabel('Ora del giorno (TIMESTAMP)', 'FontWeight', 'bold');
ylabel('Temperatura Media (°F)', 'FontWeight', 'bold');
zlabel('Carico Elettrico (LOAD)', 'FontWeight', 'bold');

% Titolo con RMSE_comb integrato
title(sprintf('Superficie del Modello Multivariato Additivo (RMSE Val: %.2f)', RMSE_comb), 'FontSize', 12);

view(-45, 25); % Angolo di rotazione ottimale per vedere la curva a U e l'onda
grid on;
hold off;

% ---> ESPORTAZIONE GRAFICO 1: SUPERFICIE 3D <---
exportgraphics(gcf, 'immagini/punto6_superficie_multivariata.png', 'Resolution', 300);

% --- I 24 "SMILE" ORARI E LO SPOSTAMENTO DEL VERTICE ---
% Controllo che gli orari siano numeri interi per poterli raggruppare
hours = round(t_valid);
hours_unique = unique(hours);

% Creo una figura per ospitare 24 grafici
figure('Name', 'Analisi Oraria: Grafici del Carico Elettrico', ...
       'NumberTitle', 'off', 'Position', [100, 100, 1100, 650]);

% Raccolgo i vertici
vertex_x = zeros(24, 1);
vertex_y = zeros(24, 1);

for i = 1:24
    % Seleziono solo i dati relativi all'ora i-esima
    % (Uso mod per gestire eventuali orari formattati)
    if ismember(i, hours_unique)
        current_time = i;
    else
        current_time = mod(i, 24); 
    end
    
    idx_hour = (hours == current_time);
    
    x_hour = x_valid(idx_hour); % Temperatura per l'ora corrente
    y_hour = y_valid(idx_hour); % Carico per l'ora corrente
    
    % Creo il subplot nella griglia 4 righe x 6 colonne
    subplot(4, 6, i);
    
    if length(x_hour) > 10 % Procedo solo se ci sono abbastanza dati
        scatter(x_hour, y_hour, 5, [0.8 0.8 0.8], 'filled', 'MarkerFaceAlpha', 0.5);
        hold on;
        
        % Fitto un modello quadratico (Grado 2)
        hour_model = fitlm(x_hour, y_hour, 'poly2');
        
        % Traccio la curva dello "smile"
        x_fluid = linspace(min(x_hour), max(x_hour), 100)';
        y_line = predict(hour_model, x_fluid);
        plot(x_fluid, y_line, 'r-', 'LineWidth', 2);
        
        % 4. Calcolo del vertice (Punto di minimo consumo)
        % Matematicamente il vertice di y = ax^2 + bx + c si trova a x = -b/(2a)
        % ma lo cerchiamo sul vettore generato per sicurezza grafica
        [min_y, idx_min] = min(y_line);
        x_vertex = x_fluid(idx_min);
        
        % Salvo il vertice in memoria
        vertex_x(i) = x_vertex;
        vertex_y(i) = min_y;
        
        % 5. Disegno il pallino blu sul vertice
        plot(x_vertex, min_y, 'bo', 'MarkerSize', 7, 'MarkerFaceColor', 'b');
        
        % Formattazione del grafico
        title(sprintf('Ora: %d:00', current_time), 'FontSize', 10);
        xlim([min(x_valid) max(x_valid)]); % Mantengo la stessa scala X per tutti
        ylim([min(y_valid) max(y_valid)]); % Mantengo la stessa scala Y per tutti
        
        % Metto le etichette assi solo sui bordi esterni
        if i >= 19
            xlabel('Temp (°F)');
        end
        if mod(i, 6) == 1
            ylabel('LOAD');
        end
        
        grid on;
        hold off;
    end
end

% ---> ESPORTAZIONE GRAFICO 2: I 24 SMILES <---
exportgraphics(gcf, 'immagini/punto6_24_smiles.png', 'Resolution', 300);

% --- STUDIO DELLO SPOSTAMENTO DEI VERTICI --- 
figure('Name', 'Analisi dei Vertici: Comfort e Carico Base', 'NumberTitle', 'off');

% Grafico 1: Come cambia la Temperatura di Comfort (asse X del vertice)
subplot(2,1,1);
plot(1:24, vertex_x, '-ko', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
xlabel('Ora del Giorno (1-24)', 'FontWeight', 'bold');
ylabel('Temp. di Comfort (°F)', 'FontWeight', 'bold');
title('Spostamento della Temperatura Ideale (Vertice X) nelle 24 Ore');
grid on;
xlim([1 24]);
xticks(1:24);

% Grafico 2: Come cambia il Carico Base (asse Y del vertice)
subplot(2,1,2);
plot(1:24, vertex_y, '-ro', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
xlabel('Ora del Giorno (1-24)', 'FontWeight', 'bold');
ylabel('Carico Base (LOAD)', 'FontWeight', 'bold');
title('Spostamento del Carico Minimo (Vertice Y) nelle 24 Ore');
grid on;
xlim([1 24]);
xticks(1:24);

% ---> ESPORTAZIONE GRAFICO 3: SPOSTAMENTO VERTICI <---
exportgraphics(gcf, 'immagini/punto6_spostamento_vertici.pdf', 'ContentType', 'vector');

% --- GRAFICO: Zoom su tre giorni per vedere l'onda ---
figure('Name', 'Confronto Modello Armonico vs Dati Reali (Zoom)', 'NumberTitle', 'off');

% Prendo un campione di 3 giorni lavorativi (3 * 24 = 72 ore) per il dettaglio giornaliero
sample_hours = 1:min(72, length(y_test));

% Traccio il carico reale
plot(sample_hours, y_test(sample_hours), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Dati Reali (Validation Set)');
hold on;

% Traccio la previsione del modello armonico
plot(sample_hours, y_pred_harm(sample_hours), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Previsione Modello Armonico');

xlabel('Ore (Campione di 3 giorni)', 'FontWeight', 'bold');
ylabel('Carico Elettrico (LOAD)', 'FontWeight', 'bold');

% Titolo con RMSE_harm integrato
title(sprintf('Onda del Modello Armonico con Interazione (RMSE Val: %.2f)', RMSE_harm), 'FontSize', 12);

legend('Location', 'best');
grid on;
hold off;

% ---> ESPORTAZIONE GRAFICO 4: ZOOM SERIE STORICA <---
exportgraphics(gcf, 'immagini/punto6_armonico_zoom.pdf', 'ContentType', 'vector');

%% 7. Proporre un modello finale migliore usando una strategia ben motivata, ad esempio selezione
% stepwise, regressione regolarizzata, modelli non lineari o altre tecniche appropriate.
disp(' ');
disp('===============================================================');
disp('--- PUNTO 7: IL MASTER MODEL (Termodinamica + Routine Umana) ---');

% 1. Recupero parametri (Calcolati nei punti precedenti)
% Uso il knot_temp trovato al Punto 3 e il k_opt trovato al Punto 5
fprintf('Costruzione del Master Model basato su:\n');
fprintf('- Nodo Termico: %.2f °F (Spline Non Lineare)\n', knot_temp);
fprintf('- Ciclo Armonico: %d Frequenze Ottimali\n', k_opt);
fprintf('- Interazione Completa: Stagione x Ora\n');

% -------------------------------------------------------------------------
% 2. COSTRUZIONE MATRICE DI ADDESTRAMENTO (Train Set)
% -------------------------------------------------------------------------
% A. Componenti Spline Non Lineare (Termodinamica)
T_f_train = max(0, knot_temp - x_train); % Lineare Inverno
T_c_train = max(0, x_train - knot_temp); % Lineare Estate
T_f2_train = T_f_train.^2;               % Quadratica Inverno (Arco)
T_c2_train = T_c_train.^2;               % Quadratica Estate (Arco)
X_temp_train = [T_f_train, T_c_train, T_f2_train, T_c2_train];

% B. Componenti Armoniche (Routine Umana)
X_time_train = [];
for j = 1:k_opt
    X_time_train = [X_time_train, sin(2*pi*j*hours_train/24), cos(2*pi*j*hours_train/24)];
end

% C. Interazioni (Moltiplico i rami termici per TUTTE le armoniche)
X_int_train = [];
for c = 1:size(X_time_train, 2)
    X_int_train = [X_int_train, T_f_train .* X_time_train(:, c), T_c_train .* X_time_train(:, c)];
end

% Matrice Master Finale
X_train_master = [X_temp_train, X_time_train, X_int_train];

% Addestramento
master_model = fitlm(X_train_master, y_train);

% -------------------------------------------------------------------------
% 3. VALIDAZIONE SUL Validation Set (Dati Futuri)
% -------------------------------------------------------------------------
T_f_test = max(0, knot_temp - x_test);
T_c_test = max(0, x_test - knot_temp);
T_f2_test = T_f_test.^2;
T_c2_test = T_c_test.^2;
X_temp_test = [T_f_test, T_c_test, T_f2_test, T_c2_test];

X_time_test = [];
for j = 1:k_opt
    X_time_test = [X_time_test, sin(2*pi*j*hours_test/24), cos(2*pi*j*hours_test/24)];
end

X_int_test = [];
for c = 1:size(X_time_test, 2)
    X_int_test = [X_int_test, T_f_test .* X_time_test(:, c), T_c_test .* X_time_test(:, c)];
end

X_test_master = [X_temp_test, X_time_test, X_int_test];
y_pred_master = predict(master_model, X_test_master);
RMSE_master = sqrt(mean((y_test - y_pred_master).^2));

fprintf('\n=> PERFORMANCE MODELLO FINALE:\n');
fprintf('RMSE Master Model (Validation Set empirico): %.2f\n', RMSE_master);
disp('===============================================================');

% -------------------------------------------------------------------------
% --- GRAFICO 3D: La Superficie del Master Model
% -------------------------------------------------------------------------
disp('Generazione Superficie 3D Definitiva...');

% Addestro il modello su tutti i dati per un grafico completo
T_f_valid = max(0, knot_temp - x_valid);
T_c_valid = max(0, x_valid - knot_temp);
T_f2_valid = T_f_valid.^2;
T_c2_valid = T_c_valid.^2;
X_temp_valid = [T_f_valid, T_c_valid, T_f2_valid, T_c2_valid];

X_time_valid = [];
for j = 1:k_opt
    X_time_valid = [X_time_valid, sin(2*pi*j*hours_valid/24), cos(2*pi*j*hours_valid/24)];
end

X_int_valid = [];
for c = 1:size(X_time_valid, 2)
    X_int_valid = [X_int_valid, T_f_valid .* X_time_valid(:, c), T_c_valid .* X_time_valid(:, c)];
end

X_valid_master = [X_temp_valid, X_time_valid, X_int_valid];
definitive_master = fitlm(X_valid_master, y_valid);

% --- PLOT 3D ---
figure('Name', 'Master Model 3D: Spline Non Lineare + Armoniche Ottime', 'NumberTitle', 'off', 'Position', [100, 100, 900, 700]);
% Nuvola di punti reale (Uso hours_valid per avere l'asse X coerente da 0 a 24)
scatter3(hours_valid, x_valid, y_valid, 10, y_valid, 'filled', 'MarkerFaceAlpha', 0.2);
colormap(turbo); hold on;

% Creazione della Griglia 3D Pulita (Asse tempo da 0 a 24, Asse Temp min-max)
t_grid = linspace(0, 24, 80)';
x_grid = linspace(min(x_valid), max(x_valid), 80)';
[T_mesh, X_mesh] = meshgrid(t_grid, x_grid);
T_col = T_mesh(:); 
X_col = X_mesh(:); 

% Costruzione feature per la griglia
Grid_f = max(0, knot_temp - X_col);
Grid_c  = max(0, X_col - knot_temp);
Grid_f2 = Grid_f.^2;
Grid_c2 = Grid_c.^2;
Grid_temp = [Grid_f, Grid_c, Grid_f2, Grid_c2];

Grid_time = [];
for j = 1:k_opt
    Grid_time = [Grid_time, sin(2*pi*j*T_col/24), cos(2*pi*j*T_col/24)];
end

Grid_int = [];
for c = 1:size(Grid_time, 2)
    Grid_int = [Grid_int, Grid_f .* Grid_time(:, c), Grid_c .* Grid_time(:, c)];
end

Grid_Master = [Grid_temp, Grid_time, Grid_int];

% Previsione e disegno della superficie
Y_col_pred = predict(definitive_master, Grid_Master);
Y_mesh = reshape(Y_col_pred, size(T_mesh));

surf(T_mesh, X_mesh, Y_mesh, 'FaceAlpha', 0.85, 'EdgeColor', 'none');

% Formattazione del grafico
xlabel('Ora del Giorno (0 - 24)', 'FontWeight', 'bold');
ylabel('Temperatura Media (°F)', 'FontWeight', 'bold');
zlabel('Carico Elettrico (LOAD)', 'FontWeight', 'bold');
title(sprintf('Superficie del Master Model Definitivo (RMSE Test: %.2f)', RMSE_master), 'FontSize', 14);
view(-50, 30); 
grid on; hold off;

% ---> ESPORTAZIONE GRAFICO 1: SUPERFICIE MASTER MODEL <---
exportgraphics(gcf, 'immagini/punto7_superficie_master_model.png', 'Resolution', 300);


%% 7.1. Rete Neurale
% =========================================================================
% --- PUNTO 7.1: CONFRONTO CON RETE NEURALE ARTIFICIALE (ANN) ---
% =========================================================================
disp(' ');
disp('--- PUNTO 7.1: ADDESTRAMENTO RETE NEURALE (DEEP LEARNING) ---');

% MODIFICA METODOLOGICA: Codifica Ciclica del Tempo
% Per evitare che la rete debba "imparare" il salto da 23:59 a 00:00,
% trasformo l'ora lineare in coordinate circolari (seno e coseno).
sin_hours_train = sin(2*pi*hours_train/24);
cos_hours_train = cos(2*pi*hours_train/24);

sin_hours_test = sin(2*pi*hours_test/24);
cos_hours_test = cos(2*pi*hours_test/24);

% Le reti neurali in MATLAB (fitnet) preferiscono avere i dati organizzati
% con le feature sulle RIGHE e le osservazioni sulle COLONNE.
% Input Rete: [Temperatura, Seno dell'Ora, Coseno dell'Ora]
X_nn_train = [x_train, sin_hours_train, cos_hours_train]'; 
Y_nn_train = y_train';             
X_nn_test = [x_test, sin_hours_test, cos_hours_test]';
Y_nn_test = y_test';

% 1. ARCHITETTURA DELLA RETE
% Creo una rete Feed-Forward con 1 livello nascosto da 15 neuroni
num_neurons = 15;
net = fitnet(num_neurons);
net.trainParam.showWindow = false; % Rimuovo la finestra grafica

disp('Addestramento della Rete Neurale in corso (può richiedere qualche secondo)...');

% 2. ADDESTRAMENTO
[net, tr] = train(net, X_nn_train, Y_nn_train);

% 3. PREVISIONE E CALCOLO ERRORE SUL Validation Set
Y_nn_pred = net(X_nn_test);
RMSE_nn = sqrt(mean((Y_nn_test - Y_nn_pred).^2));

fprintf('\n=> PERFORMANCE RETE NEURALE (Black Box):\n');
fprintf('RMSE Neural Network (Validation Set empirico): %.2f\n', RMSE_nn);
fprintf('RMSE Master Model (Punto 7): %.2f\n', RMSE_master); 

% -------------------------------------------------------------------------
% --- GRAFICO 4: SUPERFICIE 3D DELLA RETE NEURALE ---
% -------------------------------------------------------------------------
disp('Generazione Superficie 3D della Rete Neurale...');
figure('Name', 'Neural Network 3D Surface', 'NumberTitle', 'off', 'Position', [150, 150, 900, 700]);

% Metto lo scatter reale per riferimento
scatter3(hours_valid, x_valid, y_valid, 10, y_valid, 'filled', 'MarkerFaceAlpha', 0.2);
colormap(turbo); hold on;

% Modifica per la Griglia: Convertiamo anche il T_mesh in formato trigonometrico
sin_T_mesh = sin(2*pi*T_mesh(:)/24);
cos_T_mesh = cos(2*pi*T_mesh(:)/24);

% Costruisco l'input a 3 variabili per la rete usando la griglia 2D originale
Grid_NN_input = [X_mesh(:), sin_T_mesh, cos_T_mesh]'; 

% Previsione della rete sull'intera griglia
Y_NN_pred_grid = net(Grid_NN_input);

% Rimodello l'output per la funzione surf
Y_NN_mesh = reshape(Y_NN_pred_grid, size(T_mesh));

surf(T_mesh, X_mesh, Y_NN_mesh, 'FaceAlpha', 0.85, 'EdgeColor', 'none');
xlabel('Ora del Giorno (0 - 24)', 'FontWeight', 'bold');
ylabel('Temperatura Media (°F)', 'FontWeight', 'bold');
zlabel('Carico Elettrico (LOAD)', 'FontWeight', 'bold');
title(sprintf('Superficie Rete Neurale (Input Armonico) | RMSE Test: %.2f', RMSE_nn), 'FontSize', 14);
view(-50, 30); 
grid on; hold off;

% ---> ESPORTAZIONE GRAFICO 2: SUPERFICIE RETE NEURALE <---
exportgraphics(gcf, 'immagini/punto7_superficie_rete_neurale.png', 'Resolution', 300);
