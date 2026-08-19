function Uygulama()
    if ~isfile('Kusursuz_AI_Model.mat')
        errordlg('Critical Error: Kusursuz_AI_Model.mat not found. Please train and save your network first.', 'System Error');
        return;
    end
    
    data = load('Kusursuz_AI_Model.mat');
    net = data.net;
    
    fig = uifigure('Name', 'RF Filter AI Synthesizer - Extended Range Edition', 'Position', [100, 100, 1200, 750], 'Color', [0.1 0.1 0.1]);
    tabGroup = uitabgroup(fig, 'Position', [0, 0, 1200, 750]);
    
    enTab = uitab(tabGroup, 'Title', 'English Interface');
    enTab.BackgroundColor = [0.15 0.15 0.15];
    
    trTab = uitab(tabGroup, 'Title', 'Türkçe Arayüz');
    trTab.BackgroundColor = [0.15 0.15 0.15];
    
    createInterface(enTab, net, 'EN');
    createInterface(trTab, net, 'TR');
end

function createInterface(parent, net, lang)
    if strcmp(lang, 'EN')
        txt = struct('inputTitle', 'Physical Filter Target', 'fcLabel', 'Center Frequency (fc) [4.5-8.5 GHz]:', ...
                     'bwLabel', 'Bandwidth (BW) [0.1-2.0 GHz]:', 'noiseLabel', 'Environmental Noise Factor:', ...
                     'btn', ' RUN INFERENCE & SIMULATE ', 'outputTitle', '1D-CNN AI Analysis Results', ...
                     'predFc', 'AI Prediction (fc): Pending...', 'predBw', 'AI Prediction (BW): Pending...', ...
                     'errFc', 'Error (fc): % -', 'errBw', 'Error (BW): % -', 'magTitle', 'Channel 1: Magnitude (S21 dB)', ...
                     'phaseTitle', 'Channel 2: Unwrapped Phase (Rad)');
    else
        txt = struct('inputTitle', 'Fiziksel Filtre Hedefi', 'fcLabel', 'Merkez Frekansı (fc) [4.5-8.5 GHz]:', ...
                     'bwLabel', 'Bant Genişliği (BW) [0.1-2.0 GHz]:', 'noiseLabel', 'Çevresel Gürültü Çarpanı:', ...
                     'btn', ' TAHMİN YÜRÜT VE SİMÜLE ET ', 'outputTitle', '1D-CNN YZ Analiz Sonuçları', ...
                     'predFc', 'YZ Tahmini (fc): Bekleniyor...', 'predBw', 'YZ Tahmini (BW): Bekleniyor...', ...
                     'errFc', 'Sapma (fc): % -', 'errBw', 'Sapma (BW): % -', 'magTitle', 'Kanal 1: Genlik (S21 dB)', ...
                     'phaseTitle', 'Kanal 2: Faz (Unwrapped)');
    end

    axMag = uiaxes(parent, 'Position', [400, 400, 750, 280], 'Color', [0.2 0.2 0.2], 'XColor', 'w', 'YColor', 'w');
    title(axMag, txt.magTitle, 'Color', 'w'); grid(axMag, 'on');
    
    axPhase = uiaxes(parent, 'Position', [400, 60, 750, 280], 'Color', [0.2 0.2 0.2], 'XColor', 'w', 'YColor', 'w');
    title(axPhase, txt.phaseTitle, 'Color', 'w'); grid(axPhase, 'on');

    pnlIn = uipanel(parent, 'Title', txt.inputTitle, 'Position', [20, 450, 350, 230], 'BackgroundColor', [0.2 0.2 0.2], 'ForegroundColor', 'w');
    uilabel(pnlIn, 'Position', [15, 160, 230, 22], 'Text', txt.fcLabel, 'FontColor', 'w');
    spnFc = uispinner(pnlIn, 'Position', [250, 160, 80, 22], 'Limits', [4.5 8.5], 'Value', 6.5, 'Step', 0.001);
    uilabel(pnlIn, 'Position', [15, 110, 230, 22], 'Text', txt.bwLabel, 'FontColor', 'w');
    spnBw = uispinner(pnlIn, 'Position', [250, 110, 80, 22], 'Limits', [0.1 2.0], 'Value', 0.75, 'Step', 0.001);
    uilabel(pnlIn, 'Position', [15, 60, 200, 22], 'Text', txt.noiseLabel, 'FontColor', 'w');
    sldNoise = uislider(pnlIn, 'Position', [40, 45, 270, 3], 'Limits', [0 1], 'Value', 0.3);

    btn = uibutton(parent, 'push', 'Position', [20, 360, 350, 60], 'Text', txt.btn, 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', [0.7 0.1 0.1], 'FontColor', 'w');

    pnlOut = uipanel(parent, 'Title', txt.outputTitle, 'Position', [20, 60, 350, 270], 'BackgroundColor', [0.1 0.2 0.1], 'ForegroundColor', 'w');
    resFc = uilabel(pnlOut, 'Position', [15, 190, 320, 25], 'Text', txt.predFc, 'FontColor', 'g', 'FontSize', 15, 'FontWeight', 'bold');
    resBw = uilabel(pnlOut, 'Position', [15, 140, 320, 25], 'Text', txt.predBw, 'FontColor', 'g', 'FontSize', 15, 'FontWeight', 'bold');
    eFc = uilabel(pnlOut, 'Position', [15, 80, 320, 22], 'Text', txt.errFc, 'FontColor', 'w', 'FontSize', 13);
    eBw = uilabel(pnlOut, 'Position', [15, 40, 320, 22], 'Text', txt.errBw, 'FontColor', 'w', 'FontSize', 13);

    btn.ButtonPushedFcn = @(src, event) executeInference(spnFc, spnBw, sldNoise, axMag, axPhase, net, resFc, resBw, eFc, eBw, lang);
end

function executeInference(spnFc, spnBw, sldNoise, axMag, axPhase, net, resFc, resBw, eFc, eBw, lang)
    fcVal = spnFc.Value;
    bwVal = spnBw.Value;
    noiseLvl = sldNoise.Value;
    
    pts = 1001;
    % Frequency axis must match training exactly! [2 GHz - 11 GHz]
    f = linspace(2e9, 11e9, pts); 
    nFreq = (f/1e9 - fcVal) ./ bwVal;
    resp = 1 ./ (1 + 1i * nFreq).^5;
    
    mag = 20 * log10(abs(resp) + 1e-6) + (noiseLvl * 0.3) * randn(1, pts);
    phs = unwrap(angle(resp)) + (noiseLvl * 0.02) * randn(1, pts);
    
    plot(axMag, f/1e9, mag, 'c', 'LineWidth', 1.5); grid(axMag, 'on');
    plot(axPhase, f/1e9, phs, 'm', 'LineWidth', 1.5); grid(axPhase, 'on');
    
    inTensor = zeros(pts, 1, 2, 1);
    inTensor(:, 1, 1, 1) = mag';
    inTensor(:, 1, 2, 1) = phs';
    
    outNorm = predict(net, inTensor);
    
    % Updated De-Normalization logic
    pFc = outNorm(1) * 4.0 + 4.5;
    pBw = outNorm(2) * 1.9 + 0.1;
    
    errF = abs(fcVal - pFc) / fcVal * 100;
    errB = abs(bwVal - pBw) / bwVal * 100;
    
    if strcmp(lang, 'EN')
        resFc.Text = sprintf('AI Prediction (fc): %.4f GHz', pFc);
        resBw.Text = sprintf('AI Prediction (BW): %.4f GHz', pBw);
        eFc.Text = sprintf('Error (fc): %%%.4f', errF);
        eBw.Text = sprintf('Error (BW): %%%.4f', errB);
    else
        resFc.Text = sprintf('YZ Tahmini (fc): %.4f GHz', pFc);
        resBw.Text = sprintf('YZ Tahmini (BW): %.4f GHz', pBw);
        eFc.Text = sprintf('Sapma (fc): %%%.4f', errF);
        eBw.Text = sprintf('Sapma (BW): %%%.4f', errB);
    end
    
    if errF < 5, eFc.FontColor = [0 1 0]; else, eFc.FontColor = [1 1 0]; end
    if errB < 5, eBw.FontColor = [0 1 0]; else, eBw.FontColor = [1 1 0]; end
end