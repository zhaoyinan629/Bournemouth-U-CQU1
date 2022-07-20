clc; close all;clear all

fin=fopen('./out/pred_s.txt','r');
result={};
while feof(fin)==0
    str=fgetl(fin);
    str=strread(str,'%s','delimiter',' \n');
    result=[result;str];
end
data_double=str2double(result);
c = size(data_double,1)/92;
Rx_FDE1 = zeros(c,92);
for f =1:c
    for h = 1:92
        Rx_FDE1(f,h) =data_double(((f-1)*92+h),1);
    end
end

%% Parameter
NF = 256; % OFDM block size, i.e., size of IFFT/FFT or total number of subcarriers
Ndata = 92; % number of subcarriers used for data transmission
Nsym = 2000; % number of OFDM blocks

N = 4; % number of subcarriers in each subblock
K = 2; % K out of N subcarriers in each subblock are active

M1 = 4; % M1-QAM
M2 = 4; % M2-QAM

SNR = 10; % SNR (dB)

%% OFDM-DMIM transmitter
G = Ndata/N; % number of subblocks
Nc = nchoosek(N,K); % number of combinations
bi = floor(log2(Nc)); % number of bits carried by index selection
bc1 = K*log2(M1);
bc2 = (N-K)*log2(M2);
b = bi + bc1 + bc2; % number of bits transmitted per subblock
m = b*G; % number of bits transmitted per OFDM block

% Constellation symbol
% load('./data/b12')

c=1;
Bit_con_tx1 = floor(rand(1,bc1*G*Nsym)*2);
Bit_con_tx_para1 = vec2mat(Bit_con_tx1,log2(M1));
Decimal_con_tx1 = bi2de(Bit_con_tx_para1,'left-msb');

Bit_con_tx2 = floor(rand(1,bc2*G*Nsym)*2);
Bit_con_tx_para2 = vec2mat(Bit_con_tx2,log2(M2));
Decimal_con_tx2 = bi2de(Bit_con_tx_para2,'left-msb');

Con_tx1 = qammod(Decimal_con_tx1,M1,'gray') - 2;
Con_tx2 = qammod(Decimal_con_tx2,M2,'gray') + 2;
% scatterplot(Con_tx1)
% scatterplot(Con_tx2)
Con_tx_para1 = vec2mat(Con_tx1,K);
Con_tx_para2 = vec2mat(Con_tx2,N-K);

% Index symbol
Bit_index_tx = floor(rand(1,bi*G*Nsym)*2);
Bit_index_tx_para = vec2mat(Bit_index_tx,bi);

% Subblock creation
Index_map = [1 2 3 4;
             2 3 1 4;
             3 4 1 2;
             1 4 2 3];
Decimal_map = [0 1 3 2];         
         
Subblock_tx = zeros(G*Nsym,N);
Decimal_index_tx = bi2de(Bit_index_tx_para,'left-msb');

for i = 1:G*Nsym
    for j = 1:2^bi
        if Decimal_index_tx(i) == Decimal_map(j)
            Subblock_tx(i,Index_map(j,:)) = [Con_tx_para1(i,:),Con_tx_para2(i,:)];
        end
    end
end

% OFDM block creation
Data_tx = (reshape(Subblock_tx.',Ndata,Nsym)).';
Data_tx_conj = fliplr(conj(Data_tx));

% IFFT with HS
IFFT_input = zeros(Nsym,NF);
IFFT_input(:,2:Ndata+1) = Data_tx;
IFFT_input(:,NF-Ndata+1:NF) = Data_tx_conj;
IFFT_output = ifft(IFFT_input,NF,2); % oversample by lines

% S/P
elec_sig = reshape(IFFT_output',1,Nsym*NF);
% elec_sig_clip = reshape(IFFT_output_clip',1,Nsym*NF);

% Normalization electrical signal
elec_power = var(elec_sig);
Tx_signal = elec_sig/sqrt(elec_power);
% elec_power = var(elec_sig_clip);
% Tx_signal = elec_sig_clip/sqrt(elec_power);

%% AWGN channel
snr = 10^(SNR/10);
Pn_t = 1/snr;
noise = sqrt(Pn_t)*randn(1,Nsym*NF);
Rx_signal = Tx_signal + noise;

%% OFDM-DMIM receiver
% FFT
FFT_input = vec2mat(Rx_signal,NF);
FFT_output = fft(FFT_input,NF,2);
Rx = FFT_output(:,2:Ndata+1);

% Equalization
%Rx_FDE = Rx*sqrt(elec_power);
Rx_FDE = Rx_FDE1;
scatterplot(reshape(Rx_FDE,1,size(Rx_FDE,1)*size(Rx_FDE,2)));
% Rx_con = reshape(Rx_FDE,size(Rx_FDE,1)*size(Rx_FDE,2),1);
% X = real(Rx_con);
% Y = imag(Rx_con);
% colormap([0 0 0.8;0 0.9 0.9;0 1 0;1 1 0;1 0 0]) %desired colors
% minx = min(X,[],1);
% maxx = max(X,[],1);
% miny = min(Y,[],1);
% maxy = max(Y,[],1);
% nbins = [min(numel(unique(X)),200),min(numel(unique(Y)),200)];
% edges1 = linspace(minx, maxx, nbins(1)+1);
% ctrs1 = edges1(1:end-1) + .5*diff(edges1);
% edges1 = [-Inf edges1(2:end-1) Inf];
% edges2 = linspace(miny, maxy, nbins(2)+1);
% ctrs2 = edges2(1:end-1) + .5*diff(edges2);
% edges2 = [-Inf edges2(2:end-1) Inf];
% [n,p] = size(X);
% bin = zeros(n,2);
% [dum,bin(:,2)] = histc(X,edges1);
% [dum,bin(:,1)] = histc(Y,edges2);
% H = accumarray(bin,1,nbins([2 1]))./ n;
% F = H;
% F = F./max(F(:));
% ind = sub2ind(size(F),bin(:,1),bin(:,2));
% col = F(ind);
% msize = 15;
% marker = '.';
% scatter(X,Y,msize,col,marker);
% % axis([-3 3 -3 3]);
% box on
% set(gca,'FontName','Times New Roman','FontSize',16,'FontWeight','bold');
% set(gcf,'Color',[1,1,1]);
% set(gca,'linewidth',1.5);

% Subblock recovery
Subblock_rx = (reshape(Rx_FDE.',N,Nsym*G)).';

%% Maximum-likelihood (ML) detection: optimal 
Pn_f = Pn_t*2*G*N/NF;

% Constellation sets
% S1 = Con_set1;
% S2 = Con_set2;

Con_decimal_M1 = 0:(M1-1);
S1 = qammod(Con_decimal_M1,M1,'gray') - 2;
Con_decimal_M2 = 0:(M2-1);
S2 = qammod(Con_decimal_M2,M2,'gray') + 2;

% Calculate LLR value 
LLR_value = zeros(G*Nsym,N);
temp_S1 = zeros(G*Nsym,N);
temp_S2 = zeros(G*Nsym,N);

for g = 1:G*Nsym
    for i = 1:N
        for j = 1:M1
            temp_S1(g,i) = temp_S1(g,i) + ...
                     exp(- abs(Subblock_rx(g,i) - S1(j))^2/Pn_f);
        end
        
        for j = 1:M2
            temp_S2(g,i) = temp_S2(g,i) + ...
                     exp(- abs(Subblock_rx(g,i) - S2(j))^2/Pn_f);
        end
        
        LLR_value(g,i) = log(K/(N-K)) + log(temp_S1(g,i)) - log(temp_S2(g,i));
    end
end

% Find the index information
[V,I] = sort(LLR_value,2);
Index_rx_S1 = sort(I(:,3:4),2);
Index_rx_S2 = sort(I(:,1:2),2);
Index_rx = [Index_rx_S1 Index_rx_S2];

%% Index and constellaton extraction
Decimal_index_rx = zeros(G*Nsym,1);
Con_rx_para = zeros(G*Nsym,N);

for i = 1:G*Nsym
    for j = 1:2^bi
        if Index_rx(i,:) == Index_map(j,:)
            Decimal_index_rx(i) = Decimal_map(j);
            Con_rx_para(i,:) = Subblock_rx(i,Index_map(j,:));
        end
    end
end
Con_rx_para1 = Con_rx_para(:,1:K);
Con_rx_para2 = Con_rx_para(:,K+1:N);
Bit_index_rx_para = de2bi(Decimal_index_rx,'left-msb');

% Constellation demapping
Con_rx_LLR1 = reshape(Con_rx_para1.',1,Nsym*G*K).';
Con_rx_LLR2 = reshape(Con_rx_para2.',1,Nsym*G*(N-K)).';

Decimal_con_rx_LLR1 = (qamdemod(Con_rx_LLR1 + 2,M1,'gray'))';
Decimal_con_rx_LLR2 = (qamdemod(Con_rx_LLR2 - 2,M2,'gray'))';

Bit_con_rx_para_LLR1 = de2bi(Decimal_con_rx_LLR1,'left-msb');
Bit_con_rx_para_LLR2 = de2bi(Decimal_con_rx_LLR2,'left-msb');

% BER calculation
[Nerrorbit_index_LLR,BER_index_LLR] = biterr(Bit_index_tx_para,Bit_index_rx_para);
[Nerrorbit_con_LLR1,BER_con_LLR1] = biterr(Bit_con_tx_para1,Bit_con_rx_para_LLR1);
[Nerrorbit_con_LLR2,BER_con_LLR2] = biterr(Bit_con_tx_para2,Bit_con_rx_para_LLR2);

% Average BER
 BER = (Nerrorbit_index_LLR + Nerrorbit_con_LLR1 + Nerrorbit_con_LLR2)/Nsym/m;
