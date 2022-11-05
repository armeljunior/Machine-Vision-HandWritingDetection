close all

l = importdata('misalignment.mat');
n = importdata('resonance.mat');
bearing = importdata('bearing.mat');
k = importdata('gearmesh.mat');
z = importdata('imbalance.mat');

load('misalignment.mat')
load('multifault.mat')
load('resonance.mat')
load('bearing.mat')
load('gearmesh.mat')
load('imbalance.mat')

Fault1 = bearing;
Fault2 = gearmesh;
Fault3 = resonance;
Fault4 = imbalance;
Fault5 = misalignment;

t = 0:49999;

figure(1);
plot(t,bearing);
title('bearing fault')
%figure(2);

figure(6)
pwelch(bearing,[],[],[],1000)
title('Welch power spectral devectorlengthity estimate for bearing fault') 
figure(7)
pwelch(k,[],[],[],1000)
title('Welch power spectral devectorlengthity estimate for gearmesh fault') 
figure(8)
pwelch(l,[],[],[],1000)
title('Welch power spectral devectorlengthity estimate for misalignment fault') 
figure(9)
pwelch(z,[],[],[],1000)
title('Welch power spectral devectorlengthity estimate for imbalance fault') 
figure(10)
pwelch(n,[],[],[],1000)
title('Welch power spectral devectorlengthity estimate for resonance fault')

close all
%%
vectorlength = 1000;
js = length(bearing)/vectorlength;

x_normalxed = zeros(js,vectorlength);
f1s = zeros(1,js); %fault 1

for x=1:js %seperating each row and working
blocks = bearing((x-1)*vectorlength+1:x*vectorlength);

x_normalxed(x,:) = blocks - (mean(blocks)*ones(vectorlength,1));

[p1s,f1s]= pwelch(x_normalxed,[],[],[],1000);%1000 x the sampling frequency (Hz)
fs1(x) = (norm(p1s))/sqrt(vectorlength); %psd of normalxed

[B1s,A1s]= butter(11,0.1);
f2s_fil= filter(B1s,A1s,x_normalxed(x,:)); % for x1 data vector

[p2s,fs2]= pwelch(f2s_fil,[],[],[],1000);%1000 x the sampling frequency (Hz)
fs2(x) = (norm(p2s))/sqrt(vectorlength); %psd of normalxed

[B2s,A2s]= butter(13,[0.1 0.4]);
f3s_fil= filter(B2s,A2s,x_normalxed(x,:)); % for x1 data vector

[p3s,f3s]= pwelch(f3s_fil,[],[],[],1000);%1000 x the sampling frequency (Hz)
fs3(x) = (norm(p3s))/sqrt(vectorlength); %psd of normalxed

[B3s,A3s] = butter(18,0.4,'high');
f4s_fil= filter(B3s,A3s,x_normalxed(x,:)); % for x1 data vector

[p4s,f4s]= pwelch(f4s_fil,[],[],[],1000);%1000 x the sampling frequency (Hz
fs4(x) = (norm(p4s))/sqrt(vectorlength); %psd of normalxed

end
subplot(2,2,1)
plot(fs1)
subplot(2,2,2)
plot(fs2)
subplot(2,2,3)
plot(fs3)
subplot(2,2,4)
plot(fs4)

G = [Fault1;Fault2;Fault3;Fault4;Fault5];

c = corrcoef(G);

%fault1 = (feature_bearing)';


pwelch(fs1,20);
% 
% a = [Fault1;Fault2;Fault3;Fault4;Fault5];
% %G = [fault1;Fault2;Fault3];
% c=corrcoef(a); % Calculates a correlation coefficient matrix c of G 
% 
% [v,d] = eig(c); % Find the eigenvectors v and the eigenvalues d of G 
% T=[ v(:,end)';v(:,end-1)']; % Create the travectorlengthformation matrix T from
%                            % the first two principal components
% zo=T*G'; % Create a 2-dimevectorlengthional feature vector z
% figure(1)
% plot(zo(1,:),zo(2,:),'o') % Scatter plot of the 2-dimevectorlengthional features
% figure(2)
% plot(zo(1,1:50),zo(2,1:50),'b--o'); hold on;
% plot(zo(1,51:100),zo(2,51:100),'*'); hold on;
% plot(zo(1,101:150),zo(2,101:150),'^'); hold on;
% plot(zo(1,151:200),zo(2,151:200),'c*'); hold on;
% plot(zo(1,201:250),zo(2,201:250),'b--o'); hold on;
% 

