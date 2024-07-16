clc
clear
%% ����ʾ�����ݡ�
%chickenpox_dataset ����һ��ʱ����ʱ�䲽��Ӧ���·ݣ�ֵ��Ӧ�ڲ�������
%�����һ��Ԫ�����飬����ÿ��Ԫ�ؾ�Ϊ��һʱ�䲽���������ع�Ϊ��������
data = chickenpox_dataset;
data = [data{:}];

figure
plot(data)
xlabel('Month')
ylabel('Cases')
title('Monthly Cases of Chickenpox')

%% ��ѵ�����ݺͲ������ݽ��з�����
%���е�ǰ 90% ����ѵ������ 10% ���ڲ��ԡ�
numTimeStepsTrain = floor(0.9*numel(data));
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain:end);

%% ��׼������
%Ϊ�˻�ýϺõ���ϲ���ֹѵ����ɢ����ѵ�����ݱ�׼��Ϊ�������ֵ�͵�λ���
%��Ԥ��ʱ��������ʹ����ѵ��������ͬ�Ĳ�������׼���������ݡ�
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;
% LSTM�������ݱ�׼��Ҫ��ܸߡ�
% ������ֻ��ѵ�������б�׼����ԭ���Ǿ����������е�ֵֻ��ѵ�������������Բ��Լ����б�׼����

%% ׼��Ԥ���������Ӧ
%ҪԤ�������ڽ���ʱ�䲽��ֵ���뽫��Ӧָ��Ϊ��ֵ��λ��һ��ʱ�䲽��ѵ�����С�
%Ҳ����˵�����������е�ÿ��ʱ�䲽��LSTM ���綼ѧϰԤ����һ��ʱ�䲽��ֵ��
%Ԥ�������û������ʱ�䲽��ѵ�����С�
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

%% ���� LSTM ����ܹ�
%���� LSTM �ع����硣ָ�� LSTM ���� 200 ��������Ԫ
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];% %�����ع�����İ�������ģ�� ����˵���ⲻ���ڽ��з������⡣
%ָ��ѵ��ѡ�
%�����������Ϊ 'adam' ������ 250 ��ѵ����
%Ҫ��ֹ�ݶȱ�ը���뽫�ݶ���ֵ����Ϊ 1��
%ָ����ʼѧϰ�� 0.005���� 125 ��ѵ����ͨ���������� 0.2 ������ѧϰ�ʡ�
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...%ÿ������һ��������ʱ��ʱ��ѧϰ�ʾͻ����һ��ϵ����
    'LearnRateDropPeriod',125, ...���RMSE�����½�̫������LearnRateDropPeriod�Ĵ�
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ... %�����������Ϊtrue�����й�ѵ�����ȵ���Ϣ������ӡ��������С�Ĭ��ֵΪtrue��
    'Plots','training-progress');%��������ͼ ��'training-progress'�滻Ϊnone
%% ѵ�� LSTM ����
%ʹ�� trainNetwork ��ָ����ѵ��ѡ��ѵ�� LSTM ���硣
[net,info] = trainNetwork(XTrain,YTrain,layers,options);%info����ѵ����ʷ��Ϣ,loss+׼ȷ��


%LSTM������Ԥ�ⷽʽ��һ���Ǹ���Ԥ��ֵȥ��������״̬������һ���Ǹ��ݹ۲�ֵȥ��������״̬��
%ǰ�������ڵ�ά��ʱ������Ԥ�⣬�����ۻ���
%��������������������ȥ�����ĸ���ֵ���������Ԥ��ֵ��������������������Ľ�����£����Ԥ�⽫���׼ȷ��

%% Ԥ�⽫��ʱ�䲽
%ҪԤ�⽫�����ʱ�䲽��ֵ����ʹ�� predictAndUpdateState ����һ��Ԥ��һ��ʱ�䲽������ÿ��Ԥ��ʱ��������״̬������ÿ��Ԥ�⣬ʹ��ǰһ��Ԥ����Ϊ���������롣
%ʹ����ѵ��������ͬ�Ĳ�������׼���������ݡ�
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
%Ҫ��ʼ������״̬�����ȶ�ѵ������ XTrain ����Ԥ�⡣
%��������ʹ��ѵ����Ӧ�����һ��ʱ�䲽 YTrain(end) ���е�һ��Ԥ�⡣
%ѭ������Ԥ�Ⲣ��ǰһ��Ԥ�����뵽 predictAndUpdateState��
%���ڴ������ݼ��ϡ������л�������磬�� GPU �Ͻ���Ԥ�����ͨ������ CPU �Ͽ졣
%��������£��� CPU �Ͻ���Ԥ�����ͨ�����졣
%���ڵ�ʱ�䲽Ԥ�⣬��ʹ�� CPU��
%ʹ�� CPU ����Ԥ�⣬�뽫 predictAndUpdateState �� 'ExecutionEnvironment' ѡ������Ϊ 'cpu'��
net = predictAndUpdateState(net,XTrain);%���µ�XTrain�������������Ͻ��г�ʼ������״̬
[net,YPred1] = predictAndUpdateState(net,YTrain(end)); %��ѵ�������һ��������Ԥ���һ��Ԥ��ֵ������һ����ʼֵ��������Ԥ��ֵ��������״̬���еġ�

numTimeStepsTest = numel(XTest);

%% ����������֤�����������Ԥ�⣨��Ԥ��ֵ��������״̬��
for i = 2:numTimeStepsTest %�ӵڶ�����ʼ���������50�ε���Ԥ��(50Ϊ������֤��Ԥ��ֵ��0Ϊ����Ԥ���ֵ��һ��50����!
    [net,YPred1(:,i)] = predictAndUpdateState(net,YPred1(:,i-1),'ExecutionEnvironment','cpu');%predictAndUpdateState������һ��Ԥ��һ��ֵ����������״̬
end
%ʹ����ǰ����Ĳ�����Ԥ��ȥ��׼����
YPred1 = sig*YPred1 + mu;
%ѵ������ͼ�ᱨ����ݱ�׼�����ݼ�����ľ�������� (RMSE)������ȥ��׼����Ԥ��ֵ���� RMSE��
YTest = dataTest(2:end);
rmse1 = sqrt(mean((YPred1-YTest).^2))

%ʹ��Ԥ��ֵ����ѵ��ʱ��
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred1],'.-')
hold off
xlabel('Month')
ylabel('Cases')
title('Forecast')
legend('Observed','Forecast')

%��Ԥ��ֵ��������ݽ��бȽϡ�
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred1,'.-')
hold off
legend('Observed','Forecast')
ylabel('Cases')
title('Forecast')

subplot(2,1,2)
stem(YPred1 - YTest)
xlabel('Month')
ylabel('Error')
title(['RMSE = ' ,num2str(rmse1)] )  %title('RMSE = ' + rmse )

%% ʹ�ù۲�ֵ��������״̬
%��������Է���Ԥ��֮���ʱ�䲽��ʵ��ֵ�������ʹ�ù۲�ֵ������Ԥ��ֵ��������״̬��
%���ȣ���ʼ������״̬��Ҫ�������н���Ԥ�⣬��ʹ�� resetState ��������״̬��
%��������״̬�ɷ�ֹ��ǰ��Ԥ��Ӱ��������ݵ�Ԥ�⡣
% ��������״̬��Ȼ��ͨ����ѵ�����ݽ���Ԥ������ʼ������״̬��
net = resetState(net);
net = predictAndUpdateState(net,XTrain);
%��ÿ��ʱ�䲽����Ԥ�⡣����ÿ��Ԥ�⣬ʹ��ǰһʱ�䲽�Ĺ۲�ֵԤ����һ��ʱ�䲽��
%�� predictAndUpdateState �� 'ExecutionEnvironment' ѡ������Ϊ 'cpu'��
YPred2 = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred2(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end
%ʹ����ǰ����Ĳ�����Ԥ��ȥ��׼����
YPred2 = sig*YPred2 + mu;
%������������ (RMSE)��
rmse2 = sqrt(mean((YPred2-YTest).^2))
%��Ԥ��ֵ��������ݽ��бȽϡ�
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred2,'.-')
hold off
legend('Observed','Predicted')
ylabel('Cases')
title('Forecast with Updates')

subplot(2,1,2)
stem(YPred2 - YTest)
xlabel('Month')
ylabel('Error')
title(['RMSE = ' ,num2str(rmse2)] )

%����
% ������net����Ϊ.mat�ļ��������ֱ�ӵ���
save('net.mat','net');      
load('net.mat');     % ����֮ǰ���������

rec_step=10;    %����ֻԤ��10��
[net,Y]= predictAndUpdateState(net,XTest(end));
for i = 2:rec_step
    [net,Y(:,i)] = predictAndUpdateState(net,Y(:,i-1),'ExecutionEnvironment','cpu');
    %��Y����ո���input_Train�����µ�����õ���һ��������õ���Ӧ��Ԥ��ֵ��
end
%ʹ����ǰ����Ĳ�����Ԥ��ȥ��׼����
y = sig*Y + mu;


