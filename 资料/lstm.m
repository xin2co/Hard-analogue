clc
clear
%% 加载示例数据。
%chickenpox_dataset 包含一个时序，其时间步对应于月份，值对应于病例数。
%输出是一个元胞数组，其中每个元素均为单一时间步。将数据重构为行向量。
data = chickenpox_dataset;
data = [data{:}];

figure
plot(data)
xlabel('Month')
ylabel('Cases')
title('Monthly Cases of Chickenpox')

%% 对训练数据和测试数据进行分区。
%序列的前 90% 用于训练，后 10% 用于测试。
numTimeStepsTrain = floor(0.9*numel(data));
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain:end);

%% 标准化数据
%为了获得较好的拟合并防止训练发散，将训练数据标准化为具有零均值和单位方差。
%在预测时，您必须使用与训练数据相同的参数来标准化测试数据。
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;
% LSTM对于数据标准化要求很高。
% 且这里只对训练集进行标准化的原因是经过神经网络中的值只有训练集，因此无须对测试集进行标准化。

%% 准备预测变量和响应
%要预测序列在将来时间步的值，请将响应指定为将值移位了一个时间步的训练序列。
%也就是说，在输入序列的每个时间步，LSTM 网络都学习预测下一个时间步的值。
%预测变量是没有最终时间步的训练序列。
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

%% 定义 LSTM 网络架构
%创建 LSTM 回归网络。指定 LSTM 层有 200 个隐含单元
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];% %其计算回归问题的半均方误差模块 。即说明这不是在进行分类问题。
%指定训练选项。
%将求解器设置为 'adam' 并进行 250 轮训练。
%要防止梯度爆炸，请将梯度阈值设置为 1。
%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率。
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...%每当经过一定数量的时期时，学习率就会乘以一个系数。
    'LearnRateDropPeriod',125, ...如果RMSE曲线下降太慢，将LearnRateDropPeriod改大。
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ... %如果将其设置为true，则有关训练进度的信息将被打印到命令窗口中。默认值为true。
    'Plots','training-progress');%构建曲线图 将'training-progress'替换为none
%% 训练 LSTM 网络
%使用 trainNetwork 以指定的训练选项训练 LSTM 网络。
[net,info] = trainNetwork(XTrain,YTrain,layers,options);%info包含训练历史信息,loss+准确率


%LSTM有两种预测方式，一种是根据预测值去更新网络状态，另外一种是根据观测值去更新网络状态。
%前者适用于单维的时间序列预测，将会累积误差；
%后者适用于有其他变量去将外界的干扰值反馈给输出预测值的情况，由于有外界变量的介入更新，因此预测将会更准确。

%% 预测将来时间步
%要预测将来多个时间步的值，请使用 predictAndUpdateState 函数一次预测一个时间步，并在每次预测时更新网络状态。对于每次预测，使用前一次预测作为函数的输入。
%使用与训练数据相同的参数来标准化测试数据。
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
%要初始化网络状态，请先对训练数据 XTrain 进行预测。
%接下来，使用训练响应的最后一个时间步 YTrain(end) 进行第一次预测。
%循环其余预测并将前一次预测输入到 predictAndUpdateState。
%对于大型数据集合、长序列或大型网络，在 GPU 上进行预测计算通常比在 CPU 上快。
%其他情况下，在 CPU 上进行预测计算通常更快。
%对于单时间步预测，请使用 CPU。
%使用 CPU 进行预测，请将 predictAndUpdateState 的 'ExecutionEnvironment' 选项设置为 'cpu'。
net = predictAndUpdateState(net,XTrain);%将新的XTrain数据用在网络上进行初始化网络状态
[net,YPred1] = predictAndUpdateState(net,YTrain(end)); %用训练的最后一步来进行预测第一个预测值，给定一个初始值。这是用预测值更新网络状态特有的。

numTimeStepsTest = numel(XTest);

%% 进行用于验证神经网络的数据预测（用预测值更新网络状态）
for i = 2:numTimeStepsTest %从第二步开始，这里进行50次单步预测(50为用于验证的预测值，0为往后预测的值。一共50个）!
    [net,YPred1(:,i)] = predictAndUpdateState(net,YPred1(:,i-1),'ExecutionEnvironment','cpu');%predictAndUpdateState函数是一次预测一个值并更新网络状态
end
%使用先前计算的参数对预测去标准化。
YPred1 = sig*YPred1 + mu;
%训练进度图会报告根据标准化数据计算出的均方根误差 (RMSE)。根据去标准化的预测值计算 RMSE。
YTest = dataTest(2:end);
rmse1 = sqrt(mean((YPred1-YTest).^2))

%使用预测值绘制训练时序。
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

%将预测值与测试数据进行比较。
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

%% 使用观测值更新网络状态
%如果您可以访问预测之间的时间步的实际值，则可以使用观测值而不是预测值更新网络状态。
%首先，初始化网络状态。要对新序列进行预测，请使用 resetState 重置网络状态。
%重置网络状态可防止先前的预测影响对新数据的预测。
% 重置网络状态，然后通过对训练数据进行预测来初始化网络状态。
net = resetState(net);
net = predictAndUpdateState(net,XTrain);
%对每个时间步进行预测。对于每次预测，使用前一时间步的观测值预测下一个时间步。
%将 predictAndUpdateState 的 'ExecutionEnvironment' 选项设置为 'cpu'。
YPred2 = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred2(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end
%使用先前计算的参数对预测去标准化。
YPred2 = sig*YPred2 + mu;
%计算均方根误差 (RMSE)。
rmse2 = sqrt(mean((YPred2-YTest).^2))
%将预测值与测试数据进行比较。
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

%调用
% 将网络net保存为.mat文件，后面可直接调用
save('net.mat','net');      
load('net.mat');     % 导入之前保存的网络

rec_step=10;    %这里只预测10步
[net,Y]= predictAndUpdateState(net,XTest(end));
for i = 2:rec_step
    [net,Y(:,i)] = predictAndUpdateState(net,Y(:,i-1),'ExecutionEnvironment','cpu');
    %用Y代入刚刚用input_Train来更新的网络得到第一个输出并得到对应的预测值。
end
%使用先前计算的参数对预测去标准化。
y = sig*Y + mu;


