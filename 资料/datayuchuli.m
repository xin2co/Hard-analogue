% 判断缺失值和异常值并修复，顺便光滑噪音，渡边笔记
clc,clear;close all;
x = 0:0.06:10;
y = sin(x)+0.2*rand(size(x));
y(22:34) = NaN;
y(89:95) = 50;
testdata = [x' y'];

subplot(2,2,1);
plot(testdata(:,1),testdata(:,2));
title('原始数据');

%% 判断数据中是否存在缺失值
if sum(isnan(testdata(:)))
    disp('数据存在缺失值');
else
    disp('数据不存在缺失值');
end

% 判断数据中是否存在异常值
% 1.mean 三倍标准差法 2.median 离群值法 3.quartiles 非正态的离群值法 
% 4.grubbs 正态的离群值法 5.gesd 多离群值相互掩盖的离群值法
choice_1 = 5;
yichangzhi_fa = char('mean', 'median', 'quartiles', 'grubbs','gesd');
yi_chang = isoutlier(y,strtrim(yichangzhi_fa(choice_1,:)));
if sum(yi_chang)
    disp('数据存在异常值');
else
    disp('数据不存在异常值');
end

%% 对异常值赋空值
F = find(yi_chang == 1);
y(F) = NaN; % 令数据点缺失
testdata = [x' y'];

subplot(2,2,2);
plot(testdata(:,1),testdata(:,2));
title('去除差异值');

%% 对数据进行补全
% 数据补全方法选择
% 1.线性插值 linear 2.分段三次样条插值 spline 3.保形分段三次样条插值 pchip
% 4.移动滑窗插补 movmean
chazhi_fa = char('linear', 'spline', 'pchip', 'movmean');
choice_2 = 3;
if choice_2 ~= 4
    testdata_1 = fillmissing(testdata,strtrim(chazhi_fa(choice_2,:))); % strtrim 是为了去除字符串组的空格
else
    testdata_1 = fillmissing(testdata,'movmean',5); % 窗口长度为 10 的移动均值
end

subplot(2,2,3);
plot(testdata_1(:,1),testdata_1(:,2));
title('数据补全结果');

%% 进行数据平滑处理
% 滤波器选择 1.Savitzky-golay 2.rlowess 3.rloess
choice_3 = 2;
lvboqi = char('Savitzky-golay', 'rlowess', 'pchip', 'rloess');
% 通过求 n 元素移动窗口的中位数，来对数据进行平滑处理
windows = 8;
testdata_2 = smoothdata(testdata_1(:,2),strtrim(lvboqi(choice_3,:)),windows) ;

subplot(2,2,4);
plot(x,testdata_2)
title('数据平滑结果');
