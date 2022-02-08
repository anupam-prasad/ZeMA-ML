load("Achse11_Szenario.mat","trainData","trainTarget",'validation');

trainInd = validation.training(1);
for i = 1:length(trainData)
    validationData{i} = trainData{i}(~trainInd,:);
end
temp = {};
for i = 1:length(trainData)
    temp{i} = trainData{i}(trainInd,:);
end
trainData = temp;
trainTarget = (trainTarget);
validationTarget = trainTarget(~trainInd);
trainTarget = trainTarget(trainInd);
save("axAllSensors.mat", "trainData","validationData","trainTarget","validationTarget",'-v7.3','-nocompression');