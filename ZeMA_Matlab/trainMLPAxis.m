run("nnScenarioMakerAxis.mlx")
%%
[trainData4D,trainTarget, lgraph, options] = prepareTrainNet();
%%
trainedNet = trainNetwork(trainData4D,trainTarget,lgraph,options);