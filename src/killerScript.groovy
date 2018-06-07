

PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis()
pca.setup(4, 10)
double[] sample1 = [1,2,3,4,5,6,7,8,9,10]
double[] sample2 = [4,5,3,5,2,7,4,5,6,2]
double[] sample3 = [2,8,8,7,1,2,3,1,2,2]
double[] sample4 = [10,11,23,1,2,3,4,5,2,3]
pca.addSample(sample1)
pca.addSample(sample2)
pca.addSample(sample3)
pca.addSample(sample4)
pca.computeBasis(3)
println pca.sampleToEigenSpace(sample1)
//pca.eigenToSampleSpace()
println pca.errorMembership(sample1)
println pca.response(sample1)
