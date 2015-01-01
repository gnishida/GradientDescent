#pragma once

#include <vector>

class GradientDescent {
public:
	GradientDescent() {}

	std::vector<float> run(std::vector<std::pair<std::vector<float>, std::vector<float> > >& features, std::vector<int> choices, int maxIterations, float lambda, float eta, float threshold);

private:
	float negativeLogLikelihood(std::vector<std::pair<std::vector<float>, std::vector<float> > >& features, std::vector<int> choices, std::vector<float> w, float lambda);
	float dot(std::vector<float> w, std::vector<float> f);
};

