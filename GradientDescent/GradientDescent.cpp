#include "GradientDescent.h"

std::vector<float> GradientDescent::run(std::vector<std::pair<std::vector<float>, std::vector<float> > >& features, std::vector<int> choices, int maxIterations, float lambda, float eta, float threshold) {
	std::vector<float> w;

	FILE* fp = fopen("gd_curve.txt", "w");

	int numFeatures = features[0].first.size();
	w.resize(numFeatures);
	for (int k = 0; k < numFeatures; ++k) {
		w[k] = 1.0f / numFeatures;
	}

	float curE = negativeLogLikelihood(features, choices, w, lambda);
	for (int iter = 0; iter < maxIterations; ++iter) {
		fprintf(fp, "%lf\n", curE);

		std::vector<float> dw;
		dw.resize(numFeatures);
		for (int k = 0; k < numFeatures; ++k) {
			dw[k] = 0.0f;
		}

		for (int d = 0; d < features.size(); ++d) {
			float e = expf(dot(w, features[d].first) - dot(w, features[d].second));
			float a = (e / (1.0f + e) + choices[d] - 1);
			
			for (int k = 0; k < numFeatures; ++k) {
				dw[k] += (features[d].second[k] - features[d].first[k]) * a;
			}
		}

		for (int k = 0; k < numFeatures; ++k) {
			w[k] -= eta * (lambda * w[k] + dw[k]);
		}

		float nextE = negativeLogLikelihood(features, choices, w, lambda);
		if (curE - nextE < threshold) break;

		curE = nextE;
	}

	fclose(fp);

	return w;
}

float GradientDescent::negativeLogLikelihood(std::vector<std::pair<std::vector<float>, std::vector<float> > >& features, std::vector<int> choices, std::vector<float> w, float lambda) {
	int numFeatures = features[0].first.size();

	float E = 0.0f;
	for (int d = 0; d < features.size(); ++d) {
		float diff = dot(w, features[d].second) - dot(w, features[d].first);
		E += logf(1.0f + expf(diff)) + (choices[d] - 1.0f) * diff;
	}

	E += dot(w, w) * lambda / 2.0f;

	return E;
}

float GradientDescent::dot(std::vector<float> w, std::vector<float> f) {
	float ret = 0.0f;
	for (int i = 0; i < w.size(); ++i) {
		ret += w[i] * f[i];
	}

	return ret;
}