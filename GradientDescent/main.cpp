#include <vector>
#include "GradientDescent.h"

#define NUM_TASKS 20

float dot(std::vector<float> w, std::vector<float> f) {
	float ret = 0.0f;
	for (int i = 0; i < w.size(); ++i) {
		ret += w[i] * f[i];
	}

	return ret;
}

void normalize(std::vector<float>& w) {
	float total = 0.0f;
	for (int i = 0; i < w.size(); ++i) {
		total += w[i];
	}

	for (int i = 0; i < w.size(); ++i) {
		w[i] /= total;
	}
}

float randf() {
	return (float)(rand() % RAND_MAX) / RAND_MAX;
}

int main() {
	std::vector<float> w(8);
	w[0] = 0.1;
	w[1] = 0.05;
	w[2] = 0.15;
	w[3] = 0.02;
	w[4] = 0.37;
	w[5] = 0.01;
	w[6] = 0.05;
	w[7] = 0.25;

	std::vector<std::pair<std::vector<float>, std::vector<float> > > features;
	std::vector<int> choices;

	for (int iter = 0; iter < NUM_TASKS; ++iter) {
		std::vector<float> feature1(8);
		std::vector<float> feature2(8);
		for (int k = 0; k < 8; ++k) {
			feature1[k] = randf();
			feature2[k] = randf();
		}
		features.push_back(std::make_pair(feature1, feature2));

		float d1 = dot(w, feature1);
		float d2 = dot(w, feature2);

		if (d1 > d2) {
			choices.push_back(1);
		} else {
			choices.push_back(0);
		}
	}

	GradientDescent gd;
	std::vector<float> estimate_w = gd.run(features, choices, 10000, false, 0.0, 0.001, 0.001);

	normalize(estimate_w);

	printf("Result:\n");
	for (int k = 0; k < 8; ++k) {
		printf("%lf (%lf)\n", estimate_w[k], w[k]);
	}

	int correct = 0;
	int incorrect = 0;
	for (int d = 0; d < NUM_TASKS; ++d) {
		int y = dot(w, features[d].first) > dot(w, features[d].second) ? 1 : 0;
		int h = dot(estimate_w, features[d].first) > dot(estimate_w, features[d].second) ? 1 : 0;
		if (h == choices[d]) {
			printf("%d: OK\n", d);
			correct++;
		} else {
			printf("%d: NG\n", d);
			incorrect++;
		}
	}
	printf("correct: %d / %d\n", correct, correct + incorrect);

	return 0;
}