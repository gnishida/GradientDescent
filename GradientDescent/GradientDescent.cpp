#include "GradientDescent.h"

/**
 * Run gradient descent
 *
 * @features		学習データ　リストの各elementは、２つのfeatureベクトルのpair。
 * @choices			ラベルのリスト　これを学習する。
 * @maxIterations	最大ステップ数
 * @l1				L1 generalization termを使用するならtrue、L2ならfalse
 * @lambda			generalization termの係数
 * @eta				学習速度
 * @threshold		収束しきい値
 * @return			推定されたpreference vector
 */
std::vector<float> GradientDescent::run(std::vector<std::pair<std::vector<float>, std::vector<float> > >& features, std::vector<int> choices, int maxIterations, bool l1, float lambda, float eta, float threshold) {
	std::vector<float> w;

	FILE* fp = fopen("gd_curve.txt", "w");

	int numFeatures = features[0].first.size();
	w.resize(numFeatures);
	for (int k = 0; k < numFeatures; ++k) {
		w[k] = 1.0f / numFeatures;
	}

	float curE = negativeLogLikelihood(features, choices, w, l1, lambda);
	for (int iter = 0; iter < maxIterations; ++iter) {
		fprintf(fp, "%lf\n", curE);

		std::vector<float> dw;
		dw.resize(numFeatures);
		for (int k = 0; k < numFeatures; ++k) {
			dw[k] = 0.0f;
		}

		for (int d = 0; d < features.size(); ++d) {
			float e = expf(dot(w, features[d].second) - dot(w, features[d].first));
			float a = choices[d] - 1.0f / (1.0f + e);
			
			for (int k = 0; k < numFeatures; ++k) {
				dw[k] += (features[d].second[k] - features[d].first[k]) * a;
			}
		}

		for (int k = 0; k < numFeatures; ++k) {
			if (l1) {
				if (w[k] >= 0) {
					w[k] -= eta * (lambda + dw[k]);
				} else {
					w[k] -= eta * (-lambda + dw[k]);
				}
			} else {
				w[k] -= eta * (lambda * w[k] + dw[k]);
			}
		}

		float nextE = negativeLogLikelihood(features, choices, w, l1, lambda);
		if (curE - nextE < threshold) break;

		curE = nextE;
	}

	fclose(fp);

	return w;
}

float GradientDescent::negativeLogLikelihood(std::vector<std::pair<std::vector<float>, std::vector<float> > >& features, std::vector<int> choices, std::vector<float> w, bool l1, float lambda) {
	int numFeatures = features[0].first.size();

	float E = 0.0f;
	for (int d = 0; d < features.size(); ++d) {
		float diff = dot(w, features[d].second) - dot(w, features[d].first);
		E += logf(1.0f + expf(diff)) + (choices[d] - 1.0f) * diff;
	}

	if (l1) {
		for (int k = 0; k < numFeatures; ++k) {
			E += fabs(w[k]) * lambda;
		}
	} else {
		E += dot(w, w) * lambda / 2.0f;
	}

	return E;
}

float GradientDescent::dot(std::vector<float> w, std::vector<float> f) {
	float ret = 0.0f;
	for (int i = 0; i < w.size(); ++i) {
		ret += w[i] * f[i];
	}

	return ret;
}