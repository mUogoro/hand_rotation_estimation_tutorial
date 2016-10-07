#ifndef RCHORD_MEANSHIFT_HPP
#define RCHORD_MEANSHIFT_HPP

#include <vector>
#include <Eigen/Dense>


unsigned int  MultiGuessRChordMeanShift(const std::vector<Eigen::Vector3f> &votes,
					const std::vector<float> &votesWeight,
					const std::vector<Eigen::Vector3f> guesses,
					std::vector<Eigen::Vector3f> &modes,
					std::vector<int> &guessModeID,
					float b, float stepSize, float stepThr,
					unsigned int nIterThr, float sameModeThr);


void RChordMeanShift(const std::vector<Eigen::Vector3f> &votes,
		     const std::vector<float> &votesWeight,
		     const Eigen::Vector3f &guess, Eigen::Vector3f &mode,
		     float b, float stepSize, float stepThr, unsigned int nIterThr);

#endif // RMEANSHIFT_HPP
