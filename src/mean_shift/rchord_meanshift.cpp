#include <cmath>
#include <algorithm>
#include <numeric>
#include <boost/unordered_map.hpp>
#include <emmintrin.h>
#include <immintrin.h>
#include "rchord_meanshift.hpp"

#include <iostream>

static void _meanShiftIter(const std::vector<Eigen::Matrix3f> &votes,
			   const std::vector<float> &weights,
			   const Eigen::Matrix3f &guess, Eigen::Matrix3f &mode,  float b);
static void _RChordMeanShift(const std::vector<Eigen::Matrix3f> &votes,
			     const std::vector<float> &votesWeight,
			     const Eigen::Matrix3f &guess, Eigen::Matrix3f &mode,
			     float b, float stepSize, float stepThr, unsigned int nIterThr);
static inline void _cvtVEuler2Rotm(const std::vector<Eigen::Vector3f> &votesEul,
				   std::vector<Eigen::Matrix3f> &votesRotm);
static inline void _cvtVRotm2Euler(const std::vector<Eigen::Matrix3f> &votesRotm,
				   std::vector<Eigen::Vector3f> &votesEul);
static inline void _cvtEuler2Rotm(const Eigen::Vector3f &voteEul, Eigen::Matrix3f &voteRotm);
static inline void _cvtRotm2Euler(const Eigen::Matrix3f &voteRotm, Eigen::Vector3f &voteEul);
static inline float _rotmDist2(const Eigen::Matrix3f &m1, const Eigen::Matrix3f &m2);



void RChordMeanShift(const std::vector<Eigen::Vector3f> &votes,
		     const std::vector<float> &votesWeight,
		     const Eigen::Vector3f &guess, Eigen::Vector3f &mode,
		     float b, float stepSize, float stepThr, unsigned int nIterThr)
{
  std::vector<Eigen::Matrix3f> votesRotm(votes.size());
  Eigen::Matrix3f guessRotm, modeRotm;

  _cvtVEuler2Rotm(votes, votesRotm);
  _cvtEuler2Rotm(guess, guessRotm);

  _RChordMeanShift(votesRotm, votesWeight, guessRotm, modeRotm,
		   b, stepSize, stepThr, nIterThr);

  _cvtRotm2Euler(modeRotm, mode);
}



static void _RChordMeanShift(const std::vector<Eigen::Matrix3f> &votes,
			     const std::vector<float> &votesWeight,
			     const Eigen::Matrix3f &guess, Eigen::Matrix3f &mode,
			     float b, float stepSize, float stepThr, unsigned int nIterThr)
{
  Eigen::Matrix3f currGuess = guess;
  
  for (int i=0; i<nIterThr; i++)
  {
    Eigen::Matrix3f m;
    float dist;
    
    _meanShiftIter(votes, votesWeight, currGuess, m, b);
    dist = _rotmDist2(currGuess, m);
    
    // Move to the meanshift mode by stepSize. Use quaternion interpolation to
    // rotate towards the target rotation.
    Eigen::Quaternion<float> q1(currGuess);
    Eigen::Quaternion<float> q2(m);

    currGuess = q1.slerp(stepSize, q2);
    if (dist<stepThr*stepThr) break;
  }
  mode = currGuess;
}


struct CmpIdxByWeight
{
  const std::vector<float> *w;
  CmpIdxByWeight(const std::vector<float> &w):w(&w){};
  bool operator() (unsigned int i1, unsigned int i2) const {return w->at(i1)>w->at(i2);}
};

struct CmpIdxByNGuess
{
  const std::vector<int> *nGuess;
  CmpIdxByNGuess(const std::vector<int> &nGuess):nGuess(&nGuess){};
  bool operator() (unsigned int i1, unsigned int i2) const {return nGuess->at(i1)>nGuess->at(i2);}
};

unsigned int  MultiGuessRChordMeanShift(const std::vector<Eigen::Vector3f> &votes,
					const std::vector<float> &votesWeight,
					const std::vector<Eigen::Vector3f> guesses,
					std::vector<Eigen::Vector3f> &modes,
					std::vector<int> &guessModeID,
					float b, float stepSize, float stepThr,
					unsigned int nIterThr, float sameModeThr)
{
  std::vector<Eigen::Matrix3f> vRotm(votes.size());
  std::vector<Eigen::Matrix3f> gRotm(guesses.size());
  std::vector<Eigen::Matrix3f> mRotm(guesses.size());
  std::vector<Eigen::Matrix3f> cRotm;
  std::vector<int> cNGuess;
  unsigned int nCModes=1;

  _cvtVEuler2Rotm(votes, vRotm);
  _cvtVEuler2Rotm(guesses, gRotm);

  #pragma omp parallel for
  for (int i=0; i<gRotm.size(); i++)
  {
    _RChordMeanShift(vRotm, votesWeight,
		     gRotm.at(i), mRotm.at(i),
		     b, stepSize, stepThr, nIterThr);
  }

  // Greedy clustering of modes (count how many guesses reach each mode as weel)
  cRotm.push_back(mRotm.at(0));
  cNGuess.push_back(1);
  guessModeID.at(0) = 0;
  for (int i=1; i<mRotm.size(); i++)
  {
    bool newMode = true;
    for (int j=0; j<nCModes; j++)
    {
      if (_rotmDist2(mRotm.at(i), cRotm.at(j))<sameModeThr*sameModeThr)
      {
	cNGuess.at(j)++;
	guessModeID.at(i) = j;
	newMode = false;
	break;
      }
    }
    if (newMode)
    {
      cRotm.push_back(mRotm.at(i));
      cNGuess.push_back(1);
      guessModeID.at(i) = nCModes;
      nCModes++;
    }
  }

  // Sort the modes by the number of guesses that converged to it
  std::vector<unsigned int> idx;
  for (unsigned int i=0; i<nCModes; i++) idx.push_back(i);
  if (idx.size()>1) std::sort(idx.begin(), idx.end(), CmpIdxByNGuess(cNGuess));

  int outNModes = std::min(nCModes, static_cast<unsigned int>(modes.size()));
  for (int i=0; i<outNModes; i++)
  {
    _cvtRotm2Euler(cRotm.at(idx[i]), modes.at(i));
  }

  // Update the mode id for votes that reached the current node
  // Note: Set the mode id to -1 if the guess reached a discarded mode
  boost::unordered_map<int, int> sortedModeIDMap;
  for (int i=0; i<nCModes; i++) sortedModeIDMap[idx[i]] = (i<outNModes) ? i : -1;
  for (int i=0; i<votes.size(); i++) guessModeID.at(i) = sortedModeIDMap.at(guessModeID.at(i));

  return outNModes;
}



void _meanShiftIter(const std::vector<Eigen::Matrix3f> &votes,
		    const std::vector<float> &weights,
		    const Eigen::Matrix3f &guess, Eigen::Matrix3f &mode, float b)
{
  Eigen::Matrix3f sumR = Eigen::Matrix3f::Zero();
  
  // Weighted sum of rotations
  for (int i=0; i<votes.size(); i++)
  {
    // Compute the current vote weight (i.e. provided weight multiplied by Guassian kernel
    // on chordal quaternion distance (i.e. Euclidean norm of R3 representation))
    float w = weights.at(i) * exp(-_rotmDist2(guess, votes.at(i))/(b*b));
    /*
    Eigen::Quaternion<float> q1(guess);
    Eigen::Quaternion<float> q2(votes.at(i));
    float dist = q1.angularDistance(q2);
    float w = weights.at(i) * exp(-(dist*dist)/(b*b));
    */
    sumR += w*votes.at(i);
  }
  
  // Compute the average rotation as in
  // Hartley, Richard, et al. "Rotation averaging." International journal of computer vision 103.3 (2013): 267-305.
  Eigen::JacobiSVD<Eigen::Matrix3f> SVDR(sumR, Eigen::ComputeFullU|Eigen::ComputeFullV);
  mode = SVDR.matrixU()*SVDR.matrixV().transpose();
  if (mode.determinant()<0)
  {
    mode = SVDR.matrixU()*Eigen::Vector3f(1,1,-1).asDiagonal()*SVDR.matrixV().transpose();
  }
}



void _cvtVEuler2Rotm(const std::vector<Eigen::Vector3f> &votesEul,
		    std::vector<Eigen::Matrix3f> &votesRotm)
{
  for (int i=0; i<votesEul.size(); i++)
  {
    const Eigen::Vector3f &currVoteEul = votesEul.at(i);
    Eigen::Matrix3f &currVoteRotm = votesRotm.at(i);
    _cvtEuler2Rotm(currVoteEul, currVoteRotm);
  }
}

void _cvtVRotm2Euler(const std::vector<Eigen::Matrix3f> &votesRotm,
		    std::vector<Eigen::Vector3f> &votesEul)
{
  for (int i=0; i<votesEul.size(); i++)
  {
    const Eigen::Matrix3f &currVoteRotm = votesRotm.at(i);
    Eigen::Vector3f &currVoteEul = votesEul.at(i);
    _cvtRotm2Euler(currVoteRotm, currVoteEul);
  }
}

void _cvtEuler2Rotm(const Eigen::Vector3f &voteEul, Eigen::Matrix3f &voteRotm)
{
  voteRotm = Eigen::AngleAxisf(voteEul.x(), Eigen::Vector3f::UnitZ()) * 
             Eigen::AngleAxisf(voteEul.y(), Eigen::Vector3f::UnitY()) * 
             Eigen::AngleAxisf(voteEul.z(), Eigen::Vector3f::UnitX());
}

void _cvtRotm2Euler(const Eigen::Matrix3f &voteRotm, Eigen::Vector3f &voteEul)
{
  voteEul = voteRotm.eulerAngles(2,1,0);
}


float _rotmDist2(const Eigen::Matrix3f &m1, const Eigen::Matrix3f &m2)
{
  Eigen::Matrix3f rotDiff = m1-m2;
  return rotDiff.cwiseProduct(rotDiff).sum();
}
