#ifndef __PARTICLE_FILTER_HAND_LIKELIHOOD_HPP
#define __PARTICLE_FILTER_HAND_LIKELIHOOD_HPP

#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <padenti/image.hpp>
#include <padenti/mh_rtree.hpp>
#include <padenti/cl_regressor.hpp>

#include "rchord_meanshift.hpp"

#include "particle_filter_hand.hpp"
#include "particle_filter_base.hpp"

// Random Forest parameters
#define RF_FEAT_SIZE (2)
#define RF_VAL_SIZE  (4)
#define RF_N_HYPH    (2)

// Meanshift parameters
#define MS_N_HYPH    (10)
#define MS_SIGMA    (1.6223)
#define MS_STEPSIZE (0.2)
#define MS_STEPTHR  (0.025)
#define MS_SAMECTHR (0.15)
#define MS_NITERTHR (10u)
#define MS_SAMPLED_PIXELS (200u)

// Likelihood parameters
#define EVAL_SIGMA  (MS_SIGMA/3)

// Commodity typedefs to avoid long class type declaration
typedef Image<unsigned short, 1> DepthT;
typedef Image<unsigned char, 1> MaskT;
typedef Image<int, 1> PredictionT;
typedef MHRTree<short int, RF_FEAT_SIZE, RF_VAL_SIZE, RF_N_HYPH> MHRTreeT;
typedef RTreeNode<short int, RF_FEAT_SIZE, RF_VAL_SIZE> RTreeNodeT;
typedef CLRegressor<unsigned short, 1, short, RF_FEAT_SIZE, RF_VAL_SIZE> RegressorT;


class HandLikelihoodModel: public LikelihoodModel<HAND_STATE_SIZE>
{
private:
  RegressorT &regressor;
  Particle<HAND_STATE_SIZE> prevState;
  std::vector<Eigen::Matrix3f> modes;
  std::vector<float> modesW;
  std::vector<int> modesCount;
  std::vector<float> modesVar;
  int nModes;
  float evalSigma;
  float deltaT;
public:

  HandLikelihoodModel(RegressorT &regressor, 
		      float evalSigma=EVAL_SIGMA):
    regressor(regressor),
    modes(std::vector<Eigen::Matrix3f>(MS_N_HYPH)),
    modesW(std::vector<float>(MS_N_HYPH)),
    modesCount(std::vector<int>(MS_N_HYPH)),
    modesVar(std::vector<float>(MS_N_HYPH)),
    nModes(0),
    evalSigma(evalSigma)
  {}

  const std::vector<Eigen::Matrix3f> &getModes() const {return modes;}
  const std::vector<float> &getModeWeights() const {return modesW;}
  const std::vector<int> &getModeCount() const {return modesCount;}
  const std::vector<float> &getModeVar() const {return modesVar;}
  int getNModes() const {return nModes;}

  void eval(std::vector<Particle<HAND_STATE_SIZE> > &particles) const
  {
    Eigen::Quaternion<float> prevq(prevState.state[0],
				   prevState.state[1],
				   prevState.state[2],
				   prevState.state[3]);
    Eigen::Vector3d prevV(prevState.state[4],
			  prevState.state[5],
			  prevState.state[6]);

    // Compute the particles weight as the Panzer window estimate on rotation chordal
    // distance values using a Gaussian kernel
    // TODO: per-mode bandwidth as in Herdtweek paper?
    for (int i=0; i<particles.size(); i++)
    {
      Eigen::Quaternion<float> q(particles.at(i).state[0],
				 particles.at(i).state[1],
				 particles.at(i).state[2],
				 particles.at(i).state[3]);
      Eigen::Vector3d currV(particles.at(i).state[4],
			    particles.at(i).state[5],
			    particles.at(i).state[6]);
      Eigen::Matrix3f R(q);

      // Compute the likelihood
      double l = 0;
      for (int j=0; j<nModes; j++)
      {
	Eigen::Matrix3f rotDiff = R-modes.at(j);
	l += modesW.at(j) * exp(-rotDiff.cwiseProduct(rotDiff).sum()/(evalSigma*evalSigma));
	//l += modesCount.at(j)/sqrt(modesVar.at(j)) * 
	//  exp(-rotDiff.cwiseProduct(rotDiff).sum()/(modesVar.at(j)));
      }


     // Compute the velocity with respect to the previous best particle rotation
      // TODO: parameterize velocity bound
      Eigen::Quaternion<float> deltaq;
      Eigen::AngleAxisf axang;
      float V;
      deltaq = q * prevq.inverse();
      axang = deltaq;
      V = axang.angle()/deltaT;
      particles.at(i).likelihood = V<10*M_PI ? l : 0.;


      //particles.at(i).likelihood = l;      
      //double diffV = (currV-prevV).norm();
      //particles.at(i).likelihood *= exp(-(diffV*diffV)/(M_PI*M_PI));
      //particles.at(i).likelihood *= exp(-(V*V)/(25*M_PI*M_PI));
      
    }
  }

  void updateModel(const Particle<HAND_STATE_SIZE> &_prevState, 
		   const MHRTreeT &rtree, const DepthT &depthmap,
		   float _deltaT=1./30)
  {
    // Perform RF prediction on the current depthmap. The depthmap is supposed to
    // be already segmented
    // - Mask as non-zero pixels
    MaskT mask(depthmap.getWidth(), depthmap.getHeight());
    cv::Mat cvDepth(depthmap.getHeight(), depthmap.getWidth(), CV_16U,
		    reinterpret_cast<unsigned char*>(depthmap.getData()));
    cv::Mat cvMask(depthmap.getHeight(), depthmap.getWidth(), CV_8U,
		   reinterpret_cast<unsigned char*>(mask.getData()));
    cvMask.setTo(0);
    cvMask.setTo(1, cvDepth>0);

    // - perform prediction
    // TODO: handle multiple trees
    PredictionT prediction(depthmap.getWidth(), depthmap.getHeight());
    regressor.predict(0, depthmap, prediction, mask);

    // - sample image pixels, i.e. work on a subset of the per-pixel prediction
    // in order to guarantee real-time execution
    unsigned int *nonNullPixelsBuff = 
      new unsigned int[prediction.getWidth()*prediction.getHeight()];
    unsigned int nonNullPixels=0, nSamples=0;
    for (unsigned int i=0; i<prediction.getWidth()*prediction.getHeight(); i++)
    {
      if (mask.getData()[i]) nonNullPixelsBuff[nonNullPixels++]=i;
    }
    nSamples = std::min(MS_SAMPLED_PIXELS, nonNullPixels);

    if (!nSamples)
    {
      // TODO: how to handle the lack of samples?
      return;
    }

    unsigned int *samples = new unsigned int[nSamples];
    if (nSamples<MS_SAMPLED_PIXELS)
    {
      // Not enough samples (i.e. non-null pixels) to perform sampling.
      // Simply copy the whole set of samples
      std::copy(nonNullPixelsBuff, nonNullPixelsBuff+nSamples, samples);
    }
    else
    {
      boost::random::mt19937 gen(time(NULL));
      boost::random::uniform_int_distribution<> U(0, nonNullPixels-1);
      for (int i=0; i<nSamples; i++) samples[i] = nonNullPixelsBuff[U(gen)];
    }

    // Build the list of votes (i.e. Euler angles) and perform meanshift
    std::vector<Eigen::Vector3f> votes;
    std::vector<float> votesW;
    for (int i=0; i<nSamples; i++)
    {
      int nodeID = prediction.getData()[samples[i]];
      unsigned int nVotes = rtree.getNVotes()[nodeID];

      for (int j=0; j<nVotes; j++)
      {
	Eigen::Vector3f vote;
	float *votesPtr = 
	  &rtree.getVotes()[nodeID*RF_VAL_SIZE*RF_N_HYPH + j*RF_VAL_SIZE];
	float *weightPtr = 
	  &rtree.getVoteWeights()[nodeID*RF_N_HYPH+j];

	vote.z() = votesPtr[0]*M_PI/180.; //static_cast<double>(votesPtr[0])*M_PI/180.;
        vote.y() = votesPtr[1]*M_PI/180.; //static_cast<double>(votesPtr[1])*M_PI/180.;
	vote.x() = votesPtr[2]*M_PI/180.; //static_cast<double>(votesPtr[2])*M_PI/180.;
	votes.push_back(vote);
	votesW.push_back(*weightPtr);
	//votesW.push_back(static_cast<double>(*weightPtr));
      }
    }

    std::vector<Eigen::Vector3f> eulModes(MS_N_HYPH);
    std::vector<int> voteModeID(votes.size());
    nModes = MultiGuessRChordMeanShift(votes, votesW, votes, eulModes, voteModeID,
				       MS_SIGMA, MS_STEPSIZE, MS_STEPTHR,
				       MS_NITERTHR, MS_SAMECTHR);

    // Store modes found as rotation matrices
    for (int i=0; i<nModes; i++)
    {
      modes.at(i) = Eigen::AngleAxisf(eulModes.at(i).x(), Eigen::Vector3f::UnitZ()) * 
	            Eigen::AngleAxisf(eulModes.at(i).y(), Eigen::Vector3f::UnitY()) * 
	            Eigen::AngleAxisf(eulModes.at(i).z(), Eigen::Vector3f::UnitX());
    }
    
    // Update the mode weights as the sum of (weights of the) votes that converge to
    // each specific mode
    std::fill(modesW.begin(), modesW.end(), 0.);
    std::fill(modesCount.begin(), modesCount.end(), 0);
    std::fill(modesVar.begin(), modesVar.end(), 0.);
    for (int i=0; i<votes.size(); i++)
    {
      if (voteModeID.at(i)!=-1) 
      {
	modesW.at(voteModeID.at(i)) += votesW.at(i);
	modesCount.at(voteModeID.at(i))++;
	
	Eigen::Matrix3f voteMat(Eigen::AngleAxisf(votes.at(i).x()*M_PI/180,
						  Eigen::Vector3f::UnitZ()) * 
				Eigen::AngleAxisf(votes.at(i).y()*M_PI/180,
						  Eigen::Vector3f::UnitY()) * 
				Eigen::AngleAxisf(votes.at(i).z()*M_PI/180,
						  Eigen::Vector3f::UnitX()));
	Eigen::Quaternion<float> modeQ(modes.at(voteModeID.at(i)));
	Eigen::Quaternion<float> voteQ(voteMat);

	Eigen::Quaternion<double> tmp(voteQ.inverse() * modeQ);
	float distQ = 2.*acos(fabs(tmp.w()));
	modesVar.at(voteModeID.at(i)) += distQ*distQ;
      }
    }

    for (int i=0; i<nModes; i++) modesVar.at(i)/=modesCount.at(i);

    // Finally, save the previous state and time delta (they will be used to disregard unplausible rotations)
    prevState = _prevState;
    deltaT = _deltaT;

    // Done
    delete []samples;
    delete []nonNullPixelsBuff;
  }
};


#endif // __PARTICLE_FILTER_HAND_LIKELIHOOD_HPP
