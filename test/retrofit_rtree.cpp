#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <numeric>
#include <Eigen/Dense>
#include <boost/unordered_map.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <padenti/cv_image_loader.hpp>
#include <padenti/uniform_image_sampler.hpp>
#include <padenti/mh_rtree.hpp>
#include <padenti/cl_regressor.hpp>

#include "rchord_meanshift.hpp"

#define RDIM (4)
#define N_SAMPLES (2048)

// Default parameters 
#define DEFAULT_MS_SIGMA       (0.7321)
#define DEFAULT_MS_STEPSIZE    (0.2)
#define DEFAULT_MS_STEPTHR     (0.025)
#define DEFAULT_MS_SAMECTHR    (0.15)
#define DEFAULT_MS_NITERTHR    (10)
#define DEFAULT_RESERVOIR_SIZE (200)


// MeanShift number of hypothesis
#define MS_NHYPH    (2)

#define USE_CPU (false)

typedef CVImageLoader<unsigned short, 1> DepthmapLoaderT;
typedef UniformImageSampler<unsigned short, 1> SamplerT;
typedef RTree<short int, 2, RDIM> RTreeT;
typedef RTreeNode<short int, 2, RDIM> RTreeNodeT;
typedef MHRTree<short int, 2, RDIM, MS_NHYPH> MHRTreeT;
typedef CLRegressor<unsigned short, 1, short, 2, RDIM> RegressorT;
typedef Image<unsigned short, 1> DepthmapT;
typedef Image<unsigned char, 1> MaskT;
typedef Image<int, 1> PredictionT;


int main(int argc, const char *argv[])
{
  if (argc!=4 && argc!=10)
  {
    std::cout << "Usage: " 
	      << argv[0] << " TREE_FILE.xml DEPTH_LIST_FILE.txt EULER_ANGLES.txt [MS_BANDWIDTH MS_STEPSIZE MS_STEPTHR MS_GROUPINGTHR MS_NITERTHR MS_RESERVOIRSIZE]"
	      << std::endl;
    return 1;
  }

  boost::random::mt19937 gen;
  boost::random::uniform_01<> dist;
  
  // Setup the tree and classifier
  RTreeT tree;
  RegressorT regressor(".", USE_CPU);
  tree.load(argv[1]);
  regressor << tree;

  // Read the meanshift parameters from commandline
  float MS_SIGMA, MS_STEPSIZE, MS_STEPTHR, MS_SAMECTHR, MS_NITERTHR, RESERVOIR_SIZE;
  if (argc==4)
  {
    MS_SIGMA = DEFAULT_MS_SIGMA;
    MS_STEPSIZE = DEFAULT_MS_STEPSIZE;
    MS_STEPTHR = DEFAULT_MS_STEPTHR;
    MS_SAMECTHR = DEFAULT_MS_SAMECTHR;
    MS_NITERTHR = DEFAULT_MS_NITERTHR;
    RESERVOIR_SIZE = DEFAULT_RESERVOIR_SIZE;
  }
  else
  {
    MS_SIGMA = atof(argv[4]);
    MS_STEPSIZE = atof(argv[5]);
    MS_STEPTHR = atof(argv[6]);
    MS_SAMECTHR = atof(argv[7]);
    MS_NITERTHR = atof(argv[8]);
    RESERVOIR_SIZE = atoi(argv[9]);
  }

  // Scan the tree's nodes to find leaves
  boost::unordered_map<int, int> leafIdxMap;
  std::vector<int> leafIdx;
  unsigned int nNodes = (2<<(tree.getDepth()-1))-1, nLeaves=0;
  for (int i=0; i<nNodes; i++)
  {
    const RTreeNodeT &currNode = tree.getNode(i);
    if (*currNode.m_leftChild!=-1) continue;

    leafIdx.push_back(i);
    leafIdxMap[i] = nLeaves++;
  }

  // Build the vectors which will host per-leaf rotations
  std::vector<std::vector<Eigen::Vector3f> > perLeafSamples(nLeaves);
  std::vector<int> perLeafNSamples(nLeaves);
  std::fill(perLeafNSamples.begin(), perLeafNSamples.end(), 0);

  // Load the training set
  DepthmapLoaderT depthmapLoader;
  SamplerT sampler(N_SAMPLES, 0);
  
  std::vector<Eigen::Vector3f> rotations;
  std::ifstream depthList(argv[2]);
  std::ifstream eulerList(argv[3]);
  std::string depthStr, eulerStr;
  while (getline(depthList, depthStr) && getline(eulerList, eulerStr))
  {
    // Read the Euler triplet for current depth (in XYZ) and keep it in ZYX
    std::stringstream anglesStr(eulerStr);
    Eigen::Vector3f angles;
    
    anglesStr >> angles.z();
    anglesStr >> angles.y();
    anglesStr >> angles.x();
    angles *= M_PI/180.;

    // Read the depthmap and sample it
    DepthmapT depthmap = depthmapLoader.load(depthStr);
    unsigned int samples[N_SAMPLES], nSamples;

    // Trick: create the labels used for sampling as a simple mask of non-zero depthmap pixels
    MaskT mask(depthmap.getWidth(), depthmap.getHeight());
    for (int i=0; i<depthmap.getWidth()*depthmap.getHeight(); i++)
    {
      mask.getData()[i] = static_cast<unsigned char>(depthmap.getData()[i]>0);
    }

    nSamples = sampler.sample(depthmap.getData(), mask.getData(),
			      depthmap.getWidth(), depthmap.getHeight(), samples);

    // Perform leaf prediction and update the list of per leaf rotation using
    // reservoir sampling
    PredictionT prediction(depthmap.getWidth(), depthmap.getHeight());
    regressor.predict(0, depthmap, prediction, mask);
    for (int i=0; i<nSamples; i++)
    {
      int nodeId = prediction.getData()[samples[i]];
      std::vector<Eigen::Vector3f> &nodeRot = perLeafSamples.at(leafIdxMap.at(nodeId));

      int nodeNSamples = perLeafNSamples.at(leafIdxMap.at(nodeId));
      if (nodeNSamples < RESERVOIR_SIZE)
      {
	nodeRot.push_back(angles);
      }
      else
      {
	unsigned int j = roundf(dist(gen)*nodeNSamples);
	if (j<RESERVOIR_SIZE)
	{
	  nodeRot.at(j) = angles;
	}
      }
      perLeafNSamples.at(leafIdxMap.at(nodeId))++;
    }
  }

  std::cout << "Leaves' samples list built, start mean shift regression" << std::endl;

  // Init the multihyphotesis tree
  // - create an empty tree
  MHRTreeT mhRTree(tree.getID(), tree.getDepth());
  // - copy the content from the classification tree
  std::copy(tree.getLeftChildren(), tree.getLeftChildren()+nNodes, mhRTree.getLeftChildren());
  std::copy(tree.getFeatures(), tree.getFeatures()+nNodes*2, mhRTree.getFeatures());
  std::copy(tree.getThresholds(), tree.getThresholds()+nNodes, mhRTree.getThresholds());
  std::copy(tree.getValues(), tree.getValues()+nNodes*RDIM, mhRTree.getValues());

  // Perform mean shift on leaves' rotations
  std::ofstream leavesStats("rleavesStats.txt");
  for (int i=0; i<nLeaves; i++)
  {
    const std::vector<Eigen::Vector3f> &votes = perLeafSamples.at(i);
    std::vector<float> votesW(votes.size()); std::fill(votesW.begin(), votesW.end(), 1.f);
    std::vector<int> votesID(perLeafSamples.at(i).size());
    std::vector<Eigen::Vector3f> modes(MS_NHYPH);

    int nodeID = leafIdx.at(i);
    std::cout << "Node " << nodeID << ":";

    int nModes = MultiGuessRChordMeanShift(votes, votesW, votes, modes, votesID,
					   MS_SIGMA, MS_STEPSIZE, MS_STEPTHR,
					   MS_NITERTHR, MS_SAMECTHR);

    
    std::vector<int> modesW(nModes); std::fill(modesW.begin(), modesW.end(), 0);
    for (int j=0; j<votesID.size(); j++)
    {
      int id = votesID.at(j);
      if (id!=-1) modesW.at(id)++;
    }

    for (int j=0; j<nModes; j++)
      std::cout << " " << modes.at(j).z() 
		<< " " << modes.at(j).y() 
		<< " " << modes.at(j).x() 
		<< " (weight: " << modesW.at(j) << ")";
    std::cout << std::endl;
				   
    // Update the node hypothesis
    float *nodeVotesPtr = mhRTree.getVotes()+nodeID*4*MS_NHYPH;
    for (int j=0; j<MS_NHYPH; j++)
    {
      nodeVotesPtr[j*4]   = modes.at(j).z()*180.f/M_PI;
      nodeVotesPtr[j*4+1] = modes.at(j).y()*180.f/M_PI;
      nodeVotesPtr[j*4+2] = modes.at(j).x()*180.f/M_PI;
      nodeVotesPtr[j*4+3] = 0.f;
    }
    unsigned int *nodeNVotesPtr = mhRTree.getNVotes()+nodeID;
    *nodeNVotesPtr = nModes;
    float *nodeVoteWeightsPtr = mhRTree.getVoteWeights()+nodeID*MS_NHYPH;
    std::copy(modesW.begin(), modesW.begin()+nModes, nodeVoteWeightsPtr);
  }

  // Done, save the tree
  mhRTree.save("mhrtree.xml");

  return 0;
}
