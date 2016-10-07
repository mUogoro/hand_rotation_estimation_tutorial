#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <softkinetic_grabber.hpp>

#include <padenti/image.hpp>
#include <padenti/cv_image_loader.hpp>
#include <padenti/tree.hpp>
#include <padenti/mh_rtree.hpp>
#include <padenti/cl_classifier.hpp>
#include <padenti/cl_regressor.hpp>

#include "particle_filter.hpp"
#include "particle_filter_hand.hpp"
#include "particle_filter_hand_prediction.hpp"
#include "particle_filter_hand_likelihood.hpp"


/***************************************************************************/
// Particle filter parameters
/***************************************************************************/
#define Q_INIT_SIGMA   (3*1.e-2)       // Quaternion orientation std for particles initialization
#define W_INIT_SIGMA   (.5e2)          // Angular velocity std for particles initialization
#define Q_UPDATE_SIGMA (Q_INIT_SIGMA)  // Quaternion orientation std for particles update
#define W_UPDATE_SIGMA (W_INIT_SIGMA)  // Angular velocity std for particles update
#define N_PARTICLES (1.e3)
#define N_BEST_ROT (1)


/***************************************************************************/
// Capture parameters
/***************************************************************************/
#define DEPTHWIDTH  (320)
#define DEPTHHEIGHT (240)
#define COLORWIDTH  (640)
#define COLORHEIGHT (480)
#define NEARMODE    (true)
//#define REGISTERED  (true)
#define REGISTERED  (false)
#define DEPTHFPS    (30)
#define COLORFPS    (30)


/***************************************************************************/
// Segmentation parameters
/***************************************************************************/
// Threshold on posterior for hand segmentation
#define HAND_PTHR (0.5)
#define DEPTH_THR (1500) // mm
#define SEG_RADIUS (12)  // cm


/***************************************************************************/
// Typedefs specialization
/***************************************************************************/
static const size_t N_SEG_LABELS = 2;
static const size_t N_ROT = 128;
static const int FEAT_SIZE = 2;
static const int N_HYPH = 2;

// Image types
typedef Image<unsigned short, 1> DepthmapT;
//typedef Image<unsigned char, 3> ColorImageT;
typedef Image<unsigned char, 4> ColorImageT;
typedef Image<unsigned char, 1> MaskT;

// Hand segmentation typedefs
typedef Tree<short int, FEAT_SIZE, N_SEG_LABELS> SegTreeT;
typedef TreeNode<short int, FEAT_SIZE> SegTreeNodeT;
typedef Image<float, N_SEG_LABELS> SegPredictionT;
typedef CLClassifier<unsigned short, 1, short, FEAT_SIZE, N_SEG_LABELS> SegClassifierT;

// Orientation regression typedefs
typedef MHRTree<short int, FEAT_SIZE, 4, N_HYPH> RotTreeT;
typedef RTreeNode<short int, FEAT_SIZE, 4> RotTreeNodeT;
typedef Image<int, 1> RotPredictionT;
typedef CLRegressor<unsigned short, 1, short, FEAT_SIZE, 4> RegressorT;



// Set to true if no OpenCL enabled GPU is available 
#define USE_CPU (false)

// Original training parameters
#define TRAIN_SEG_WIDTH  (320.)
#define TRAIN_SEG_HEIGHT (240.)
#define TRAIN_SEG_FX     (287.9079)
#define TRAIN_SEG_FY     (287.9079)

#define TRAIN_ROT_WIDTH  (320.)
#define TRAIN_ROT_HEIGHT (240.)
#define TRAIN_ROT_FX     (240.3077)
#define TRAIN_ROT_FY     (240.0749)


float _chordDist(const Eigen::Matrix3f &m1, const Eigen::Matrix3f &m2);

int main(int argc, const char *argv[])
{
  if (argc<5)
  {
    std::cout << "Usage: " << argv[0]
	      << " N_SEG_TREES SEG_TREE0.XML ... SEG_TREEN.XML"
	      << " N_PROT_TREES PROT_TREE0.XML ... PROT_TREEN.XML" << std::endl;
    return 1;
  }

  /*************************** Initialization ***************************/
  // Init the grabber
  RGBDGrabberParams params = {1, DEPTHWIDTH, DEPTHHEIGHT, DEPTHFPS,
			      COLORWIDTH, COLORHEIGHT, COLORFPS,
			      NEARMODE};
  //OpenNI2FrameGrabber grabber(params);
  SoftkineticGrabber grabber(params);
  float grabberFX = static_cast<float>(DEPTHWIDTH)/(2.*tan(grabber.getDepthHFOV()*M_PI/360));
  float grabberFY = static_cast<float>(DEPTHHEIGHT)/(2.*tan(grabber.getDepthVFOV()*M_PI/360));


  // Init hand segmentation classifier
  int nSegTrees = atoi(argv[1]);
  SegTreeT *segTrees = new SegTreeT[nSegTrees];
  SegClassifierT segClassifier(".", USE_CPU);

  for (int t=0; t<nSegTrees; t++)
  {
    segTrees[t].load(argv[t+2], t);

    // Scale all features 
    size_t nNodes = (2<<(segTrees[t].getDepth()-1))-1;
    for (int n=0; n<nNodes; n++)
    {
      const SegTreeNodeT &currTreeNode = segTrees[t].getNode(n);
      float u = static_cast<float>(currTreeNode.m_feature[0]);
      float v = static_cast<float>(currTreeNode.m_feature[1]);

      // Adjust for resolution
      u *= DEPTHWIDTH/TRAIN_SEG_WIDTH;
      v *= DEPTHHEIGHT/TRAIN_SEG_WIDTH;

      // Adjust for focal length
      u *= grabberFX/TRAIN_SEG_FX;
      v *= grabberFY/TRAIN_SEG_FY;

      currTreeNode.m_feature[0] = static_cast<short>(u);
      currTreeNode.m_feature[1] = static_cast<short>(v);
    }
  
    segClassifier << segTrees[t];
  }


  // Init rotation classifier
  int nRotTrees = atoi(argv[nSegTrees+2]);
  RotTreeT *rotTrees = new RotTreeT[nRotTrees];
  RegressorT regressor(".", USE_CPU);

  for (int t=0; t<nRotTrees; t++)
  {
    rotTrees[t].load(argv[t+nSegTrees+3], t);

    size_t nNodes = (2<<(rotTrees[t].getDepth()-1))-1;
    for (int n=0; n<nNodes; n++)
    {
      const RotTreeNodeT &currTreeNode = rotTrees[t].getNode(n);
      float u = static_cast<float>(currTreeNode.m_feature[0]);
      float v = static_cast<float>(currTreeNode.m_feature[1]);

      // Adjust for resolution
      u *= DEPTHWIDTH/TRAIN_SEG_WIDTH;
      v *= DEPTHHEIGHT/TRAIN_SEG_WIDTH;

      // Adjust for focal length
      u *= grabberFX/TRAIN_SEG_FX;
      v *= grabberFY/TRAIN_SEG_FY;

      currTreeNode.m_feature[0] = static_cast<short>(u);
      currTreeNode.m_feature[1] = static_cast<short>(v);
    }
    
    regressor << rotTrees[t];
  }

  float HFOV = grabber.getDepthHFOV();
  float VFOV = grabber.getDepthVFOV();
  float FX = DEPTHWIDTH/(2.f*tan(HFOV*M_PI/180/2));
  float FY = DEPTHHEIGHT/(2.f*tan(VFOV*M_PI/180/2));


  // Init the particle filter
  // Note: init state as rest pose with the palm facing the camera
  double initState[HAND_STATE_SIZE] = {0, 0.707, -0.707, 0.};
  HandPredictionModel handPredictionModel(Q_INIT_SIGMA, W_INIT_SIGMA,
					  Q_UPDATE_SIGMA, W_UPDATE_SIGMA);
  HandLikelihoodModel handLikelihoodModel(regressor);
  ParticleFilter<HAND_STATE_SIZE> particleFilter(handPredictionModel, handLikelihoodModel,
						 N_PARTICLES, initState);

  /*************************** Prediction loop ***************************/
  SegPredictionT segPrediction(DEPTHWIDTH, DEPTHHEIGHT);
  RotPredictionT rotPrediction(DEPTHWIDTH, DEPTHHEIGHT);
  cv::Rect handROI(0, 0, DEPTHWIDTH, DEPTHHEIGHT); // TODO: better estimate of initial position
  cv::Vec3f handCentroid(0,0,60);
  Particle<HAND_STATE_SIZE> prevParticle = particleFilter.getBestParticle();

  while (true)
  {
    // Acquire frames from the camera
    DepthmapT currDepth(DEPTHWIDTH, DEPTHHEIGHT);
    ColorImageT currColor(COLORWIDTH, COLORHEIGHT);
    MaskT currMask(DEPTHWIDTH, DEPTHHEIGHT);

    grabber.copyDepth(currDepth.getData());
    grabber.copyColor(currColor.getData());

    // Blur the depth frame
    cv::Mat cvDepth(DEPTHHEIGHT, DEPTHWIDTH, CV_16U,
		    reinterpret_cast<void*>(currDepth.getData()));
    cv::Mat cvMask(DEPTHHEIGHT, DEPTHWIDTH, CV_8U,
		   reinterpret_cast<void*>(currMask.getData()));

    cv::bitwise_and(cvDepth>0, cvDepth<DEPTH_THR, cvMask);
    cvDepth.setTo(0, cvDepth>=DEPTH_THR);
    cv::medianBlur(cvDepth, cvDepth, 5);

    // Foreground (i.e. hand) pixels classification
    segClassifier.predict(currDepth, segPrediction, currMask); 
    
    // Wrap classification results within OpenCV matrices
    cv::Mat cvHand(DEPTHHEIGHT, DEPTHWIDTH, CV_32F,
		   reinterpret_cast<void*>(segPrediction.getData()));
    cv::Mat cvBackground(DEPTHHEIGHT, DEPTHWIDTH, CV_32F,
			 reinterpret_cast<void*>(segPrediction.getData()+DEPTHWIDTH*DEPTHHEIGHT));
    cv::Mat cvColor(COLORHEIGHT, COLORWIDTH, CV_8UC3,
    //cv::Mat cvColor(COLORHEIGHT, COLORWIDTH, CV_8UC4,
		    reinterpret_cast<void*>(currColor.getData()));

    // Threshold posterior
    cvHand.setTo(0, cvHand<HAND_PTHR);

    // Camshift for 3D hand position tracking
    cv::RotatedRect csHandROI = cv::CamShift(cvHand, handROI,
					     cv::TermCriteria(cv::TermCriteria::EPS |
							      cv::TermCriteria::COUNT, 10, 1 ));

    // Compute bounding box 2D image coordinates
    if (csHandROI.center.x>=0 && csHandROI.center.x<currDepth.getWidth() &&
    	csHandROI.center.y>=0 && csHandROI.center.y<currDepth.getHeight())
    {
      float t;

      handCentroid[2] = cvDepth.at<unsigned short>(csHandROI.center.y, csHandROI.center.x)/10.f;
      handCentroid[0] = (csHandROI.center.x-DEPTHWIDTH/2)/FX*handCentroid[2];
      handCentroid[1] = (csHandROI.center.y-DEPTHHEIGHT/2)/FY*handCentroid[2];

      handROI.x = (handCentroid[0]-SEG_RADIUS)/handCentroid[2]*FX + DEPTHWIDTH/2;
      handROI.y = (handCentroid[1]-SEG_RADIUS)/handCentroid[2]*FY + DEPTHHEIGHT/2;
     
      t = (handCentroid[0]+SEG_RADIUS)/handCentroid[2]*FX + DEPTHWIDTH/2;
      handROI.width = t-handROI.x;
      t = (handCentroid[1]+SEG_RADIUS)/handCentroid[2]*FY + DEPTHHEIGHT/2;
      handROI.height = t-handROI.y;

      // Crop the bounding box inside image plane
      handROI.x = std::min(std::max(0, handROI.x), DEPTHWIDTH-2);
      handROI.y = std::min(std::max(0, handROI.y), DEPTHHEIGHT-2);
      handROI.width = std::min(std::max(1, handROI.width), DEPTHWIDTH-handROI.x-1);
      handROI.height = std::min(std::max(1, handROI.height), DEPTHHEIGHT-handROI.y-1);
    }

    // Orientation prediction using particle filter:
    // use camshift rectangle as segmentation mask
    cvMask.setTo(0);
    cvMask(handROI) = 255;
    cvMask.setTo(0, cvDepth>(handCentroid[2]+SEG_RADIUS)*10.f);
    cvMask.setTo(0, cvDepth<(handCentroid[2]-SEG_RADIUS)*10.f);
    cvDepth.setTo(0, cvMask==0);

    // TODO: how to handle multiple trees?
    handLikelihoodModel.updateModel(prevParticle, rotTrees[0], currDepth);
    particleFilter.step(1./30);
    const Particle<HAND_STATE_SIZE> &bestParticle = particleFilter.getBestParticle();

    // Find the mode closer to the best particle
    const std::vector<Eigen::Matrix3f> &modes = handLikelihoodModel.getModes();
    Eigen::Quaternion<double> q(bestParticle.state[0],
				bestParticle.state[1],
				bestParticle.state[2],
				bestParticle.state[3]);
    Eigen::Matrix3f rotm = q.toRotationMatrix().cast<float>();
    int bestMode = 0;
    float bestDist = _chordDist(rotm, modes.at(0));
    for (int i=1; i<handLikelihoodModel.getNModes(); i++)
    {
      float currDist = _chordDist(rotm, modes.at(i));
      if (currDist<bestDist)
      {
	bestDist = currDist;
	bestMode = i;
      }
    }
    rotm = modes.at(bestMode);


    // Done with rotation estimation for the current frame.
    // Display the results (foreground pixels, bagkground pixels, rotated principal axes
    cv::Mat cvBGRMask;
    cv::cvtColor(cvMask, cvBGRMask, CV_GRAY2BGR);
    cv::line(cvBGRMask, cv::Point(handROI.x+handROI.width/2,handROI.y+handROI.height/2),
    	     cv::Point(handROI.x+handROI.width/2  + 40.*rotm(0,0)/N_BEST_ROT,
    		       handROI.y+handROI.height/2 + 40.*rotm(1,0)/N_BEST_ROT),
    	     cv::Scalar(255,0,0), 4);
    cv::line(cvBGRMask, cv::Point(handROI.x+handROI.width/2,handROI.y+handROI.height/2),
    	     cv::Point(handROI.x+handROI.width/2  + 40.*rotm(0,1)/N_BEST_ROT,
    		       handROI.y+handROI.height/2 + 40.*rotm(1,1)/N_BEST_ROT),
    	     cv::Scalar(0,255,0), 4);
    cv::line(cvBGRMask, cv::Point(handROI.x+handROI.width/2,handROI.y+handROI.height/2),
    	     cv::Point(handROI.x+handROI.width/2  + 40.*rotm(0,2)/N_BEST_ROT,
    		       handROI.y+handROI.height/2 + 40.*rotm(1,2)/N_BEST_ROT),
    	     cv::Scalar(0,0,255), 4);
    cv::circle(cvBGRMask, cv::Point(csHandROI.center.x, csHandROI.center.y),
	       4, cv::Scalar(0,0,255), -1);
    cv::rectangle(cvBGRMask, handROI, cv::Scalar(0,255,0), 4);
    cv::rectangle(cvBGRMask, csHandROI.boundingRect(), cv::Scalar(0,0,255), 4);

    cv::Mat cvBGRColor;
    //cv::cvtColor(cvColor, cvBGRColor, CV_RGBA2BGRA);
    cv::cvtColor(cvColor, cvBGRColor, CV_RGB2BGR);
    cv::imshow("color", cvColor);
    cv::imshow("hand", cvHand);
    cv::imshow("mask", cvBGRMask);

    prevParticle = bestParticle;

    char key = cv::waitKey(1);
    if (key=='q')
    {
      break;
    }
  }

  cv::destroyAllWindows();
  delete []segTrees;
  delete []rotTrees;

  return 0;
}


float _chordDist(const Eigen::Matrix3f &m1, const Eigen::Matrix3f &m2)
{
  Eigen::Matrix3f rotDiff = m1-m2;
  return rotDiff.cwiseProduct(rotDiff).sum();
}
