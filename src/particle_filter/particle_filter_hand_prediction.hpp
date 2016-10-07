#ifndef __PARTICLE_FILTER_HAND_PREDICTION_HPP
#define __PARTICLE_FILTER_HAND_PREDICTION_HPP

#include <ctime>
#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>

#include "particle_filter_hand.hpp"
#include "particle_filter_base.hpp"

#define SMOOTH_FACTOR (1.)

class HandPredictionModel: public PredictionModel<HAND_STATE_SIZE>
{
private:
  double initQSigma;
  double initWSigma;
  double updateQSigma;
  double updateWSigma;
  
  
  void _updateParticle(Particle<HAND_STATE_SIZE> &p,
		       double deltaT, double qSigma, double wSigma,
		       boost::random::mt19937 &rng) const
  {
    // Quaternion kinematics from
    // Sola, Joan. "Quaternion kinematics for the error-state KF." Laboratoire d’Analyse et d’Architecture des Systemes-Centre national de la recherche scientifique (LAAS-CNRS), Toulouse, France, Tech. Rep (2012).
    boost::random::normal_distribution<double> Nq(0., qSigma);
    boost::random::normal_distribution<double> Nw(0., wSigma);
    boost::random::uniform_01<double> U;

    // Current state quaternions (orientation and angular velocity)
    Eigen::Quaternion<double> qt, wt;
    qt.w() = p.state[0];
    qt.x() = p.state[1];
    qt.y() = p.state[2];
    qt.z() = p.state[3];
    wt.w() = 0.;
    wt.x() = p.state[4]; 
    wt.y() = p.state[5]; 
    wt.z() = p.state[6];
    
    // Perturbation quaternions
    Eigen::Quaternion<double> deltaqt, deltawt;
    //deltaqt.w() =1. +Nq(rng);
    //deltaqt.x() = Nq(rng);
    //deltaqt.y() = Nq(rng);
    //deltaqt.z() = Nq(rng);
    Eigen::AngleAxisd deltaaxt;
    deltaaxt.angle() = Nq(rng);
    deltaaxt.axis().x() = -1.+2.*U(rng);
    deltaaxt.axis().y() = -1.+2.*U(rng);
    deltaaxt.axis().z() = -1.+2.*U(rng);
    deltaqt = deltaaxt;
    deltawt.w() = 0;
    deltawt.x() = Nw(rng);
    deltawt.y() = Nw(rng);
    deltawt.z() = Nw(rng);

    // Update quaternion rotation
    Eigen::Quaternion<double> qtw,qt1;
    // - quaternion update (for non zero velocity)
    if (deltaT)
    {
      qtw.w() = cos(wt.norm()*deltaT/2);
      qtw.x() = wt.x()/wt.norm()*sin(wt.norm()*deltaT/2);
      qtw.y() = wt.y()/wt.norm()*sin(wt.norm()*deltaT/2);
      qtw.z() = wt.z()/wt.norm()*sin(wt.norm()*deltaT/2);
      // - multiply by perturbation and update
      qt1 = deltaqt * qtw * qt;
    }
    else
    {
      // Null velocity, apply the random perturbation only
      qt1 = deltaqt * qt;
    }
	// Normalize
	qt1.normalize();

    // Update velocity
    Eigen::Quaternion<double> deltaq, wt1;
    Eigen::AngleAxisd axang;
    if (deltaT)
    {
      deltaq = qt1 * qt.inverse();
      axang = deltaq;
      wt1.w() = 0.;
      wt1.x() = axang.axis().x()*axang.angle()/deltaT + deltawt.x();
      wt1.y() = axang.axis().y()*axang.angle()/deltaT + deltawt.y();
      wt1.z() = axang.axis().z()*axang.angle()/deltaT + deltawt.z();
    }
    else
    {
      wt1.w() = 0.;
      wt1.x() = wt.x()+deltawt.x();
      wt1.y() = wt.y()+deltawt.y();
      wt1.z() = wt.z()+deltawt.z();
    }


    // Done, update particle status
    Eigen::Quaternion<double> iq;
    iq = qt.slerp(SMOOTH_FACTOR, qt1);
    p.state[0] = iq.w();
    p.state[1] = iq.x();
    p.state[2] = iq.y();
    p.state[3] = iq.z();
    
    p.state[4] = SMOOTH_FACTOR*wt1.x() + (1.-SMOOTH_FACTOR)*wt.x();
    p.state[5] = SMOOTH_FACTOR*wt1.y() + (1.-SMOOTH_FACTOR)*wt.y();
    p.state[6] = SMOOTH_FACTOR*wt1.z() + (1.-SMOOTH_FACTOR)*wt.z();
  }


  void _updateParticles(std::vector<Particle<HAND_STATE_SIZE> > &particles,
			double deltaT, double qSigma, double wSigma) const
  {
    boost::random::mt19937 rng(time(NULL));
    for (std::vector<Particle<HAND_STATE_SIZE> >::iterator it=particles.begin();
	 it!=particles.end(); ++it)
    {
      Particle<HAND_STATE_SIZE> &p = *it;
      _updateParticle(p, deltaT, qSigma, wSigma, rng);
    }
  }

public:
  HandPredictionModel(double initQSigma, double initWSigma,
		      double updateQSigma, double updateWSigma):
    initQSigma(initQSigma),
    initWSigma(initWSigma),
    updateQSigma(updateQSigma),
    updateWSigma(updateWSigma)
  {}

  void init(const double *initState,
	    std::vector<Particle<HAND_STATE_SIZE> > &particles) const
  {
    // Set all particle to the initial state
    for (std::vector<Particle<HAND_STATE_SIZE> >::iterator it=particles.begin();
	 it!=particles.end(); ++it)
    {
      Particle<HAND_STATE_SIZE> &p = *it;
      std::copy(initState, initState+HAND_STATE_SIZE, p.state);
      p.likelihood = 0.;
    }
    
    _updateParticles(particles, 0., initQSigma, initWSigma);
  }

  void update(std::vector<Particle<HAND_STATE_SIZE> > &particles, double deltaT) const
  {
    _updateParticles(particles, deltaT, updateQSigma, updateWSigma);
  }
};




#endif // __PARTICLE_FILTER_HAND_PREDICTION_HPP
