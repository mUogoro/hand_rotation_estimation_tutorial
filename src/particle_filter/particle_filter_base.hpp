#ifndef __PARTICLE_FILTER_BASE_HPP
#define __PARTICLE_FILTER_BASE_HPP

#include <vector>
#include <algorithm>
#include <boost/random/mersenne_twister.hpp>

template <size_t Size>
class Particle
{
public:
  double state[Size];
  double likelihood;

  Particle()
  {
    std::fill(state, state+Size, 0.);
    likelihood = 0;
  }

  Particle(const Particle& p)
  {
    std::copy(p.state, p.state+Size, state);
    likelihood = p.likelihood;
  }

  Particle &operator=(const Particle &p)
  {
    if (this!=&p)
    {
      std::copy(p.state, p.state+Size, state);
      likelihood = p.likelihood;
    }
    return *this;
  }
};


template <size_t Size>
class PredictionModel
{
public:
  virtual void init(const double *initState,
		    std::vector<Particle<Size> > &particles) const =0;
  virtual void update(std::vector<Particle<Size> > &particles, double deltaT) const =0;
};


template <size_t Size>
class LikelihoodModel
{
public:
  virtual void eval(std::vector<Particle<Size> > &particles) const =0;
};


template <size_t Size>
class ParticleFilter
{
private:
  const PredictionModel<Size> &pm;
  const LikelihoodModel<Size> &lm;
  std::vector<Particle<Size> > particles;
  int epoch;
  boost::random::mt19937 rng;
public:
  ParticleFilter(const PredictionModel<Size> &pm, const LikelihoodModel<Size> &lm,
		 unsigned int nParticles=100, const double *initState=NULL);
  void step(double deltaT);
  
  const std::vector<Particle<Size> > &getParticles() const;
  const Particle<Size> &getBestParticle() const;
};


#endif // __PARTICLE_FILTER_BASE_HPP
