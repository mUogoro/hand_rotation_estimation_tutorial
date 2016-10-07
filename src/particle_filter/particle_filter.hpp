#ifndef __PARTICLE_FILTER_HPP
#define __PARTICLE_FILTER_HPP


#include <boost/random/discrete_distribution.hpp>

#include "particle_filter_base.hpp"


template <size_t Size>
ParticleFilter<Size>::ParticleFilter(const PredictionModel<Size> &pm,
				     const LikelihoodModel<Size> &lm,
				     unsigned int nParticles,
				     const double *initState):
  pm(pm),
  lm(lm),
  particles(std::vector<Particle<Size> >(nParticles)),
  epoch(0)
{
  if (initState)
  {
    pm.init(initState, particles);
  }
  else
  {
    double *_initState = new double[Size];
    std::fill(_initState, _initState+Size, 0);
    pm.init(initState, particles);
    delete []_initState;
  }
}


template <size_t Size>
const std::vector<Particle<Size> > &ParticleFilter<Size>::getParticles() const
{
  return particles;
}

template <size_t Size>
const Particle<Size> &ParticleFilter<Size>::getBestParticle() const
{
  
  double bestLikelihood = particles.at(0).likelihood;
  int bestID = 0;

  for (typename std::vector<Particle<Size> >::const_iterator it=particles.begin()+1;
       it!=particles.end(); ++it)
  {
    const Particle<Size> &p = *it;
    if (p.likelihood>bestLikelihood)
    {
      bestLikelihood = p.likelihood;
      bestID = it-particles.begin();
    }
  }

  return particles.at(bestID);
}


template <size_t Size>
void ParticleFilter<Size>::step(double deltaT)
{
  // Create a copy of the current particles
  std::vector<Particle<Size> > tempParticles(particles.size());
  std::copy(particles.begin(), particles.end(), tempParticles.begin());

  // Particle filter iteration
  // - update the particles state using the provided state-update model
  pm.update(tempParticles, deltaT);

  // - compute the importance weights using the provided likelihood model
  lm.eval(tempParticles);
  
  // Normalize the particles weight (and build the list of normalized weights
  // used in the next selection step)
  double wSum = 0.;
  for (typename std::vector<Particle<Size> >::iterator it=tempParticles.begin();
       it!=tempParticles.end(); ++it)
  {
    const Particle<Size> &p = *it;
    wSum += p.likelihood;
  }

  std::vector<double> normW;
  for (typename std::vector<Particle<Size> >::iterator it=tempParticles.begin();
       it!=tempParticles.end(); ++it)
  {
    Particle<Size> &p = *it;
    p.likelihood /= wSum;
    normW.push_back(p.likelihood);
  }

  // - selection step
  // TODO: parameterize at which epoch execute selection
  //if (!(epoch%10))
  if (true)
  {
    boost::random::discrete_distribution<> D(normW.begin(), normW.end());

    std::vector<int> idx;
    for (int i=0; i<tempParticles.size(); i++) idx.push_back(D(rng));
  
    // Copy back the selected particles
    for (int i=0; i<idx.size(); i++) particles.at(i) = tempParticles.at(idx.at(i));
  }
  else
  {
    std::copy(tempParticles.begin(), tempParticles.end(), particles.begin());
  }

  epoch++;

  // Done
}

#endif // __PARTICLE_FILTER_HPP
