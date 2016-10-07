/******************************************************************************
 * Padenti Library
 *
 * Copyright (C) 2015  Daniele Pianu <daniele.pianu@ieiit.cnr.it>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 ******************************************************************************/

#ifndef __MH_RTREE_HPP
#define __MH_RTREE_HPP


#include <padenti/rtree.hpp>


/** \todo Avoid redundant code from MHTree */
template <typename FeatType, unsigned int FeatDim, unsigned int RDim, unsigned int VDim>
class MHRTree: public RTree<FeatType, FeatDim, RDim>
{
private:
  float *m_votes;
  unsigned int *m_nVotes;
  float *m_voteWeights;

  void _init()
  {
    unsigned int tDepth = RTree<FeatType, FeatDim, RDim>::getDepth();
    unsigned int nNodes = (2<<(tDepth-1))-1;

    m_votes = new float[nNodes*RDim*VDim];
    m_nVotes = new unsigned int[nNodes];
    m_voteWeights = new float[nNodes*VDim];

    std::fill(m_votes, m_votes+nNodes*RDim*VDim, 0);
    std::fill(m_nVotes, m_nVotes+nNodes, 0);
    std::fill(m_voteWeights, m_voteWeights+nNodes*VDim, 0);
  }

  void _clean()
  {
    delete []m_voteWeights;
    delete []m_nVotes;
    delete []m_votes;
  }

public:
  MHRTree():RTree<FeatType, FeatDim, RDim>(),
	   m_votes(NULL), m_nVotes(NULL), m_voteWeights(NULL){};
  MHRTree(unsigned int id, unsigned int depth):
    RTree<FeatType, FeatDim, RDim>(id, depth)
  {
    _init();
  }
  ~MHRTree()
  {
    _clean();
  }

  float *getVotes() const {return m_votes;}
  unsigned int *getNVotes() const {return m_nVotes;}
  float *getVoteWeights() const {return m_voteWeights;}

  void load(const std::string &treePath, int idx=-1)
  {
    unsigned int currTDepth = RTree<FeatType, FeatDim, RDim>::getDepth();
    RTree<FeatType, FeatDim, RDim>::load(treePath, idx);

    boost::property_tree::ptree pt;
    boost::property_tree::read_xml(treePath, pt);

    unsigned int tDepth = RTree<FeatType, FeatDim, RDim>::getDepth();
    unsigned int nNodes = (2<<(tDepth-1))-1;
    if (currTDepth<tDepth)
    {
      _clean();
      _init();
    }
    
    unsigned int id = RTree<FeatType, FeatDim, RDim>::getID();
    for (unsigned int i=0; i<nNodes; i++)
    {
      std::stringstream nodePathStream;
      std::stringstream nodeValueStream;
      const RTreeNode<FeatType, FeatDim, RDim> &currNode =
	RTree<FeatType, FeatDim, RDim>::getNode(i);
      if (*currNode.m_leftChild!=-1) continue;

      nodePathStream << "Trees.Tree" << id << ".Node" << i << ".Votes";
      nodeValueStream.str(pt.get<std::string>(nodePathStream.str()));

      float *votesPtr = m_votes+i*RDim*VDim;
      for (int j=0; j<RDim*VDim; j++)
      {
	nodeValueStream >> votesPtr[j];
      }
      
      nodePathStream.clear(); nodePathStream.str("");
      nodeValueStream.clear(); nodeValueStream.str("");
      nodePathStream << "Trees.Tree" << id << ".Node" << i << ".NVotes";
      m_nVotes[i] = pt.get<unsigned int>(nodePathStream.str());

      nodePathStream.clear(); nodePathStream.str("");
      nodeValueStream.clear(); nodeValueStream.str("");
      nodePathStream << "Trees.Tree" << id << ".Node" << i << ".VoteWeights";
      nodeValueStream.str(pt.get<std::string>(nodePathStream.str()));

      float *voteWeightsPtr = m_voteWeights+i*VDim;
      for (int j=0; j<VDim; j++)
      {
	nodeValueStream >> voteWeightsPtr[j];
      }
    }
  }
  
  void save(const std::string &treePath, int idx=-1) const
  {
    RTree<FeatType, FeatDim, RDim>::save(treePath, idx);
    
    boost::property_tree::ptree pt;
    boost::property_tree::read_xml(treePath, pt);

    unsigned int tDepth = RTree<FeatType, FeatDim, RDim>::getDepth();
    unsigned int nNodes = (2<<(tDepth-1))-1;
    
    unsigned int id = RTree<FeatType, FeatDim, RDim>::getID();
    for (unsigned int i=0; i<nNodes; i++)
    {
      std::stringstream nodePathStream;
      std::stringstream nodeValueStream;
      const RTreeNode<FeatType, FeatDim, RDim> &currNode = 
	RTree<FeatType, FeatDim, RDim>::getNode(i);
      if (*currNode.m_leftChild!=-1) continue;

      float *votesPtr = m_votes+i*RDim*VDim;
      nodePathStream << "Trees.Tree" << id << ".Node" << i << ".Votes";
      for (int j=0; j<RDim*VDim-1; j++)
      {
	nodeValueStream << votesPtr[j] << " ";
      }
      nodeValueStream << votesPtr[RDim*VDim-1];
      pt.add(nodePathStream.str(), nodeValueStream.str());

      nodePathStream.clear(); nodePathStream.str("");
      nodeValueStream.clear(); nodeValueStream.str("");
      nodePathStream << "Trees.Tree" << id << ".Node" << i << ".NVotes";
      pt.add(nodePathStream.str(), m_nVotes[i]);

      nodePathStream.clear(); nodePathStream.str("");
      nodeValueStream.clear(); nodeValueStream.str("");
      nodePathStream << "Trees.Tree" << id << ".Node" << i << ".VoteWeights";
      float *voteWeightsPtr = m_voteWeights+i*VDim;
      for (int j=0; j<VDim-1; j++)
      {
	nodeValueStream << voteWeightsPtr[j] << " ";
      }
      nodeValueStream << voteWeightsPtr[VDim-1];
      pt.add(nodePathStream.str(), nodeValueStream.str());
    }

    boost::property_tree::write_xml(treePath, pt);
  }
};

#endif // __MH_RTREE_HPP
