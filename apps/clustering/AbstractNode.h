/*
 * AbstractNode.h
 *
 *  Created on: Jun 30, 2011
 *      Author: rashid
 */
#include <stdlib.h>
#include<assert.h>
#include<limits>

#ifndef ABSTRACTNODE_H_
#define ABSTRACTNODE_H_
using namespace std;
/**
 * A Intensity spectrum which can vary over both color (RGB) and time (discrete instants)
 */
class AbstractNode {
protected:
	/**
	 * Each node in a light cut tree is given a unique id between zero and the total number of nodes in the tree
	 */
	int nodeIdAndFlags;
	/**
	 * position of point or center of cluster's bounding box (or to-vector for infinite lights)
	 */
	float x;
	float y;
	float z;
	/**
	 * Intensity or strength of this point or cluster
	 */
	float intensityRed, intensityGreen, intensityBlue; //total intensity
	short startTime;
	short endTime;
	//start and end times represented in timeVector vector
	vector<float> * timeVector; //fractional intensity per time interval
	//**********STATIC DATA MEMBERS********************//
	//if startTime==endTime then the Intensity time is always the same and points to this one
	static const float singleTimeVector [];
	static const int ML_CLAMP ;//= 0x02; //subject to clamping?
	static const int ML_COMBO_MASK;// = 0xFC; //flags that should be shared with a parent
	static const int ML_MATCH_MASK;// = 0xFC;
	static const int ML_ID_SHIFT;// = 8;
	/**
	 * Number of representatives stored per node
	 */
	static int globalNumReps;// = -1;
	/**
	 * Does each representative represent a different time instant or are all at the same time instant
	 */
	static bool globalMultitime;// = false;
	static vector<vector<float> > *repRandomNums;
	//**********END STATIC DATA MEMBERS********************//

public:
	/**
	 * Creates a new instance of MLIntensity
	 */
	AbstractNode() {
		timeVector=NULL;
		startTime = -1; //deliberately invalid value
		nodeIdAndFlags = -1 << ML_ID_SHIFT; //must be set to a correct value later
	}

	short getStartTime(){
		return startTime;
	}
	short getEndTime(){
			return endTime;
		}
	/**
	 * Get the total intensity of this light or cluster
	 * as a scalar (ie averaged down to a single number)
	 *
	 * @return Scalar total intensity
	 */
	float getScalarTotalIntensity() const{
		return (1.0f / 3.0f) * (intensityRed + intensityGreen + intensityBlue);
	}

	/**
	 * What fraction of the total intensity was at the given time instant
	 */
	float getRelativeIntensity(int inTime) const{
		if (inTime < startTime || inTime > endTime) {
			return 0;
		}
		return (*timeVector)[inTime - startTime];
	}

	/**
	 * Set the Intensity of this node to be equal to the specified
	 * spectrum scaled by the specified factor and at the single instant specified
	 *
	 * @param inScaleFactor Factor to scale spectrum by before setting Intensity
	 */
	void setIntensity(double inScaleFactor, short inTime) {
		intensityRed = (float) inScaleFactor;
		intensityGreen = (float) inScaleFactor;
		intensityBlue = (float) inScaleFactor;
		if (inTime == -1) {
			inTime = 0;
		}
		if (inTime >= 0) {
			startTime = inTime;
			endTime = inTime;
			timeVector = new vector<float>(1);//singleTimeVector;
			(*timeVector)[0] = 1.0;
		} else {
			//negative value used as signal that should be uniform across all time
			int len = -inTime;
			startTime = 0;
			endTime = (short) (len - 1);
			timeVector = new vector<float>(len);
			for(int i=0;i<len;i++)
				(*timeVector)[i]= 1.0f / len;
			scaleIntensity(len);
		}
	}

	/**
	 * Set this nodes maximum intensity to be the sum of the maximum intensities
	 * of the two specified nodes.
	 *
	 * @param inNodeA Node to sum maximum intensity from
	 * @param inNodeB Node to sum maximum intensity from
	 */
	void setSummedIntensity(AbstractNode & inA, AbstractNode &inB) {
		intensityRed = inA.intensityRed + inB.intensityRed;
		intensityGreen = inA.intensityGreen + inB.intensityGreen;
		intensityBlue = inA.intensityBlue + inB.intensityBlue;
		startTime = inA.startTime < inB.startTime ? inA.startTime : inB.endTime;
		endTime = inA.startTime < inB.startTime ? inB.startTime : inA.endTime;
		if (startTime != endTime) {
			int len = endTime - startTime + 1;
			if (timeVector == NULL || ((int)timeVector->size()) < len) {
				timeVector = new vector<float>(len);
			} else {
				for (int i = 0; i < (int)timeVector->size(); i++) {
					(*timeVector)[i] = 0;
				}
			}
			float weightA = inA.getScalarTotalIntensity();
			float weightB = inB.getScalarTotalIntensity();
			float invDenom = 1.0f / (weightA + weightB);
			weightA *= invDenom;
			weightB *= invDenom;
			for (int i = inA.startTime; i <= inA.endTime; i++) {
				(*timeVector)[i - startTime] += weightA * (*inA.timeVector)[i - inA.startTime];
			}
			for (int i = inB.startTime; i <= inB.endTime; i++) {
				(*timeVector)[i - startTime] += weightB * (*inB.timeVector)[i - inB.startTime];
			}
		} else {
			timeVector = new vector<float>(1);
			(*timeVector)[0] = 1.0f;
		}
	}

	/**
	 * Scale the maximum intensity of this node by a given factor
	 *
	 * @param inScale Factor to scale maximum intensity by
	 */
	void scaleIntensity(double inScale) {
		float scale = (float) inScale;
		intensityRed *= scale;
		intensityGreen *= scale;
		intensityBlue *= scale;
	}

	static void setGlobalNumReps() {
		if (globalNumReps == 1) {
			//nothing changed
			return;
		}
		//trees must be rebuilt for this to take effect
		globalNumReps = 1;
		double inc = 1.0f / 1;
		repRandomNums = new vector<vector<float> >(256);
		for (int i = 0; i < (int)repRandomNums->size(); i++) {
			vector<float> * ranVec = new vector<float>(1);
			(*ranVec)[0] = (((float)rand()) / (numeric_limits<int>::max())) * inc;
			//now randomly permute the numbers
			(*repRandomNums)[i] = *ranVec;
		}
	}

	static void setGlobalMultitime() {
		//trees must be rebuilt for this to take effect
		globalMultitime = false;
	}

	float getX() {
		return x;
	}

	float getY() {
		return y;
	}

	float getZ() {
		return z;
	}

	void setCombinedFlags(AbstractNode &a, AbstractNode &b) {
		nodeIdAndFlags = (a.nodeIdAndFlags | b.nodeIdAndFlags) & ML_COMBO_MASK;
		nodeIdAndFlags |= a.nodeIdAndFlags & b.nodeIdAndFlags & ML_CLAMP;
		//clamp only if both children use clamping
		if ((a.nodeIdAndFlags & ML_MATCH_MASK) != (b.nodeIdAndFlags & ML_MATCH_MASK)) {
			assert(false && "Runtime Exception!");
		}
	}

	virtual bool isLeaf()=0;

	virtual int size()=0;
	friend std::ostream& operator<<(std::ostream& s, AbstractNode & a);
	bool equals(AbstractNode & other){
		if(x!=other.x)return false;
		if(y!=other.y)return false;
		if(z!=other.z)return false;
		if(intensityRed!=other.intensityRed)return false;
		if(intensityGreen!=other.intensityGreen)return false;
		if(intensityBlue!=other.intensityBlue)return false;
		if(startTime!=other.startTime)return false;
		if(startTime!=other.startTime)return false;
		//start and end times represented in timeVector vector
		return( timeVector==other.timeVector);
	}

};
std::ostream& operator<<(std::ostream& s, AbstractNode & a) {
	s << "AbsNode:: ID:" << a.nodeIdAndFlags << ",[" << a.x << "," << a.y
			<< "," << a.z << "], T:[" << a.startTime << "," << a.endTime << "]";
	s << "I: [" << a.intensityRed << "," << a.intensityGreen << ","
			<< a.intensityBlue << "]" << "\nTime Vector:";

	for (int i = 0; i < (int) a.timeVector->size(); i++)
		s << "" << (*a.timeVector)[i] << ",";
	return s;

}
const float AbstractNode::singleTimeVector []= { 1.0 };
const int AbstractNode::ML_CLAMP = 0x02; //subject to clamping?
const int AbstractNode::ML_COMBO_MASK = 0xFC; //flags that should be shared with a parent
const int AbstractNode::ML_MATCH_MASK = 0xFC;
const int AbstractNode::ML_ID_SHIFT = 8;
int AbstractNode::globalNumReps;// = -1;
bool AbstractNode::globalMultitime = false;
vector<vector<float> > *AbstractNode::repRandomNums = NULL;
#endif /* ABSTRACTNODE_H_ */
